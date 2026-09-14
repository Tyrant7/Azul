//! Policy-prior beam search for the Azul reinforcement-learning actor.

use azul_movegen::{GameState, Move};
use rl_env::{ACTION_FEATURE_SIZE, ActorPolicy, encode_state, get_device, legal_move_features};
use tch::{Kind, Tensor};

/// Search settings for the policy-prior beam engine.
#[derive(Debug, Clone, Copy)]
pub struct BeamConfig {
    /// Number of partial lines retained after each search ply.
    pub beam_width: usize,
    /// Number of plies, including the root move, explored by the search.
    pub depth: usize,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            depth: 2,
        }
    }
}

impl BeamConfig {
    fn validate(self) {
        assert!(self.beam_width > 0, "beam width must be positive");
        assert!(self.depth > 0, "beam depth must be positive");
    }
}

/// Actor-backed beam-search policy that returns the first move of the best line.
pub struct BeamPolicy {
    actor: ActorPolicy,
    config: BeamConfig,
}

impl BeamPolicy {
    /// Loads an actor checkpoint with the requested beam-search settings.
    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
        config: BeamConfig,
    ) -> Result<Self, tch::TchError> {
        config.validate();
        Ok(Self {
            actor: ActorPolicy::load(path)?,
            config,
        })
    }

    /// Selects a legal move by retaining the highest-probability partial lines.
    pub fn choose_move(&self, game: &GameState) -> Option<Move> {
        if game.is_game_over() {
            return None;
        }

        let root_moves = scored_moves(&self.actor, game);
        if root_moves.is_empty() {
            return None;
        }

        let mut beam = root_moves
            .into_iter()
            .filter_map(|(first_move, score)| {
                let mut state = game.clone();
                state.make_move(&first_move).ok()?;
                advance_round(&mut state);
                Some(BeamNode {
                    state,
                    first_move,
                    score,
                })
            })
            .collect::<Vec<_>>();
        retain_best(&mut beam, self.config.beam_width);

        for _ in 1..self.config.depth {
            let mut expanded = Vec::new();
            for node in beam.drain(..) {
                if node.state.is_game_over() {
                    expanded.push(node);
                    continue;
                }
                for (next_move, move_score) in scored_moves(&self.actor, &node.state) {
                    let mut state = node.state.clone();
                    if state.make_move(&next_move).is_err() {
                        continue;
                    }
                    advance_round(&mut state);
                    expanded.push(BeamNode {
                        state,
                        first_move: node.first_move.clone(),
                        score: node.score + move_score,
                    });
                }
            }
            if expanded.is_empty() {
                break;
            }
            retain_best(&mut expanded, self.config.beam_width);
            beam = expanded;
        }

        beam.into_iter()
            .max_by(|left, right| left.score.total_cmp(&right.score))
            .map(|node| node.first_move)
    }
}

/// One partial game line retained by the beam.
struct BeamNode {
    state: GameState,
    first_move: Move,
    score: f32,
}

/// Scores legal moves with the actor's masked log-probabilities.
fn scored_moves(actor: &ActorPolicy, game: &GameState) -> Vec<(Move, f32)> {
    let legal_moves = legal_move_features(game);
    if legal_moves.is_empty() {
        return Vec::new();
    }
    let action_values: Vec<_> = legal_moves
        .iter()
        .flat_map(|(_, features)| features.iter().copied())
        .collect();
    let action_features = Tensor::from_slice(&action_values)
        .reshape([1, legal_moves.len() as i64, ACTION_FEATURE_SIZE as i64])
        .to_device(get_device());
    let logits = actor.forward(&encode_state(game).unsqueeze(0), &action_features);
    let log_probs = logits.log_softmax(-1, Kind::Float).squeeze_dim(0);

    legal_moves
        .into_iter()
        .enumerate()
        .map(|(index, (choice, _))| (choice, log_probs.double_value(&[index as i64]) as f32))
        .collect()
}

/// Keeps the highest-scoring partial lines in descending score order.
fn retain_best(nodes: &mut Vec<BeamNode>, width: usize) {
    nodes.sort_unstable_by(|left, right| right.score.total_cmp(&left.score));
    nodes.truncate(width);
}

/// Advances a simulated position past a completed round before deeper search.
fn advance_round(state: &mut GameState) {
    if state.round_over() && !state.is_game_over() {
        state.setup_next_round();
    }
}
