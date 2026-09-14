//! Critic-guided beam search for the Azul reinforcement-learning reference policy.

use azul_movegen::{GameState, Move};
use rl_env::{
    ACTION_FEATURE_SIZE, ActorPolicy, CriticPolicy, encode_state, get_device, legal_move_features,
};
use tch::{Device, Kind, Tensor};

/// Search settings for the critic-guided beam engine.
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
            depth: 8,
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
    critic: CriticPolicy,
    config: BeamConfig,
}

impl BeamPolicy {
    /// Loads matching actor and critic checkpoints with the requested settings.
    pub fn load<P: AsRef<std::path::Path>, C: AsRef<std::path::Path>>(
        actor_path: P,
        critic_path: C,
        config: BeamConfig,
    ) -> Result<Self, tch::TchError> {
        config.validate();
        Ok(Self {
            actor: ActorPolicy::load(actor_path)?,
            critic: CriticPolicy::load(critic_path)?,
            config,
        })
    }

    /// Selects a legal move by retaining the highest-probability partial lines.
    pub fn choose_move(&self, game: &GameState) -> Option<Move> {
        if game.is_game_over() {
            return None;
        }

        let root_player = game.get_active_player();
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
                    policy_score: score,
                    decision_player: root_player,
                    value: 0.0,
                })
            })
            .collect::<Vec<_>>();
        evaluate_nodes(&self.critic, &mut beam, root_player);
        retain_best(&mut beam, self.config.beam_width, root_player);

        for _ in 1..self.config.depth {
            let mut expanded = Vec::new();
            for node in beam.drain(..) {
                if node.state.is_game_over() {
                    expanded.push(node);
                    continue;
                }
                let decision_player = node.state.get_active_player();
                let mut children = Vec::new();
                for (next_move, move_score) in scored_moves(&self.actor, &node.state) {
                    let mut state = node.state.clone();
                    if state.make_move(&next_move).is_err() {
                        continue;
                    }
                    advance_round(&mut state);
                    let score_sign = if decision_player == root_player {
                        1.0
                    } else {
                        -1.0
                    };
                    children.push(BeamNode {
                        state,
                        first_move: node.first_move.clone(),
                        policy_score: node.policy_score + score_sign * move_score,
                        decision_player,
                        value: 0.0,
                    });
                }
                evaluate_nodes(&self.critic, &mut children, root_player);
                let branch_width = if decision_player == root_player {
                    self.config.beam_width
                } else {
                    1
                };
                retain_best(&mut children, branch_width, root_player);
                expanded.extend(children);
            }
            if expanded.is_empty() {
                break;
            }
            retain_best(&mut expanded, self.config.beam_width, root_player);
            beam = expanded;
        }

        beam.into_iter()
            .max_by(|left, right| {
                left.value
                    .total_cmp(&right.value)
                    .then_with(|| left.policy_score.total_cmp(&right.policy_score))
            })
            .map(|node| node.first_move)
    }
}

/// One partial game line retained by the beam.
struct BeamNode {
    state: GameState,
    first_move: Move,
    policy_score: f32,
    decision_player: usize,
    value: f32,
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
fn retain_best(nodes: &mut Vec<BeamNode>, width: usize, root_player: usize) {
    nodes.sort_unstable_by(|left, right| {
        let value_ordering = if left.decision_player == root_player {
            right.value.total_cmp(&left.value)
        } else {
            left.value.total_cmp(&right.value)
        };
        value_ordering.then_with(|| right.policy_score.total_cmp(&left.policy_score))
    });
    nodes.truncate(width);
}

/// Evaluates search leaves from the root player's perspective in one batch.
fn evaluate_nodes(critic: &CriticPolicy, nodes: &mut [BeamNode], root_player: usize) {
    let mut nonterminal_indices = Vec::new();
    let mut states = Vec::new();
    for index in 0..nodes.len() {
        if nodes[index].state.is_game_over() {
            nodes[index].value = if nodes[index].state.get_winner() == root_player {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        } else {
            nonterminal_indices.push(index);
            states.push(encode_state(&nodes[index].state));
        }
    }
    if states.is_empty() {
        return;
    }

    let values = critic
        .values(&Tensor::stack(&states, 0))
        .to_device(Device::Cpu);
    let values: Vec<f32> = Vec::try_from(&values)
        .expect("batched critic values should convert to a one-dimensional vector");
    for (index, value) in nonterminal_indices.into_iter().zip(values) {
        let active_player = nodes[index].state.get_active_player();
        nodes[index].value = if active_player == root_player {
            value
        } else {
            -value
        };
    }
}

/// Advances a simulated position past a completed round before deeper search.
fn advance_round(state: &mut GameState) {
    if state.round_over() && !state.is_game_over() {
        state.setup_next_round();
    }
}
