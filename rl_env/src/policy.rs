//! Inference policies and checkpoint persistence for Azul.

use std::path::Path;

use azul_movegen::{GameState, Move};
use tch::{Tensor, nn, no_grad};

use crate::get_device;
use crate::net::{ActionConditionedActor, ResNetwork, initialize_actor, initialize_critic};
use crate::{ACTION_FEATURE_SIZE, encode_state, legal_move_features};

/// A trained actor loaded for inference without PPO optimizer state.
pub struct ActorPolicy {
    var_store: nn::VarStore,
    actor: ActionConditionedActor,
}

/// A trained scalar critic loaded for inference.
pub struct CriticPolicy {
    _var_store: nn::VarStore,
    critic: ResNetwork,
}

impl CriticPolicy {
    /// Loads a scalar critic checkpoint.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, tch::TchError> {
        let mut var_store = nn::VarStore::new(get_device());
        let critic = initialize_critic(&var_store.root());
        var_store.load(path)?;
        Ok(Self {
            _var_store: var_store,
            critic,
        })
    }

    /// Evaluates a batch of player-relative states with the critic.
    pub fn values(&self, states: &Tensor) -> Tensor {
        no_grad(|| self.critic.values(states))
    }

    /// Evaluates one game state and returns its scalar critic value.
    pub fn value(&self, game: &GameState) -> f32 {
        self.values(&encode_state(game).unsqueeze(0))
            .double_value(&[0]) as f32
    }
}

impl ActorPolicy {
    /// Loads actor weights from a LibTorch var-store checkpoint.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, tch::TchError> {
        let mut var_store = nn::VarStore::new(get_device());
        let actor = initialize_actor(&var_store.root());
        load_actor_weights(&mut var_store, path.as_ref())?;
        Ok(Self { var_store, actor })
    }

    /// Saves this actor's weights to a LibTorch var-store checkpoint.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), tch::TchError> {
        self.var_store.save(path)
    }

    /// Scores a batch of states and candidate action features.
    pub fn forward(&self, states: &Tensor, action_features: &Tensor) -> Tensor {
        no_grad(|| self.actor.forward(states, action_features))
    }

    /// Selects the highest-scoring legal move for a two-player game state.
    pub fn choose_move(&self, game: &GameState) -> Option<Move> {
        if game.get_boards().len() != 2 {
            return None;
        }
        let legal_moves = legal_move_features(game);
        if legal_moves.is_empty() {
            return None;
        }
        let action_features: Vec<_> = legal_moves
            .iter()
            .flat_map(|(_, features)| features.iter().copied())
            .collect();
        let action_features = Tensor::from_slice(&action_features)
            .reshape([1, legal_moves.len() as i64, ACTION_FEATURE_SIZE as i64])
            .to_device(get_device());
        let logits = self.forward(&encode_state(game).unsqueeze(0), &action_features);
        let selected = logits.argmax(-1, false).int64_value(&[0]) as usize;
        Some(legal_moves[selected].0.clone())
    }
}

/// Loads actor weights while padding checkpoints from before spatial move features were added.
pub(crate) fn load_actor_weights(
    var_store: &mut nn::VarStore,
    path: &Path,
) -> Result<(), tch::TchError> {
    let checkpoint = tch::Tensor::load_multi_with_device(path, get_device())?
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();

    for (name, mut destination) in var_store.variables() {
        let source = checkpoint.get(&name).ok_or_else(|| {
            tch::TchError::Kind(format!("actor checkpoint is missing variable {name}"))
        })?;
        let destination_shape = destination.size();
        let source_shape = source.size();
        let weights = if name == "action_input.weight"
            && source_shape.len() == 2
            && destination_shape.len() == 2
            && source_shape[0] == destination_shape[0]
            && source_shape[1] < destination_shape[1]
        {
            let padded = tch::Tensor::zeros(&destination_shape, (source.kind(), get_device()));
            padded.narrow(1, 0, source_shape[1]).copy_(source);
            padded
        } else {
            if source_shape != destination_shape {
                return Err(tch::TchError::Kind(format!(
                    "actor variable {name} has checkpoint shape {source_shape:?}, expected {destination_shape:?}"
                )));
            }
            source.shallow_clone()
        };
        tch::no_grad(|| destination.copy_(&weights));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ActorPolicy;
    use azul_movegen::GameState;
    use tch::nn;

    #[test]
    fn chooses_a_legal_move_from_an_initial_game() {
        let var_store = nn::VarStore::new(crate::get_device());
        let actor = crate::net::initialize_actor(&var_store.root());
        let policy = ActorPolicy { var_store, actor };
        let mut game = GameState::new(2, 1).expect("game should initialize");
        game.setup_next_round();

        let selected = policy
            .choose_move(&game)
            .expect("initial game has legal moves");
        assert!(game.get_valid_moves().contains(&selected));
    }
}
