//! Inference policies and checkpoint persistence for Azul.

use std::path::Path;

use azul_movegen::{GameState, Move};
use tch::{Tensor, nn, no_grad};

use crate::get_device;
use crate::net::{ActionConditionedActor, initialize_actor};
use crate::{ACTION_FEATURE_SIZE, encode_state, legal_move_features};

/// A trained actor loaded for inference without PPO optimizer state.
pub struct ActorPolicy {
    var_store: nn::VarStore,
    actor: ActionConditionedActor,
}

impl ActorPolicy {
    /// Loads actor weights from a LibTorch var-store checkpoint.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, tch::TchError> {
        let mut var_store = nn::VarStore::new(get_device());
        let actor = initialize_actor(&var_store.root());
        var_store.load(path)?;
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
