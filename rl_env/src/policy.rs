//! Inference policies and checkpoint persistence for Azul.

use std::path::Path;

use tch::{Tensor, nn, no_grad};

use crate::get_device;
use crate::net::{ActionConditionedActor, initialize_actor};

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
}
