//! Neural-network definitions used by the reinforcement-learning trainer.

use tch::{
    Tensor,
    nn::{self, Module},
};

use crate::{ACTION_FEATURE_SIZE, OBSERVATION_SIZE};

const INPUT_SIZE: i64 = OBSERVATION_SIZE as i64;
const ACTION_FEATURES: i64 = ACTION_FEATURE_SIZE as i64;
const HIDDEN: i64 = 256;
const NUM_BLOCKS: usize = 4;

fn linear(vs: nn::Path, in_dim: i64, out_dim: i64, ws_init: nn::Init) -> nn::Linear {
    nn::linear(
        vs,
        in_dim,
        out_dim,
        nn::LinearConfig {
            ws_init,
            bs_init: Some(nn::Init::Const(0.0)),
            bias: true,
        },
    )
}

fn hidden_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(vs, in_dim, out_dim, nn::init::DEFAULT_KAIMING_NORMAL)
}

fn scaled_linear(vs: nn::Path, in_dim: i64, out_dim: i64, scale: f64) -> nn::Linear {
    linear(
        vs,
        in_dim,
        out_dim,
        nn::Init::Randn {
            mean: 0.0,
            stdev: scale,
        },
    )
}

fn head_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(
        vs,
        in_dim,
        out_dim,
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.01,
        },
    )
}

#[derive(Debug)]
struct ResBlock {
    fc1: nn::Linear,
    fc2: nn::Linear,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}

impl ResBlock {
    fn new(vs: &nn::Path, in_dim: i64, dim: i64) -> Self {
        Self {
            fc1: hidden_linear(vs / "fc1", in_dim, dim),
            fc2: scaled_linear(vs / "fc2", dim, dim, 1. / NUM_BLOCKS as f64),
            norm1: nn::layer_norm(vs / "norm1", vec![in_dim], Default::default()),
            norm2: nn::layer_norm(vs / "norm2", vec![dim], Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let residual = xs;
        let out = xs
            .apply(&self.norm1)
            .apply(&self.fc1)
            .elu()
            .apply(&self.norm2)
            .apply(&self.fc2);
        out + residual
    }
}

/// Encodes a state observation into the shared hidden representation.
#[derive(Debug)]
struct StateEncoder {
    input: nn::Linear,
    blocks: Vec<ResBlock>,
}

impl StateEncoder {
    fn new(vs: &nn::Path) -> Self {
        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            blocks.push(ResBlock::new(&(vs / format!("block{i}")), HIDDEN, HIDDEN));
        }

        Self {
            input: hidden_linear(vs / "input", INPUT_SIZE, HIDDEN),
            blocks,
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = self.input.forward(xs).elu();
        for block in &self.blocks {
            xs = block.forward(&xs);
        }
        xs
    }
}

/// Residual multilayer perceptron used for state-value estimation.
#[derive(Debug)]
pub struct ResNetwork {
    encoder: StateEncoder,
    head: nn::Linear,
}

impl Module for ResNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.head.forward(&self.encoder.forward(xs))
    }
}

/// Scores candidate actions conditioned on a shared state representation.
#[derive(Debug)]
pub struct ActionConditionedActor {
    encoder: StateEncoder,
    action_input: nn::Linear,
    action_output: nn::Linear,
}

impl ActionConditionedActor {
    /// Returns one unnormalized logit for every candidate action.
    pub fn forward(&self, states: &Tensor, action_features: &Tensor) -> Tensor {
        let state_features = self.encoder.forward(states);
        let state_size = state_features.size();
        let action_size = action_features.size();
        assert_eq!(
            state_size.len(),
            2,
            "states must have shape [batch, features]"
        );
        assert_eq!(
            action_size.len(),
            3,
            "action features must have shape [batch, candidates, features]"
        );
        assert_eq!(state_size[0], action_size[0]);
        assert_eq!(action_size[2], ACTION_FEATURES);

        let batch_size = action_size[0];
        let candidate_count = action_size[1];
        let expanded_states = state_features
            .unsqueeze(1)
            .expand([batch_size, candidate_count, HIDDEN], true);
        let inputs = Tensor::cat(&[expanded_states, action_features.shallow_clone()], -1)
            .view([-1, HIDDEN + ACTION_FEATURES]);
        self.action_output
            .forward(&self.action_input.forward(&inputs).elu())
            .view([batch_size, candidate_count])
    }
}

/// Builds an actor that scores each candidate move from the current state.
pub fn initialize_actor(vs: &nn::Path) -> ActionConditionedActor {
    ActionConditionedActor {
        encoder: StateEncoder::new(&(vs / "state")),
        action_input: hidden_linear(vs / "action_input", HIDDEN + ACTION_FEATURES, HIDDEN),
        action_output: head_linear(vs / "action_output", HIDDEN, 1),
    }
}

/// Builds a value network whose output estimates the current state value.
pub fn initialize_critic(vs: &nn::Path) -> ResNetwork {
    ResNetwork {
        encoder: StateEncoder::new(vs),
        head: head_linear(vs / "head", HIDDEN, 1),
    }
}
