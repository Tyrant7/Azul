//! Neural-network definitions used by the reinforcement-learning trainer.

use tch::{
    Kind, Tensor,
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
    value_min: f64,
    value_max: f64,
    value_bins: i64,
    bin_width: f64,
    sigma: f64,
}

impl Module for ResNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.head.forward(&self.encoder.forward(xs))
    }
}

impl ResNetwork {
    /// Decodes categorical value logits into scalar expectations over the support.
    pub fn decode_values(&self, logits: &Tensor) -> Tensor {
        let atoms = self.support_atoms(logits.device());
        (logits.softmax(-1, Kind::Float) * atoms).sum_dim_intlist([-1].as_ref(), false, Kind::Float)
    }

    /// Runs the critic and decodes its categorical output into scalar values.
    pub fn values(&self, states: &Tensor) -> Tensor {
        let logits = <Self as Module>::forward(self, states);
        self.decode_values(&logits)
    }

    /// Computes cross-entropy against Gaussian-smoothed scalar return targets.
    pub fn hl_gauss_loss(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        let target_distribution = self.target_distribution(targets, logits.device());
        let log_probs = logits.log_softmax(-1, Kind::Float);
        -(target_distribution * log_probs)
            .sum_dim_intlist([-1].as_ref(), false, Kind::Float)
            .mean(Kind::Float)
    }

    /// Builds the normalized Gaussian mass assigned to each support bin.
    fn target_distribution(&self, targets: &Tensor, device: tch::Device) -> Tensor {
        let target = targets.clamp(self.value_min, self.value_max).unsqueeze(-1);
        let atoms = self.support_atoms(device);
        let lower = &atoms - self.bin_width / 2.0;
        let upper = &atoms + self.bin_width / 2.0;
        let scale = self.sigma * 2.0_f64.sqrt();
        let lower_cdf = (((&lower - &target) / scale).erf() + 1.0) * 0.5;
        let upper_cdf = (((&upper - &target) / scale).erf() + 1.0) * 0.5;
        let masses = (upper_cdf - lower_cdf).clamp_min(1e-12);
        &masses / masses.sum_dim_intlist([-1].as_ref(), true, Kind::Float)
    }

    /// Returns evenly spaced support atoms on the requested device.
    fn support_atoms(&self, device: tch::Device) -> Tensor {
        Tensor::linspace(
            self.value_min,
            self.value_max,
            self.value_bins,
            (Kind::Float, device),
        )
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

/// Builds a categorical value network whose expected support value estimates the current state value.
pub fn initialize_critic(
    vs: &nn::Path,
    value_bins: usize,
    value_min: f32,
    value_max: f32,
    sigma_ratio: f32,
) -> ResNetwork {
    assert!(value_bins >= 2);
    assert!(value_min.is_finite() && value_max.is_finite() && value_min < value_max);
    assert!(sigma_ratio.is_finite() && sigma_ratio > 0.0);
    let bin_width = (value_max as f64 - value_min as f64) / (value_bins - 1) as f64;
    ResNetwork {
        encoder: StateEncoder::new(vs),
        head: head_linear(vs / "head", HIDDEN, value_bins as i64),
        value_min: value_min as f64,
        value_max: value_max as f64,
        value_bins: value_bins as i64,
        bin_width,
        sigma: sigma_ratio as f64 * bin_width,
    }
}

#[cfg(test)]
mod tests {
    use super::initialize_critic;
    use crate::get_device;
    use tch::{Kind, Tensor, nn};

    #[test]
    fn hl_gauss_targets_are_normalized() {
        let var_store = nn::VarStore::new(get_device());
        let critic = initialize_critic(&var_store.root(), 5, -2.0, 2.0, 0.5);
        let targets = Tensor::from_slice(&[-2.0_f32, 0.0, 2.0]).to_device(get_device());
        let logits = Tensor::zeros([3, 5], (Kind::Float, get_device()));
        let distribution = critic.target_distribution(&targets, get_device());
        let loss = critic.hl_gauss_loss(&logits, &targets);

        assert!(loss.isfinite().all().int64_value(&[]) != 0);
        for row in 0..3 {
            let sum = distribution.get(row).sum(Kind::Float).double_value(&[]);
            assert!((sum - 1.0).abs() < 1e-6);
        }
        assert_eq!(distribution.argmax(-1, false).int64_value(&[1]), 2);
    }

    #[test]
    fn uniform_logits_decode_to_the_support_midpoint() {
        let var_store = nn::VarStore::new(get_device());
        let critic = initialize_critic(&var_store.root(), 5, -2.0, 2.0, 0.5);
        let logits = Tensor::zeros([1, 5], (Kind::Float, get_device()));
        let value = critic.decode_values(&logits).double_value(&[0]);

        assert!(value.abs() < 1e-6);
    }
}
