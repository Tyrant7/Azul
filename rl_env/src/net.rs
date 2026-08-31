use tch::{
    Tensor,
    nn::{self, Module},
};

use crate::{ACTION_SPACE_SIZE, OBSERVATION_SIZE};

const INPUT_SIZE: i64 = OBSERVATION_SIZE as i64;
const OUTPUT_SIZE: i64 = ACTION_SPACE_SIZE as i64;
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

#[derive(Debug)]
pub struct ResNetwork {
    input: nn::Linear,
    blocks: Vec<ResBlock>,
    head: nn::Linear,
}

impl Module for ResNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = self.input.forward(xs).elu();
        for block in &self.blocks {
            xs = block.forward(&xs);
        }
        self.head.forward(&xs)
    }
}

pub fn initialize_actor(vs: &nn::Path) -> ResNetwork {
    let mut blocks = Vec::with_capacity(NUM_BLOCKS);
    blocks.push(ResBlock::new(&(vs / "block0"), HIDDEN, HIDDEN));
    for i in 0..NUM_BLOCKS - 1 {
        blocks.push(ResBlock::new(&(vs / format!("block{i}")), HIDDEN, HIDDEN));
    }

    ResNetwork {
        input: hidden_linear(vs / "input", INPUT_SIZE, HIDDEN),
        blocks,
        head: head_linear(vs / "head", HIDDEN, OUTPUT_SIZE),
    }
}

pub fn initialize_critic(vs: &nn::Path) -> ResNetwork {
    let mut blocks = Vec::with_capacity(NUM_BLOCKS);
    blocks.push(ResBlock::new(&(vs / "block0"), HIDDEN, HIDDEN));
    for i in 0..NUM_BLOCKS - 1 {
        blocks.push(ResBlock::new(&(vs / format!("block{i}")), HIDDEN, HIDDEN));
    }

    ResNetwork {
        input: hidden_linear(vs / "input", INPUT_SIZE, HIDDEN),
        blocks,
        head: head_linear(vs / "head", HIDDEN, 1),
    }
}
