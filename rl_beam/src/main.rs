use std::{env, io};

use azul_movegen::{GameState, Move};
use interface::engine::{Engine, SearchTime, run_engine};
use rl_beam::{BeamConfig, BeamPolicy};

/// Exposes the beam-search policy through the project's UAI runtime.
struct RlBeamEngine {
    policy: BeamPolicy,
}

impl Engine for RlBeamEngine {
    fn name(&self) -> &str {
        "Azul PPO Beam"
    }

    fn author(&self) -> &str {
        "Azul project"
    }

    fn choose_move(&mut self, game: &GameState, _time: SearchTime) -> io::Result<Move> {
        self.policy
            .choose_move(game)
            .ok_or_else(|| io::Error::other("beam policy could not select a legal move"))
    }
}

fn main() -> io::Result<()> {
    let mut arguments = env::args().skip(1);
    let checkpoint = arguments
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing actor checkpoint"))?;
    let beam_width = parse_argument(
        arguments.next(),
        "beam width",
        BeamConfig::default().beam_width,
    )?;
    let depth = parse_argument(arguments.next(), "beam depth", BeamConfig::default().depth)?;
    let critic_checkpoint = arguments
        .next()
        .unwrap_or_else(|| "checkpoints/azul_critic.ot".to_owned());
    let policy = BeamPolicy::load(
        &checkpoint,
        critic_checkpoint,
        BeamConfig { beam_width, depth },
    )
    .map_err(|error| io::Error::other(format!("failed to load beam checkpoints: {error}")))?;
    run_engine(RlBeamEngine { policy })
}

/// Parses an optional positive numeric engine argument.
fn parse_argument(value: Option<String>, name: &str, default: usize) -> io::Result<usize> {
    value
        .map(|value| {
            value.parse::<usize>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid {name}: {value}"),
                )
            })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}
