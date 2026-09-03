use std::{env, io};

use azul_movegen::{GameState, Move};
use interface::engine::{Engine, SearchTime};
use rl_env::ActorPolicy;

struct RlEngine {
    policy: ActorPolicy,
}

impl Engine for RlEngine {
    fn name(&self) -> &str {
        "Azul PPO"
    }

    fn author(&self) -> &str {
        "Azul project"
    }

    fn choose_move(&mut self, game: &GameState, _time: SearchTime) -> io::Result<Move> {
        self.policy
            .choose_move(game)
            .ok_or_else(|| io::Error::other("policy could not select a legal move"))
    }
}

fn main() -> io::Result<()> {
    let checkpoint = env::args()
        .nth(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing actor checkpoint"))?;
    let policy = ActorPolicy::load(&checkpoint)
        .map_err(|error| io::Error::other(format!("failed to load actor checkpoint: {error}")))?;
    interface::engine::run_engine(RlEngine { policy })
}
