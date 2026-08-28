use std::io;

use azul_movegen::GameState;
use interface::engine::{Engine, SearchTime, run_engine};
use rand::prelude::IndexedRandom;

fn main() {
    run_engine(RandomEngine).unwrap_or_else(|error| eprintln!("engine error: {error}"));
}

/// Selects uniformly from the legal moves in the current Azul position.
struct RandomEngine;

impl Engine for RandomEngine {
    fn name(&self) -> &str {
        "Azul Random Engine"
    }

    fn author(&self) -> &str {
        "Azul contributors"
    }

    fn choose_move(
        &mut self,
        game: &GameState,
        _time: SearchTime,
    ) -> io::Result<azul_movegen::Move> {
        game.get_valid_moves()
            .choose(&mut rand::rng())
            .cloned()
            .ok_or_else(|| io::Error::other("position has no legal moves"))
    }
}
