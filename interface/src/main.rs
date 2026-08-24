//! Command-line interface and protocol support for Azul engines.
//!
//! This crate sits above [`azul_movegen`]. It parses engine and match
//! configuration, formats game state for humans or machine clients, converts
//! game state to and from AzulFEN, and parses the six-digit move notation used
//! by the [current UAI draft](../protocol.md).
//!
//! The [`protocol`] module owns CLI configuration, protocol and time-control
//! types, and move parsing. The [`mod@format`] module renders movegen values,
//! while [`mod@parsing`] handles AzulFEN components and complete game states.
//!
//! The executable is currently a development harness: it can spawn configured
//! child processes and run a local human-input game, but full UAI message
//! dispatch, engine I/O, tournaments, timing, and recovery are still pending.

pub mod format;
pub mod parsing;
pub mod protocol;

use std::{io, process::Command};

use azul_movegen::GameState;
use clap::Parser;
use rand::{Rng, seq::IndexedRandom};

use crate::{
    format::ProtocolFormat,
    protocol::{Cli, Protocol},
};

fn main() {
    let cli = Cli::parse();
    println!("{:#?}", cli);

    // Spawn configured engines; their I/O is not yet connected to the game loop.

    let engines = cli
        .engines
        .iter()
        .map(|e| {
            let args: Vec<String> = e
                .args
                .clone()
                .map(|s| s.split_whitespace().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            Command::new(&e.path)
                .args(args)
                .spawn()
                .expect("Failed to start engine")
        })
        .collect::<Vec<_>>();

    for eng in engines {
        println!("{:?}", eng);
    }

    let seed = rand::rng().random();
    let mut gamestate = GameState::new(2, seed).expect("two-player game state must be valid");
    gamestate.setup_next_round();
    println!("{}", gamestate.fmt_protocol(Protocol::Human));

    listen_for_input(gamestate, Protocol::Human);
}

/// Runs the interactive human-input game loop.
fn listen_for_input(mut gamestate: GameState, protocol: Protocol) {
    loop {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let input = input.trim();
        let choice = match protocol::parse_move(input) {
            Ok(m) => m,
            Err(e) => {
                println!("Invalid move: {:?}", e);
                continue;
            }
        };
        println!("move: {:?}", choice);

        match gamestate.make_move(&choice) {
            Err(_) => println!("Illegal move"),
            Ok(_) => println!("{}", gamestate.fmt_protocol(protocol)),
        };

        if gamestate.round_over() {
            gamestate.setup_next_round();
        }

        if gamestate.is_game_over() {
            break;
        }
    }
    println!("Game over");
    println!("Winner: player {}", gamestate.get_winner());
}

/// Runs a random-playout loop for the supplied game state.
fn random_playout(mut gamestate: GameState, protocol: Protocol) {
    loop {
        io::stdin()
            .read_line(&mut String::new())
            .expect("Failed to read input");

        let moves = gamestate.get_valid_moves();
        let selection = moves.choose(&mut rand::rng()).cloned().unwrap_or_default();
        println!("selection: {:?}", selection);

        match gamestate.make_move(&selection) {
            Err(_) => println!("Illegal move"),
            Ok(_) => println!("{}", gamestate.fmt_protocol(protocol)),
        };

        if gamestate.round_over() {
            gamestate.setup_next_round();
        }

        if gamestate.is_game_over() {
            break;
        }
    }
    println!("Game over");
    println!("Winner: player {}", gamestate.get_winner());
}
