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
//! child processes, perform the UAI startup sequence, enforce per-engine time
//! controls, and run a local UAI game or human-input game. Tournaments and
//! recovery are still pending.

pub mod format;
pub mod parsing;
pub mod protocol;

mod process;

use std::{io, process::Command, time::Duration};

use azul_movegen::GameState;
use clap::Parser;
use rand::{Rng, seq::IndexedRandom};

use crate::{
    format::ProtocolFormat,
    process::EngineProcess,
    protocol::{Cli, Protocol, play_uai_game, uai_ready},
};

fn main() {
    let cli = Cli::parse();
    println!("{:#?}", cli);

    // Spawn configured engines; protocol dispatch remains separate from the human loop.

    let mut engines = cli
        .engines
        .iter()
        .map(|e| {
            let args: Vec<String> = e
                .args
                .clone()
                .map(|s| s.split_whitespace().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            let mut command = Command::new(&e.path);
            command.args(args);
            if let Some(dir) = &e.dir {
                command.current_dir(dir);
            }
            EngineProcess::spawn(&mut command).expect("Failed to start engine")
        })
        .collect::<Vec<_>>();

    let startup_timeout = Duration::from_secs(cli.timeout as u64);
    for (engine, config) in engines.iter_mut().zip(&cli.engines) {
        if matches!(config.proto, Protocol::UAI) {
            let identity = protocol::uai_handshake(engine, startup_timeout)
                .expect("UAI engine handshake failed");
            uai_ready(engine, startup_timeout).expect("UAI engine readiness check failed");
            println!(
                "UAI engine: {} by {}",
                identity.name.as_deref().unwrap_or("<unnamed>"),
                identity.author.as_deref().unwrap_or("<unknown author>")
            );
        }
    }

    println!("started {} engine process(es)", engines.len());

    let all_uai = cli
        .engines
        .iter()
        .all(|config| matches!(config.proto, Protocol::UAI));
    if all_uai && (2..=4).contains(&engines.len()) {
        let seed = cli.seed.unwrap_or_else(|| rand::rng().random());
        let mut gamestate =
            GameState::new(engines.len(), seed).expect("supported player count must be valid");
        gamestate.setup_next_round();
        let time_controls = cli
            .engines
            .iter()
            .map(|config| {
                config
                    .tc
                    .clone()
                    .expect("engine time control must be configured")
            })
            .collect::<Vec<_>>();
        match play_uai_game(&mut engines, gamestate, &time_controls) {
            Ok(gamestate) => {
                println!("{}", gamestate.fmt_protocol(Protocol::Human));
                println!("Game over");
                println!("Winner: player {}", gamestate.get_winner());
            }
            Err(error) => eprintln!("UAI game failed: {error}"),
        }
    } else {
        let seed = cli.seed.unwrap_or_else(|| rand::rng().random());
        let mut gamestate = GameState::new(2, seed).expect("two-player game state must be valid");
        gamestate.setup_next_round();
        println!("{}", gamestate.fmt_protocol(Protocol::Human));
        listen_for_input(gamestate, Protocol::Human);
    }

    for engine in engines {
        let _ = engine.shutdown(Duration::from_millis(100));
    }
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
