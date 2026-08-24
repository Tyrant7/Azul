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
//! logging/resource-limit infrastructure are still pending.

pub mod format;
pub mod parsing;
pub mod protocol;

mod process;

use std::{io, path::PathBuf, sync::Arc, time::Duration};

use azul_movegen::GameState;
use clap::Parser;
use rand::{Rng, seq::IndexedRandom};

use crate::{
    format::ProtocolFormat,
    process::{EngineLaunch, EngineProcess, ProcessDiagnostics},
    protocol::{Cli, GameResult, Protocol, play_uai_game_with_recovery, uai_ready},
};

fn main() {
    let cli = Cli::parse();
    if !cli.quiet {
        println!("{:#?}", cli);
    }

    // Spawn configured engines; protocol dispatch remains separate from the human loop.

    let launches = cli
        .engines
        .iter()
        .map(|e| {
            let args: Vec<String> = e
                .args
                .clone()
                .map(|s| s.split_whitespace().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            EngineLaunch::new(e.path.clone(), args, e.dir.clone())
        })
        .collect::<Vec<_>>();
    let mut engines = launches
        .iter()
        .map(|launch| EngineProcess::spawn_launch(launch).expect("Failed to start engine"))
        .collect::<Vec<_>>();

    let diagnostics = if cli.debug || cli.stderr || cli.log {
        let log_path = cli.log.then(|| diagnostics_log_path(&cli.out));
        Some(Arc::new(
            ProcessDiagnostics::new(cli.stderr, cli.debug, log_path)
                .expect("Failed to initialize engine diagnostics"),
        ))
    } else {
        None
    };
    for (engine_index, engine) in engines.iter_mut().enumerate() {
        engine.configure_diagnostics(diagnostics.clone(), engine_index);
    }

    let startup_timeout = Duration::from_secs(cli.timeout as u64);
    for (player, (engine, config)) in engines.iter_mut().zip(&cli.engines).enumerate() {
        if matches!(config.proto, Protocol::UAI) {
            let mut restarts = 0;
            loop {
                let startup = protocol::uai_handshake(engine, startup_timeout)
                    .and_then(|identity| uai_ready(engine, startup_timeout).map(|_| identity));
                match startup {
                    Ok(identity) => {
                        if !cli.quiet {
                            println!(
                                "UAI engine: {} by {}",
                                identity.name.as_deref().unwrap_or("<unnamed>"),
                                identity.author.as_deref().unwrap_or("<unknown author>")
                            );
                        }
                        break;
                    }
                    Err(error)
                        if cli.recover
                            && restarts < 1
                            && startup_failure_is_recoverable(&error) =>
                    {
                        restarts += 1;
                        if let Err(restart_error) =
                            engine.restart(&launches[player], Duration::from_millis(100))
                        {
                            eprintln!(
                                "UAI engine {player} failed during startup and could not restart: {restart_error}"
                            );
                            return;
                        }
                    }
                    Err(error) => {
                        eprintln!("UAI engine {player} failed during startup: {error}");
                        return;
                    }
                }
            }
        }
    }

    if !cli.quiet {
        println!("started {} engine process(es)", engines.len());
    }

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
        match play_uai_game_with_recovery(
            &mut engines,
            &launches,
            gamestate,
            &time_controls,
            cli.recover,
            startup_timeout,
        ) {
            Ok(GameResult::Completed(gamestate)) => {
                if !cli.quiet {
                    println!("{}", gamestate.fmt_protocol(Protocol::Human));
                }
                println!("Game over");
                println!("Winner: player {}", gamestate.get_winner());
            }
            Ok(GameResult::Forfeit { game, failure }) => {
                if !cli.quiet {
                    println!("{}", game.fmt_protocol(Protocol::Human));
                }
                eprintln!(
                    "Player {} forfeited ({:?}): {}",
                    failure.player, failure.reason, failure.message
                );
                println!("Game over by forfeit");
            }
            Err(error) => eprintln!("UAI game failed: {error}"),
        }
    } else {
        let seed = cli.seed.unwrap_or_else(|| rand::rng().random());
        let mut gamestate = GameState::new(2, seed).expect("two-player game state must be valid");
        gamestate.setup_next_round();
        if !cli.quiet {
            println!("{}", gamestate.fmt_protocol(Protocol::Human));
        }
        listen_for_input(gamestate, Protocol::Human, cli.quiet);
    }

    for engine in engines {
        let _ = engine.shutdown(Duration::from_millis(100));
    }
}

/// Derives the default engine communication log path from the result path.
fn diagnostics_log_path(output_path: &str) -> PathBuf {
    let mut path = PathBuf::from(output_path);
    if path.as_os_str().is_empty() {
        path.push("azul-interface.log");
    } else {
        path.set_extension("log");
    }
    path
}

/// Identifies startup failures that may be fixed by replacing the process.
fn startup_failure_is_recoverable(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::UnexpectedEof | io::ErrorKind::TimedOut
    ) || error.to_string().starts_with("error ")
}

/// Runs the interactive human-input game loop.
fn listen_for_input(mut gamestate: GameState, protocol: Protocol, quiet: bool) {
    loop {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let input = input.trim();
        let choice = match protocol::parse_move(input) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Invalid move: {:?}", e);
                continue;
            }
        };
        if !quiet {
            println!("move: {:?}", choice);
        }

        match gamestate.make_move(&choice) {
            Err(_) => eprintln!("Illegal move"),
            Ok(_) if !quiet => println!("{}", gamestate.fmt_protocol(protocol)),
            Ok(_) => {}
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
