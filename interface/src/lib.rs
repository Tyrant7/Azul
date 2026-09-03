//! Reusable Azul interface protocols and serialization.

use std::{io, path::PathBuf, sync::Arc, time::Duration};

use azul_movegen::GameState;
use clap::Parser;
use rand::Rng;

pub mod engine;
pub mod format;
pub mod parsing;
pub mod process;
pub mod protocol;
pub mod resource;

pub use format::ProtocolFormat;

pub fn run() {
    let cli = crate::protocol::Cli::parse();
    if !cli.quiet {
        println!("{:#?}", cli);
    }

    let launches = cli
        .engines
        .iter()
        .filter(|e| matches!(e.proto, protocol::Protocol::UAI))
        .map(|e| {
            let args: Vec<String> = e
                .args
                .clone()
                .map(|s| s.split_whitespace().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            process::EngineLaunch::new(e.path.clone(), args, e.dir.clone())
                .with_limits(e.limit_mem, e.limit_threads)
                .expect("invalid engine resource limits")
        })
        .collect::<Vec<_>>();
    let mut engines = launches
        .iter()
        .map(|launch| process::EngineProcess::spawn_launch(launch).expect("Failed to start engine"))
        .collect::<Vec<_>>();

    let diagnostics = if cli.debug || cli.stderr || cli.log {
        let log_path = cli.log.then(|| diagnostics_log_path(&cli.out));
        Some(Arc::new(
            process::ProcessDiagnostics::new(cli.stderr, cli.debug, log_path)
                .expect("Failed to initialize engine diagnostics"),
        ))
    } else {
        None
    };
    for (engine_index, engine) in engines.iter_mut().enumerate() {
        engine.configure_diagnostics(diagnostics.clone(), engine_index);
    }

    let startup_timeout = Duration::from_secs(cli.timeout as u64);
    let mut uai_index = 0;
    for config in &cli.engines {
        if matches!(config.proto, protocol::Protocol::UAI) {
            let player = uai_index;
            let engine = &mut engines[player];
            let mut restarts = 0;
            loop {
                let startup =
                    protocol::uai_handshake(engine, startup_timeout).and_then(|identity| {
                        protocol::uai_ready(engine, startup_timeout).map(|_| identity)
                    });
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
            uai_index += 1;
        }
    }

    if !cli.quiet {
        println!("started {} engine process(es)", engines.len());
    }

    let all_uai = cli
        .engines
        .iter()
        .all(|config| matches!(config.proto, protocol::Protocol::UAI));
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
        match protocol::play_uai_game(
            &mut engines,
            &launches,
            gamestate,
            &time_controls,
            cli.recover,
            startup_timeout,
        ) {
            Ok(protocol::GameResult::Completed(gamestate)) => {
                if !cli.quiet {
                    println!("{}", gamestate.fmt_protocol(protocol::Protocol::Human));
                }
                println!("Game over");
                println!("Winner: player {}", gamestate.get_winner());
            }
            Ok(protocol::GameResult::Forfeit { game, failure }) => {
                if !cli.quiet {
                    println!("{}", game.fmt_protocol(protocol::Protocol::Human));
                }
                eprintln!(
                    "Player {} forfeited ({:?}): {}",
                    failure.player, failure.reason, failure.message
                );
                println!("Game over by forfeit");
            }
            Err(error) => eprintln!("UAI game failed: {error}"),
        }
    } else if cli.engines.len() == 2
        && cli
            .engines
            .iter()
            .filter(|config| matches!(config.proto, protocol::Protocol::Human))
            .count()
            == 1
    {
        let human_player = cli
            .engines
            .iter()
            .position(|config| matches!(config.proto, protocol::Protocol::Human))
            .expect("mixed game must have a human player");
        let engine_config = cli
            .engines
            .iter()
            .find(|config| matches!(config.proto, protocol::Protocol::UAI))
            .expect("mixed game must have a UAI engine");
        let seed = cli.seed.unwrap_or_else(|| rand::rng().random());
        let mut gamestate = GameState::new(2, seed).expect("two-player game state must be valid");
        gamestate.setup_next_round();
        match protocol::play_human_uai_game(
            engines
                .first_mut()
                .expect("mixed game must have one UAI engine"),
            launches
                .first()
                .expect("mixed game must have one UAI launch"),
            gamestate,
            human_player,
            engine_config
                .tc
                .clone()
                .expect("engine time control must be configured"),
            cli.recover,
            startup_timeout,
        ) {
            Ok(protocol::GameResult::Completed(gamestate)) => {
                if !cli.quiet {
                    println!("{}", gamestate.fmt_protocol(protocol::Protocol::Human));
                }
                println!("Game over");
                println!("Winner: player {}", gamestate.get_winner());
            }
            Ok(protocol::GameResult::Forfeit { game, failure }) => {
                if !cli.quiet {
                    println!("{}", game.fmt_protocol(protocol::Protocol::Human));
                }
                eprintln!(
                    "Player {} forfeited ({:?}): {}",
                    failure.player, failure.reason, failure.message
                );
                println!("Game over by forfeit");
            }
            Err(error) => eprintln!("Mixed game failed: {error}"),
        }
    } else {
        let seed = cli.seed.unwrap_or_else(|| rand::rng().random());
        let mut gamestate = GameState::new(2, seed).expect("two-player game state must be valid");
        gamestate.setup_next_round();
        if !cli.quiet {
            println!("{}", gamestate.fmt_protocol(protocol::Protocol::Human));
        }
        listen_for_input(gamestate, protocol::Protocol::Human, cli.quiet);
    }

    for engine in engines {
        let _ = engine.shutdown(Duration::from_millis(100));
    }
}

fn diagnostics_log_path(output_path: &str) -> PathBuf {
    let mut path = PathBuf::from(output_path);
    if path.as_os_str().is_empty() {
        path.push("azul-interface.log");
    } else {
        path.set_extension("log");
    }
    path
}

fn startup_failure_is_recoverable(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::UnexpectedEof | io::ErrorKind::TimedOut
    ) || error.to_string().starts_with("error ")
}

fn listen_for_input(mut gamestate: GameState, protocol: protocol::Protocol, quiet: bool) {
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
