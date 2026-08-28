//! Reusable UAI runtime for engines that supply only move-selection policy.

use std::io::{self, BufRead, Write};
use std::time::Duration;

use azul_movegen::{GameState, Move};

use crate::parsing::FromAzulFEN;

/// Search-time information supplied to an engine for one move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchTime {
    /// The engine may search for at most this fixed duration.
    Fixed { limit: Duration },
    /// The engine may choose its own allocation from this remaining clock.
    Clock {
        remaining: Duration,
        increment: Duration,
    },
}

/// A move-selection policy that can run behind the standard UAI loop.
pub trait Engine {
    /// Returns the engine identity sent during the UAI handshake.
    fn name(&self) -> &str;

    /// Returns the engine author sent during the UAI handshake.
    fn author(&self) -> &str;

    /// Selects a legal move using the supplied position and time context.
    fn choose_move(&mut self, game: &GameState, time: SearchTime) -> io::Result<Move>;
}

/// Runs an engine policy against standard input and output.
pub fn run_engine<E: Engine>(engine: E) -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    run_engine_with_io(engine, stdin.lock(), &mut stdout)
}

/// Runs an engine policy against caller-provided line input and output.
pub fn run_engine_with_io<E, R, W>(mut engine: E, input: R, output: &mut W) -> io::Result<()>
where
    E: Engine,
    R: BufRead,
    W: Write,
{
    let mut game = None;
    for line in input.lines() {
        let command = line?;
        match command.as_str() {
            "uai" => {
                writeln!(output, "id name {}", engine.name())?;
                writeln!(output, "id author {}", engine.author())?;
                writeln!(output, "uaiok")?;
            }
            "isready" => writeln!(output, "readyok")?,
            "newgame" => game = None,
            "quit" => break,
            "stop" => {}
            command if command.starts_with("position fen ") => {
                let fen = command
                    .strip_prefix("position fen ")
                    .expect("command prefix was checked");
                game = Some(
                    GameState::from_azul_fen(&format!("{fen}\n"))
                        .map_err(|_| invalid_command("invalid AzulFEN position"))?,
                );
            }
            command if command.starts_with("go movetime ") => {
                let milliseconds = parse_duration(command, "go movetime ")?;
                write_bestmove(
                    &mut engine,
                    game.as_ref(),
                    SearchTime::Fixed {
                        limit: milliseconds,
                    },
                    output,
                )?;
            }
            command if command.starts_with("go clock ") => {
                let (remaining, increment) = parse_clock(command)?;
                write_bestmove(
                    &mut engine,
                    game.as_ref(),
                    SearchTime::Clock {
                        remaining,
                        increment,
                    },
                    output,
                )?;
            }
            command if command.starts_with("setoption ") || command.is_empty() => {}
            _ => writeln!(output, "error invalid command")?,
        }
        output.flush()?;
    }
    Ok(())
}

/// Requests and formats one move from the policy.
fn write_bestmove<E: Engine, W: Write>(
    engine: &mut E,
    game: Option<&GameState>,
    time: SearchTime,
    output: &mut W,
) -> io::Result<()> {
    let game = game.ok_or_else(|| invalid_command("go received before position"))?;
    let choice = engine.choose_move(game, time)?;
    if !game.get_valid_moves().contains(&choice) {
        return Err(invalid_command("engine returned an illegal move"));
    }
    writeln!(output, "bestmove {choice}")
}

/// Parses a non-negative millisecond duration from a command.
fn parse_duration(command: &str, prefix: &str) -> io::Result<Duration> {
    let milliseconds = command
        .strip_prefix(prefix)
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| invalid_command("invalid millisecond duration"))?;
    Ok(Duration::from_millis(milliseconds))
}

/// Parses the project-local clock command: `go clock REMAINING INCREMENT`.
fn parse_clock(command: &str) -> io::Result<(Duration, Duration)> {
    let mut values = command
        .strip_prefix("go clock ")
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse::<u64>());
    let remaining = values
        .next()
        .transpose()
        .map_err(|_| invalid_command("invalid remaining clock"))?
        .ok_or_else(|| invalid_command("missing remaining clock"))?;
    let increment = values
        .next()
        .transpose()
        .map_err(|_| invalid_command("invalid clock increment"))
        .and_then(|value| value.ok_or_else(|| invalid_command("missing clock increment")))?;
    if values.next().is_some() {
        return Err(invalid_command("too many clock fields"));
    }
    Ok((
        Duration::from_millis(remaining),
        Duration::from_millis(increment),
    ))
}

/// Creates a protocol error for a malformed engine command or response.
fn invalid_command(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::{Engine, SearchTime, run_engine_with_io};
    use azul_movegen::{GameState, Move};
    use std::io::{self, Cursor};
    use std::time::Duration;

    struct TestEngine;

    impl Engine for TestEngine {
        fn name(&self) -> &str {
            "Test Engine"
        }

        fn author(&self) -> &str {
            "Test Author"
        }

        fn choose_move(&mut self, game: &GameState, time: SearchTime) -> io::Result<Move> {
            assert_eq!(
                time,
                SearchTime::Clock {
                    remaining: Duration::from_secs(10),
                    increment: Duration::from_secs(2),
                }
            );
            game.get_valid_moves()
                .into_iter()
                .next()
                .ok_or_else(|| io::Error::other("no legal move"))
        }
    }

    #[test]
    fn passes_clock_information_to_the_engine_policy() {
        let mut game = GameState::new(2, 1).unwrap();
        game.setup_next_round();
        let fen = interface_fen(&game);
        let input = format!("position fen {fen}go clock 10000 2000\nquit\n");
        let mut output = Vec::new();

        run_engine_with_io(TestEngine, Cursor::new(input), &mut output).unwrap();

        assert!(String::from_utf8(output).unwrap().starts_with("bestmove "));
    }

    fn interface_fen(game: &GameState) -> String {
        use crate::parsing::ToAzulFEN;
        game.to_azul_fen()
    }
}
