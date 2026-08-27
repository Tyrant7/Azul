use std::io::{self, BufRead, Write};

use azul_movegen::GameState;
use interface::parsing::FromAzulFEN;

use rand::prelude::IndexedRandom;

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut game = None;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("input error: {error}");
                break;
            }
        };

        if handle_command(&line, &mut game, &mut stdout).is_err() {
            let _ = writeln!(stdout, "error invalid command");
            let _ = stdout.flush();
        }
        if line == "quit" {
            break;
        }
    }
}

fn handle_command(
    command: &str,
    game: &mut Option<GameState>,
    stdout: &mut impl Write,
) -> io::Result<()> {
    match command {
        "uai" => {
            writeln!(stdout, "id name Azul Random Engine")?;
            writeln!(stdout, "id author Azul contributors")?;
            writeln!(stdout, "uaiok")?;
        }
        "isready" => writeln!(stdout, "readyok")?,
        "newgame" => *game = None,
        "quit" => {}
        "go" => write_bestmove(game, stdout)?,
        command if command.starts_with("position fen ") => {
            let fen = command.strip_prefix("position fen ").unwrap();
            *game = Some(
                GameState::from_azul_fen(&format!("{fen}\n"))
                    .map_err(|_| invalid_command("invalid AzulFEN position"))?,
            );
        }
        "stop" => {}
        _ if command.starts_with("setoption ") => {}
        _ if command.is_empty() => {}
        _ => return Err(invalid_command("unsupported UAI command")),
    }
    stdout.flush()
}

fn write_bestmove(game: &Option<GameState>, stdout: &mut impl Write) -> io::Result<()> {
    let game = game
        .as_ref()
        .ok_or_else(|| invalid_command("go received before position"))?;
    let moves = game.get_valid_moves();
    let choice = moves
        .choose(&mut rand::rng())
        .ok_or_else(|| invalid_command("position has no legal moves"))?;
    writeln!(stdout, "bestmove {choice}")
}

fn invalid_command(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::handle_command;

    #[test]
    fn announces_uai_identity_and_readiness() {
        let mut output = Vec::new();
        let mut game = None;
        handle_command("uai", &mut game, &mut output).unwrap();
        handle_command("isready", &mut game, &mut output).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("id name Azul Random Engine\n"));
        assert!(output.contains("id author Azul contributors\n"));
        assert!(output.contains("uaiok\nreadyok\n"));
    }

    #[test]
    fn rejects_go_without_a_position() {
        let mut output = Vec::new();
        let mut game = None;
        assert!(handle_command("go", &mut game, &mut output).is_err());
    }
}
