use std::{env, error::Error, io, io::Write};

use azul_movegen::GameState;
use rl_beam::{BeamConfig, BeamPolicy};
use rl_env::ActorPolicy;

const DEFAULT_GAMES: usize = 1_000;
const DEFAULT_SEED: u64 = 0xA2_55_10_01;
const DIAGNOSTIC_INTERVAL: usize = 10;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let checkpoint = arguments.next().ok_or(
        "usage: matchup ACTOR_CHECKPOINT [GAMES] [BEAM_WIDTH] [DEPTH] [CRITIC_CHECKPOINT]",
    )?;
    let games = parse_argument(arguments.next(), "games", DEFAULT_GAMES)?;
    if games == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "games must be positive").into());
    }
    let beam_width = parse_argument(
        arguments.next(),
        "beam width",
        BeamConfig::default().beam_width,
    )?;
    let depth = parse_argument(arguments.next(), "depth", BeamConfig::default().depth)?;
    let critic_checkpoint = arguments
        .next()
        .unwrap_or_else(|| "checkpoints/reference_critic.ot".to_owned());

    let baseline = ActorPolicy::load(&checkpoint)?;
    let beam = BeamPolicy::load(
        &checkpoint,
        critic_checkpoint,
        BeamConfig { beam_width, depth },
    )?;
    let mut beam_wins = 0;
    let mut baseline_wins = 0;

    for game_index in 0..games {
        let beam_player = game_index % 2;
        let winner = play_game(
            &baseline,
            &beam,
            beam_player,
            DEFAULT_SEED.wrapping_add(game_index as u64),
        )?;
        if winner == beam_player {
            beam_wins += 1;
        } else {
            baseline_wins += 1;
        }

        let completed_games = game_index + 1;
        if completed_games % DIAGNOSTIC_INTERVAL == 0 || completed_games == games {
            print_progress(completed_games, games, beam_wins, baseline_wins)?;
        }
    }

    println!(
        "matchup games={} beam_width={} depth={} beam_wins={} baseline_wins={} beam_win_rate={:.3}",
        games,
        beam_width,
        depth,
        beam_wins,
        baseline_wins,
        beam_wins as f32 / games as f32,
    );
    Ok(())
}

/// Prints the current matchup totals at a fixed interval during long runs.
fn print_progress(
    completed_games: usize,
    total_games: usize,
    beam_wins: usize,
    baseline_wins: usize,
) -> io::Result<()> {
    println!(
        "progress games={completed_games}/{total_games} beam_wins={beam_wins} baseline_wins={baseline_wins} beam_win_rate={:.3}",
        beam_wins as f32 / completed_games as f32,
    );
    io::stdout().flush()
}

/// Plays one seeded game with the beam policy assigned to one player.
fn play_game(
    baseline: &ActorPolicy,
    beam: &BeamPolicy,
    beam_player: usize,
    seed: u64,
) -> Result<usize, Box<dyn Error>> {
    let mut game = GameState::new(2, seed)
        .map_err(|error| io::Error::other(format!("failed to create game: {error:?}")))?;
    game.setup_next_round();
    while !game.is_game_over() {
        if game.round_over() {
            game.setup_next_round();
            continue;
        }
        let active_player = game.get_active_player();
        let choice = if active_player == beam_player {
            beam.choose_move(&game)
        } else {
            baseline.choose_move(&game)
        }
        .ok_or("policy returned no legal move")?;
        game.make_move(&choice).map_err(|error| {
            io::Error::other(format!("policy returned illegal move: {error:?}"))
        })?;
    }
    Ok(game.get_winner())
}

/// Parses an optional positive numeric matchup argument.
fn parse_argument(
    value: Option<String>,
    name: &str,
    default: usize,
) -> Result<usize, Box<dyn Error>> {
    match value {
        Some(value) => Ok(value
            .parse::<usize>()
            .map_err(|_| format!("invalid {name}: {value}"))?),
        None => Ok(default),
    }
}
