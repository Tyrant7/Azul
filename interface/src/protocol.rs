//! Command-line configuration, protocol modes, and move parsing.

use azul_movegen::{Row, Tile, game_move::Move};
use clap::{Parser, ValueEnum};
use std::num::ParseIntError;

/// Configuration for one engine participating in a match.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to the engine executable.
    pub path: String,
    /// Protocol or interaction mode used by the engine.
    pub proto: Protocol,
    /// Time control assigned to the engine.
    pub tc: Option<TimeControl>,
    /// Working directory for the engine process.
    pub dir: Option<String>,
    /// Additional arguments passed to the engine process.
    pub args: Option<String>,
    /// Optional display name for the engine.
    pub name: Option<String>,
    /// Optional per-engine memory limit value.
    pub limit_mem: Option<u64>,
    /// Optional per-engine thread limit.
    pub limit_threads: Option<u32>,
}

/// Output and interaction mode used for an engine.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Protocol {
    /// Human-readable state and move interaction.
    Human,
    /// The [draft Universal Azul Interface protocol](../protocol.md).
    UAI,
}

/// Time limit assigned to an engine.
#[derive(Debug, Clone)]
pub enum TimeControl {
    /// Base time and per-move increment, expressed in seconds.
    Increment(u32, u32),
    /// Fixed time per move, expressed in milliseconds.
    Fixed(u32),
}

/// Tournament pairing strategy.
#[derive(ValueEnum, Debug, Clone)]
pub enum TournamentStyle {
    /// One engine plays a series against each opponent.
    Gauntlet,
    /// Each engine plays the other engines in scheduled pairings.
    RoundRobin,
    /// Pairings are assigned by Swiss-style results.
    Swiss,
    /// Pairings are selected randomly.
    Random,
}

// Example with two engine configurations:
// cargo run -p interface -- --engine "path=path tc=60+1" "path=path tc=60+2" --out "path"

/// Command-line configuration for the interface executable.
#[derive(Parser, Debug)]
#[command(name = "azul-interface", about = "Manages Azul engine matches")]
pub struct Cli {
    // Engine configuration
    #[arg(
        long = "engine", 
        required = true,
        value_parser = parse_engine,
        num_args = 2..,
        value_delimiter = None,
        help = "Define two or more engine configurations as key=value descriptors"
    )]
    pub engines: Vec<EngineConfig>,

    // Match configuration
    #[arg(long, value_enum, help = "Select the tournament pairing strategy")]
    pub tournament: Option<TournamentStyle>,

    #[arg(
        long,
        value_name = "N",
        default_value_t = 1,
        help = "Number of games to run concurrently"
    )]
    pub concurrency: usize,

    #[arg(long, value_name = "PATH", help = "Path for match results")]
    pub out: String,

    #[arg(
        long,
        value_name = "PATH",
        help = "Resume from saved tournament results"
    )]
    pub resume: Option<String>,

    #[arg(long, value_name = "N", help = "Number of rounds or matches")]
    pub rounds: Option<usize>,

    #[arg(
        long,
        value_name = "N",
        default_value_t = 1,
        help = "Number of games per match"
    )]
    pub games: usize,

    #[arg(long, action, help = "Repeat the tournament or match")]
    pub repeat: bool,

    #[arg(
        long = "max-games",
        value_name = "N",
        help = "Maximum total number of games"
    )]
    pub max_games: Option<usize>,

    #[arg(
        long,
        value_name = "N",
        help = "Seed for reproducible tournament randomness"
    )]
    pub seed: Option<u64>,

    #[arg(long, value_name = "PATH", help = "Starting positions or opening book")]
    pub openings: Option<String>,

    #[arg(long, action, help = "Balance starting sides between engines")]
    pub swap: bool,

    #[arg(
        long,
        value_name = "N",
        default_value_t = 10,
        help = "Engine response timeout value"
    )]
    pub timeout: usize,

    #[arg(long, action, help = "Recover from an engine process failure")]
    pub recover: bool,

    // Diagnostics and logging
    #[arg(long, action, help = "Print the interface version")]
    pub version: bool,

    #[arg(
        long = "dry-run",
        action,
        help = "Validate configuration without starting games"
    )]
    pub dry_run: bool,

    #[arg(long = "check-engines", action, help = "Check engine handshakes")]
    pub check_engines: bool,

    #[arg(long, action, help = "Print match summaries")]
    pub summary: bool,

    #[arg(long, action, help = "Display engine input and output")]
    pub debug: bool,

    #[arg(long, action, help = "Write engine communication to a log")]
    pub log: bool,

    #[arg(long, action, help = "Display command-line and engine errors")]
    pub stderr: bool,

    #[arg(long, action, help = "Suppress normal output")]
    pub quiet: bool,
}

/// Parses one whitespace-separated engine descriptor.
fn parse_engine(s: &str) -> Result<EngineConfig, String> {
    let mut config = EngineConfig {
        path: String::new(),
        proto: Protocol::UAI,
        tc: None,
        dir: None,
        args: None,
        name: None,
        limit_mem: None,
        limit_threads: None,
    };

    for part in s.split_whitespace() {
        let mut kv = part.splitn(2, "=");
        let key = kv.next().unwrap();
        let val = kv
            .next()
            .ok_or_else(|| format!("Invalid engine arg: {}", part))?;

        match key {
            "path" => config.path = val.to_string(),
            "proto" => {
                config.proto = match val.to_lowercase().as_str() {
                    "uai" => Protocol::UAI,
                    "human" => Protocol::Human,
                    _ => return Err(format!("Invalid protocol: {}", val)),
                }
            }
            "tc" => {
                if config.tc.is_some() {
                    return Err("Cannot specify both tc and st for the same engine".to_string());
                }
                if let Some((base, inc)) = val.split_once('+') {
                    let base = base.parse::<u32>().map_err(|_| "Invalid base time")?;
                    let increment = inc.parse::<u32>().map_err(|_| "Invalid increment")?;
                    config.tc = Some(TimeControl::Increment(base, increment));
                } else {
                    let base = val.parse::<u32>().map_err(|_| "Invalid time format")?;
                    config.tc = Some(TimeControl::Increment(base, 0));
                }
            }
            "st" => {
                if config.tc.is_some() {
                    return Err("Cannot specify both tc and st for the same engine".to_string());
                }
                config.tc = Some(TimeControl::Fixed(
                    val.parse().map_err(|_| "Invalid time format")?,
                ))
            }
            "dir" => config.dir = Some(val.to_string()),
            "args" => config.args = Some(val.to_string()),
            "name" => config.name = Some(val.to_string()),
            "limit_mem" => config.limit_mem = val.parse().ok(),
            "limit_threads" => config.limit_threads = val.parse().ok(),
            _ => return Err(format!("Unknown engine key: {}", key)),
        };
    }

    if config.path.is_empty() {
        return Err("Missing required key: path".to_string());
    } else if config.tc.is_none() {
        return Err("Missing required key: tc".to_string());
    }

    Ok(config)
}

/// Indicates that a six-digit move could not be parsed.
#[derive(Debug)]
pub struct ParseMoveError;

impl From<ParseIntError> for ParseMoveError {
    fn from(_: ParseIntError) -> Self {
        ParseMoveError
    }
}

/*
Moves contain three decimal, two-digit components: bowl index, tile type, and destination row.
For example, 040102 selects tile type 1 from bowl index 4 and sends it to wall row index 1
(the second wall row). Bowl 00 is the centre area, and row 00 is the floor.
*/
/// Parses a six-digit move into a [`Move`].
///
/// The components are a zero-based bowl index, tile type, and destination row.
/// Row `00` maps to [`Row::Floor`]; any other row value maps to a zero-based
/// [`Row::Wall`] index by subtracting one.
///
/// This function parses the move's shape and numeric fields only; legality is
/// checked later by [`azul_movegen::GameState::make_move`].
pub fn parse_move(input: &str) -> Result<Move, ParseMoveError> {
    if input.len() != 6 {
        return Err(ParseMoveError);
    }
    let (bowl, other) = input.split_at(2);
    let (tile_type, row) = other.split_at(2);

    let bowl = bowl.parse::<usize>()?;
    let tile_type = tile_type.parse::<Tile>()?;
    let row = row.parse::<usize>()?;
    let row = if row == 0 {
        Row::Floor
    } else {
        Row::Wall(row - 1)
    };
    Ok(Move {
        bowl,
        tile_type,
        row,
    })
}
