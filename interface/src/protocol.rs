use azul_movegen::{Row, Tile, game_move::Move};
use clap::{Parser, ValueEnum};
use std::num::ParseIntError;

#[derive(Debug, Clone)]
struct EngineConfig {
    pub path: String,
    pub proto: Protocol,
    pub tc: Option<TimeControl>,
    pub dir: Option<String>,
    pub args: Option<String>,
    pub name: Option<String>,
    pub limit_mem: Option<u64>,
    pub limit_threads: Option<u32>,
}

#[derive(Debug, Clone)]
enum TimeControl {
    Increment(u32, u32),
    Fixed(u32),
}

#[derive(ValueEnum, Clone)]
enum TournamentStyle {
    Gauntlet,
    RoundRobin,
    Swiss,
    Random,
}

#[derive(Parser)]
#[command(name = "azul-interface", about = "Manages Azul engine matches")]
struct Cli {
    // =====================
    // Engines
    // =====================
    #[arg(long = "engine", value_parser = parse_engine)]
    pub engines: Vec<EngineConfig>,

    // =====================
    // Match config
    // =====================
    #[arg(long, value_enum)]
    pub tournament: Option<TournamentStyle>,

    #[arg(long, value_name = "N")]
    pub concurrency: usize,

    #[arg(long, value_name = "PATH")]
    pub out: String,

    #[arg(long, value_name = "PATH")]
    pub resume: String,

    #[arg(long, value_name = "N")]
    pub rounds: usize,

    #[arg(long, value_name = "N")]
    pub games: usize,

    #[arg(long, action)]
    pub repeat: bool,

    #[arg(long = "max-games", value_name = "N")]
    pub max_games: usize,

    #[arg(long, value_name = "N")]
    pub seed: u64,

    #[arg(long, value_name = "PATH")]
    pub openings: String,

    #[arg(long, action)]
    pub swap: bool,

    #[arg(long, value_name = "N")]
    pub timeout: usize,

    #[arg(long, action)]
    pub recover: bool,

    // =====================
    // Debugging and logging
    // =====================
    #[arg(long, action)]
    pub version: bool,

    #[arg(long = "dry-run", action)]
    pub dry_run: bool,

    #[arg(long = "check-engines", action)]
    pub check_engines: bool,

    #[arg(long, action)]
    pub summary: bool,

    #[arg(long, action)]
    pub debug: bool,

    #[arg(long, action)]
    pub log: bool,

    #[arg(long, action)]
    pub stderr: bool,

    #[arg(long, action)]
    pub quiet: bool,
}

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
                    let base = s.parse::<u32>().map_err(|_| "Invalid time format")?;
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

pub fn full_parse() {
    let cli = Cli::parse();
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Protocol {
    Human,
    UAI,
}

#[derive(Debug)]
pub struct ParseMoveError;

impl From<ParseIntError> for ParseMoveError {
    fn from(_: ParseIntError) -> Self {
        ParseMoveError
    }
}

/*
Here we expect moves in the format of `bowl, tile_type, row` where each input is a two-digit number
ex. 040102 would correspond to the fourth bowl, first tile type, and second row of our own board
Note: Bowl 00 will always correspond to the centre area, and row 00 will always correspond to the penalty area
*/
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
