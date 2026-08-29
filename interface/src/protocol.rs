//! Command-line configuration, protocol modes, and move parsing.

use crate::parsing::ToAzulFEN;
use crate::process::{EngineLaunch, EngineProcess};
use azul_movegen::{GameState, Row, Tile, game_move::Move};
use clap::{Parser, ValueEnum};
use std::{
    io,
    num::ParseIntError,
    time::{Duration, Instant},
};

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
    /// Optional per-engine memory limit in mebibytes.
    pub limit_mem: Option<u64>,
    /// Per-engine thread limit, defaulting to one.
    pub limit_threads: u32,
}

/// Output and interaction mode used for an engine.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Protocol {
    /// Human-readable state and move interaction.
    Human,
    /// The [draft Universal Azul Interface protocol](../protocol.md).
    UAI,
}

/// Identity and option declarations returned during a UAI handshake.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct EngineIdentity {
    /// Display name returned by the engine.
    pub(crate) name: Option<String>,
    /// Author or organization returned by the engine.
    pub(crate) author: Option<String>,
    /// Raw `option` declarations returned by the engine.
    pub(crate) options: Vec<String>,
}

/// Performs the UAI startup handshake for one managed engine process.
pub(crate) fn uai_handshake(
    process: &mut EngineProcess,
    timeout: Duration,
) -> io::Result<EngineIdentity> {
    process.send_line("uai")?;
    let mut identity = EngineIdentity::default();

    loop {
        let line = process.recv_stdout(timeout)?.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "engine closed stdout during uai",
            )
        })?;

        if line == "uaiok" {
            if identity.name.is_none() || identity.author.is_none() {
                return Err(handshake_error(
                    "engine completed uai without id name and id author",
                ));
            }
            return Ok(identity);
        }
        if let Some(name) = line.strip_prefix("id name ") {
            if name.is_empty() || identity.name.replace(name.to_owned()).is_some() {
                return Err(handshake_error(
                    "engine returned an invalid or duplicate id name",
                ));
            }
            continue;
        }
        if let Some(author) = line.strip_prefix("id author ") {
            if author.is_empty() || identity.author.replace(author.to_owned()).is_some() {
                return Err(handshake_error(
                    "engine returned an invalid or duplicate id author",
                ));
            }
            continue;
        }
        if let Some(option) = line.strip_prefix("option ") {
            if option.is_empty() {
                return Err(handshake_error(
                    "engine returned an empty option declaration",
                ));
            }
            identity.options.push(option.to_owned());
            continue;
        }
        if line.starts_with("error ") {
            return Err(handshake_error(line));
        }
        return Err(handshake_error(format!(
            "unexpected response during uai handshake: {line}"
        )));
    }
}

/// Creates a protocol error with a stable error category for callers.
fn handshake_error(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Waits for an engine to finish initialization before a game starts.
pub(crate) fn uai_ready(process: &mut EngineProcess, timeout: Duration) -> io::Result<()> {
    process.send_line("isready")?;
    let line = process.recv_stdout(timeout)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "engine closed stdout during isready",
        )
    })?;
    if line == "readyok" {
        return Ok(());
    }
    if line.starts_with("error ") {
        return Err(engine_response_error(line));
    }
    Err(engine_response_error(format!(
        "unexpected response during isready: {line}"
    )))
}

/// Sends the command that resets an engine's game-specific state.
pub(crate) fn send_new_game(process: &mut EngineProcess) -> io::Result<()> {
    process.send_line("newgame")
}

/// Sends the complete current position to an engine as AzulFEN.
pub(crate) fn send_position(process: &mut EngineProcess, game: &GameState) -> io::Result<()> {
    let fen = game.to_azul_fen();
    let fen = fen.strip_suffix('\n').ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "AzulFEN position is missing its final newline",
        )
    })?;
    if fen.contains(['\r', '\n']) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "AzulFEN position contains an embedded newline",
        ));
    }
    process.send_line(&format!("position fen {fen}"))
}

/// Requests a move with a fixed millisecond budget.
pub(crate) fn send_go_movetime(process: &mut EngineProcess, budget: Duration) -> io::Result<()> {
    process.send_line(&format!("go movetime {}", budget.as_millis()))
}

/// Requests a move while giving the engine its remaining clock and increment.
pub(crate) fn send_go_clock(
    process: &mut EngineProcess,
    remaining: Duration,
    increment: Duration,
) -> io::Result<()> {
    process.send_line(&format!(
        "go clock {} {}",
        remaining.as_millis(),
        increment.as_millis()
    ))
}

/// Tracks one player's configured clock across turns.
#[derive(Debug, Clone)]
struct PlayerClock {
    control: TimeControl,
    remaining: Duration,
}

impl PlayerClock {
    /// Creates a clock initialized from its time-control configuration.
    fn new(control: TimeControl) -> Self {
        let remaining = match &control {
            TimeControl::Increment(base, _) => Duration::from_secs(*base as u64),
            TimeControl::Fixed(milliseconds) => Duration::from_millis(*milliseconds as u64),
        };
        Self { control, remaining }
    }

    /// Returns the hard deadline budget for the next move.
    fn move_budget(&self) -> Duration {
        self.remaining
    }

    /// Sends the time-control information appropriate for the next move.
    fn send_go(&self, process: &mut EngineProcess) -> io::Result<()> {
        match self.control {
            TimeControl::Fixed(milliseconds) => {
                send_go_movetime(process, Duration::from_millis(milliseconds as u64))
            }
            TimeControl::Increment(_, increment) => send_go_clock(
                process,
                self.remaining,
                Duration::from_secs(increment as u64),
            ),
        }
    }

    /// Applies elapsed time and any configured increment after a move.
    fn finish_move(&mut self, elapsed: Duration) -> io::Result<()> {
        if elapsed > self.remaining {
            return Err(timeout_error("engine exceeded its time control"));
        }

        if let TimeControl::Increment(_, increment) = &self.control {
            self.remaining = self.remaining - elapsed + Duration::from_secs(*increment as u64);
        }
        Ok(())
    }
}

/// Classifies a failure attributable to one engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EngineFailureKind {
    /// The engine did not return a move before its deadline.
    Timeout,
    /// The engine process exited or otherwise became unusable.
    Crash,
    /// The engine's input or output pipe was closed unexpectedly.
    BrokenPipe,
    /// The engine explicitly rejected a command or position.
    ErrorResponse,
    /// The engine returned a response outside the protocol grammar.
    MalformedResponse,
    /// The engine returned a syntactically valid but illegal move.
    IllegalMove,
    /// The operating system terminated the engine for exceeding a limit.
    ResourceLimit,
}

impl EngineFailureKind {
    /// Returns whether a bounded restart may recover this failure.
    fn recoverable(self) -> bool {
        matches!(self, Self::Crash | Self::BrokenPipe | Self::ErrorResponse)
    }
}

/// Records the engine and reason responsible for a forfeited game.
#[derive(Debug)]
pub(crate) struct EngineForfeit {
    /// Index of the player whose engine forfeited.
    pub(crate) player: usize,
    /// Classified reason for the forfeit.
    pub(crate) reason: EngineFailureKind,
    /// Human-readable diagnostic detail.
    pub(crate) message: String,
}

/// Result of one UAI game, including engine forfeits.
#[derive(Debug)]
pub(crate) enum GameResult {
    /// The game reached a rules-defined terminal state.
    Completed(GameState),
    /// An engine failed and the game ended by forfeit.
    Forfeit {
        /// State immediately before the failed move or command.
        game: GameState,
        /// Engine failure that ended the game.
        failure: EngineForfeit,
    },
}

/// Maximum recovery attempts permitted for one engine in one game.
const MAX_RESTARTS_PER_ENGINE: usize = 1;

/// Runs a UAI game with optional bounded process recovery.
pub(crate) fn play_uai_game(
    processes: &mut [EngineProcess],
    launches: &[EngineLaunch],
    mut game: GameState,
    time_controls: &[TimeControl],
    recover: bool,
    startup_timeout: Duration,
) -> io::Result<GameResult> {
    if processes.len() != game.get_boards().len()
        || processes.len() != launches.len()
        || processes.len() != time_controls.len()
        || !(2..=4).contains(&processes.len())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "a UAI game requires one launch, time control, and engine for each of two to four players",
        ));
    }

    let mut clocks = time_controls
        .iter()
        .cloned()
        .map(PlayerClock::new)
        .collect::<Vec<_>>();
    let mut restart_counts = vec![0; processes.len()];

    for player in 0..processes.len() {
        if let Err(error) = send_new_game(&mut processes[player]) {
            let failure = transport_failure(error);
            if !recover_engine(
                &mut processes[player],
                &launches[player],
                &game,
                &mut restart_counts[player],
                recover,
                startup_timeout,
            ) {
                return Ok(GameResult::Forfeit {
                    game,
                    failure: EngineForfeit {
                        player,
                        reason: failure.kind,
                        message: failure.message,
                    },
                });
            }
        }
    }

    while !game.is_game_over() {
        let active_player = game.get_active_player();
        let budget = clocks[active_player].move_budget();
        if budget.is_zero() {
            return Ok(GameResult::Forfeit {
                game,
                failure: EngineForfeit {
                    player: active_player,
                    reason: EngineFailureKind::Timeout,
                    message: String::from("engine has no time remaining"),
                },
            });
        }
        let process = processes.get_mut(active_player).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "game active player has no matching engine",
            )
        })?;
        let started = Instant::now();
        let deadline = started + budget;
        let choice = loop {
            match request_move(process, &game, &clocks[active_player], deadline) {
                Ok(choice) => break choice,
                Err(failure) => {
                    if !failure.kind.recoverable()
                        || !recover_engine(
                            process,
                            &launches[active_player],
                            &game,
                            &mut restart_counts[active_player],
                            recover,
                            startup_timeout,
                        )
                    {
                        return Ok(GameResult::Forfeit {
                            game,
                            failure: EngineForfeit {
                                player: active_player,
                                reason: failure.kind,
                                message: failure.message,
                            },
                        });
                    }
                }
            }
        };
        if clocks[active_player]
            .finish_move(started.elapsed())
            .is_err()
        {
            return Ok(GameResult::Forfeit {
                game,
                failure: EngineForfeit {
                    player: active_player,
                    reason: EngineFailureKind::Timeout,
                    message: String::from("engine exceeded its time control"),
                },
            });
        }
        if let Err(error) = game.make_move(&choice) {
            let _ = error;
            return Ok(GameResult::Forfeit {
                game,
                failure: EngineForfeit {
                    player: active_player,
                    reason: EngineFailureKind::IllegalMove,
                    message: String::from("engine returned an illegal move"),
                },
            });
        }

        if game.round_over() {
            game.setup_next_round();
        }
    }

    Ok(GameResult::Completed(game))
}

/// Sends one position/search request and reads the resulting move.
fn request_move(
    process: &mut EngineProcess,
    game: &GameState,
    clock: &PlayerClock,
    deadline: Instant,
) -> Result<Move, EngineTurnFailure> {
    send_position(process, game).map_err(|error| transport_failure_at("position", error))?;
    if deadline.saturating_duration_since(Instant::now()) < Duration::from_millis(1) {
        return Err(EngineTurnFailure::new(
            EngineFailureKind::Timeout,
            "engine has no time remaining for its move",
        ));
    }
    clock
        .send_go(process)
        .map_err(|error| transport_failure_at("go", error))?;
    receive_bestmove(process, deadline)
}

/// Reads engine output until the next move response, ignoring search updates.
fn receive_bestmove(
    process: &mut EngineProcess,
    deadline: Instant,
) -> Result<Move, EngineTurnFailure> {
    loop {
        let timeout = deadline.saturating_duration_since(Instant::now());
        let line = process
            .recv_stdout(timeout)
            .map_err(transport_failure)?
            .ok_or_else(|| {
                EngineTurnFailure::new(
                    EngineFailureKind::Crash,
                    "engine closed stdout while searching",
                )
            })?;
        if line.starts_with("info ") {
            continue;
        }
        if line.starts_with("error ") {
            return Err(EngineTurnFailure::new(
                EngineFailureKind::ErrorResponse,
                line,
            ));
        }
        let move_choice = parse_bestmove(&line).map_err(|_| {
            EngineTurnFailure::new(
                EngineFailureKind::MalformedResponse,
                format!("invalid bestmove response: {line}"),
            )
        })?;
        if Instant::now() > deadline {
            return Err(EngineTurnFailure::new(
                EngineFailureKind::Timeout,
                "engine exceeded its time control",
            ));
        }
        return Ok(move_choice);
    }
}

/// Failure returned while an engine is producing one move.
#[derive(Debug)]
struct EngineTurnFailure {
    kind: EngineFailureKind,
    message: String,
}

impl EngineTurnFailure {
    /// Creates a classified turn failure.
    fn new(kind: EngineFailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// Converts an engine I/O failure into a recovery classification.
fn transport_failure(error: io::Error) -> EngineTurnFailure {
    transport_failure_at("engine transport", error)
}

/// Converts an I/O failure while handling a named protocol stage.
fn transport_failure_at(context: &str, error: io::Error) -> EngineTurnFailure {
    let kind = match error.kind() {
        io::ErrorKind::TimedOut => EngineFailureKind::Timeout,
        io::ErrorKind::BrokenPipe => EngineFailureKind::BrokenPipe,
        io::ErrorKind::Other if error.to_string().contains("resource limit") => {
            EngineFailureKind::ResourceLimit
        }
        _ => EngineFailureKind::Crash,
    };
    EngineTurnFailure::new(kind, format!("{context}: {error}"))
}

/// Restarts, handshakes, resets, and restores one engine when permitted.
fn recover_engine(
    process: &mut EngineProcess,
    launch: &EngineLaunch,
    game: &GameState,
    restart_count: &mut usize,
    recover: bool,
    startup_timeout: Duration,
) -> bool {
    if !recover || *restart_count >= MAX_RESTARTS_PER_ENGINE {
        return false;
    }
    *restart_count += 1;

    process
        .restart(launch, Duration::from_millis(100))
        .and_then(|_| uai_handshake(process, startup_timeout).map(|_| ()))
        .and_then(|_| uai_ready(process, startup_timeout))
        .and_then(|_| send_new_game(process))
        .and_then(|_| send_position(process, game))
        .is_ok()
}

/// Creates a stable timeout error for a player that used too much time.
fn timeout_error(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::TimedOut, message.into())
}

/// Creates an error for a malformed or unusable engine response.
fn engine_response_error(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
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
        limit_threads: 1,
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
            "limit_mem" => {
                let limit = val
                    .parse::<u64>()
                    .map_err(|_| "Invalid memory limit in MiB")?;
                if limit == 0 {
                    return Err("Memory limit must be greater than zero MiB".to_string());
                }
                config.limit_mem = Some(limit);
            }
            "limit_threads" => {
                let limit = val.parse::<u32>().map_err(|_| "Invalid thread limit")?;
                if limit == 0 {
                    return Err("Thread limit must be greater than zero".to_string());
                }
                config.limit_threads = limit;
            }
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
    if input.len() != 6 || !input.bytes().all(|byte| byte.is_ascii_digit()) {
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

/// Indicates that an engine's `bestmove` response could not be parsed.
#[derive(Debug)]
pub struct ParseBestMoveError;

/// Parses a strict `bestmove <move>` engine response.
///
/// The response must contain exactly one six-digit move payload. Move legality
/// is checked separately by [`azul_movegen::GameState::make_move`].
pub fn parse_bestmove(response: &str) -> Result<Move, ParseBestMoveError> {
    let move_text = response
        .strip_prefix("bestmove ")
        .ok_or(ParseBestMoveError)?;
    parse_move(move_text).map_err(|_| ParseBestMoveError)
}

#[cfg(test)]
mod tests {
    use super::{
        Cli, Protocol, TimeControl, parse_bestmove, parse_engine, parse_move, play_uai_game,
        send_go_clock, send_go_movetime, send_new_game, send_position, uai_handshake, uai_ready,
    };
    use crate::parsing::ToAzulFEN;
    use crate::process::{EngineLaunch, EngineProcess};
    use azul_movegen::{Bag, Board, Bowl, GameState, Row, Tile, game_move::Move};
    use clap::Parser;
    use std::{
        fs,
        io::ErrorKind,
        process::Command,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    /// Builds a platform-native child command for UAI handshake tests.
    fn fixture(script: &str) -> Command {
        #[cfg(windows)]
        {
            let mut command = Command::new(std::env::var_os("COMSPEC").unwrap());
            command.args(["/Q", "/V:ON", "/C", script]);
            command
        }

        #[cfg(not(windows))]
        {
            let mut command = Command::new("sh");
            command.args(["-c", script]);
            command
        }
    }

    /// Builds a reusable launch specification for a platform-native fixture.
    fn launch_in_dir(script: &str, current_dir: Option<String>) -> EngineLaunch {
        #[cfg(windows)]
        {
            EngineLaunch::new(
                std::env::var("COMSPEC").unwrap(),
                vec![
                    String::from("/Q"),
                    String::from("/V:ON"),
                    String::from("/C"),
                    script.into(),
                ],
                current_dir,
            )
        }

        #[cfg(not(windows))]
        {
            EngineLaunch::new(
                String::from("sh"),
                vec![String::from("-c"), script.into()],
                current_dir,
            )
        }
    }

    /// Builds a reusable launch specification for a fixture in the current directory.
    fn launch(script: &str) -> EngineLaunch {
        launch_in_dir(script, None)
    }

    #[test]
    fn parse_engine_accepts_all_descriptor_fields() {
        let config = parse_engine(
            "path=engine proto=HuMaN tc=60+5 dir=work args=--seed name=Test limit_mem=1024 limit_threads=4",
        )
        .unwrap();

        assert_eq!(config.path, "engine");
        assert!(matches!(config.proto, Protocol::Human));
        assert!(matches!(config.tc, Some(TimeControl::Increment(60, 5))));
        assert_eq!(config.dir.as_deref(), Some("work"));
        assert_eq!(config.args.as_deref(), Some("--seed"));
        assert_eq!(config.name.as_deref(), Some("Test"));
        assert_eq!(config.limit_mem, Some(1024));
        assert_eq!(config.limit_threads, 4);
    }

    #[test]
    fn parse_engine_accepts_fixed_and_zero_increment_time_controls() {
        let incremental = parse_engine("path=engine tc=60").unwrap();
        assert!(matches!(
            incremental.tc,
            Some(TimeControl::Increment(60, 0))
        ));

        let fixed = parse_engine("path=engine st=500").unwrap();
        assert!(matches!(fixed.tc, Some(TimeControl::Fixed(500))));
    }

    #[test]
    fn parse_engine_rejects_invalid_descriptors() {
        for descriptor in [
            "tc=60",
            "path=engine",
            "path=engine tc=60 st=500",
            "path=engine tc=bad",
            "path=engine tc=60 limit_mem=bad",
            "path=engine tc=60 limit_mem=0",
            "path=engine tc=60 limit_threads=bad",
            "path=engine tc=60 limit_threads=0",
            "path=engine tc=60 unknown=value",
        ] {
            assert!(parse_engine(descriptor).is_err(), "accepted {descriptor}");
        }
    }

    #[test]
    fn parse_engine_defaults_to_one_thread() {
        let config = parse_engine("path=engine tc=60").unwrap();

        assert_eq!(config.limit_mem, None);
        assert_eq!(config.limit_threads, 1);
    }

    #[test]
    fn cli_requires_two_engines_and_an_output_path() {
        let parsed = Cli::try_parse_from([
            "azul-interface",
            "--engine",
            "path=first tc=60",
            "path=second st=500",
            "--out",
            "results.azl",
            "--tournament",
            "round-robin",
            "--games",
            "3",
            "--dry-run",
            "--quiet",
            "--debug",
        ])
        .unwrap();

        assert_eq!(parsed.engines.len(), 2);
        assert_eq!(parsed.out, "results.azl");
        assert!(matches!(
            parsed.tournament,
            Some(super::TournamentStyle::RoundRobin)
        ));
        assert_eq!(parsed.games, 3);
        assert!(parsed.dry_run);
        assert!(parsed.quiet);
        assert!(parsed.debug);
        assert_eq!(parsed.concurrency, 1);
        assert_eq!(parsed.timeout, 10);

        assert!(Cli::try_parse_from(["azul-interface", "--engine", "path=only tc=60"]).is_err());
    }

    #[test]
    fn parse_move_decodes_floor_and_wall_destinations() {
        assert_eq!(
            parse_move("000000").unwrap(),
            Move {
                bowl: 0,
                tile_type: 0,
                row: Row::Floor,
            }
        );
        assert_eq!(
            parse_move("040102").unwrap(),
            Move {
                bowl: 4,
                tile_type: 1,
                row: Row::Wall(1),
            }
        );
    }

    #[test]
    fn parse_move_rejects_malformed_input_without_panicking() {
        for input in ["", "00000", "0000000", "00a000", "€123"] {
            assert!(parse_move(input).is_err(), "accepted {input:?}");
        }
    }

    #[test]
    fn parse_bestmove_decodes_the_move_payload() {
        assert_eq!(
            parse_bestmove("bestmove 040102").unwrap(),
            Move {
                bowl: 4,
                tile_type: 1,
                row: Row::Wall(1),
            }
        );
        assert_eq!(
            parse_bestmove("bestmove 000000").unwrap(),
            Move {
                bowl: 0,
                tile_type: 0,
                row: Row::Floor,
            }
        );
    }

    #[test]
    fn parse_bestmove_rejects_noncanonical_responses() {
        for response in [
            "bestmove",
            "bestmove ",
            "bestmove 04010a",
            "bestmove 040102 extra",
            "move 040102",
            " bestmove 040102",
        ] {
            assert!(
                parse_bestmove(response).is_err(),
                "accepted response {response:?}"
            );
        }
    }

    #[test]
    fn uai_handshake_collects_identity_and_options() {
        #[cfg(windows)]
        let script = "set /p line=& echo id name TestEngine& echo id author TestAuthor& echo option name Skill type spin default 5& echo uaiok";
        #[cfg(not(windows))]
        let script = "IFS= read line; printf 'id name TestEngine\\nid author TestAuthor\\noption name Skill type spin default 5\\nuaiok\\n'";

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        let identity = uai_handshake(&mut process, Duration::from_secs(1)).unwrap();

        assert_eq!(identity.name.as_deref(), Some("TestEngine"));
        assert_eq!(identity.author.as_deref(), Some("TestAuthor"));
        assert_eq!(identity.options, vec!["name Skill type spin default 5"]);
    }

    #[test]
    fn uai_handshake_rejects_engine_errors() {
        #[cfg(windows)]
        let script = "set /p line=& echo error unsupported";
        #[cfg(not(windows))]
        let script = "IFS= read line; printf 'error unsupported\\n'";

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        let error = uai_handshake(&mut process, Duration::from_secs(1)).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn uai_ready_waits_for_readyok() {
        #[cfg(windows)]
        let script = "set /p line=& echo readyok";
        #[cfg(not(windows))]
        let script = "IFS= read line; printf 'readyok\\n'";

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        uai_ready(&mut process, Duration::from_secs(1)).unwrap();
    }

    #[test]
    fn player_clock_adds_increment_after_each_move() {
        let mut clock = super::PlayerClock::new(TimeControl::Increment(60, 5));

        assert_eq!(clock.move_budget(), Duration::from_secs(60));
        clock.finish_move(Duration::from_secs(2)).unwrap();
        assert_eq!(clock.move_budget(), Duration::from_secs(63));
    }

    #[test]
    fn player_clock_rejects_elapsed_time_beyond_the_budget() {
        let mut clock = super::PlayerClock::new(TimeControl::Fixed(10));

        let error = clock.finish_move(Duration::from_millis(11)).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::TimedOut);
    }

    #[test]
    fn play_uai_game_dispatches_turns_and_applies_bestmove() {
        #[cfg(windows)]
        let active_script = "set /p line=& echo info depth 1& echo bestmove 010401& set /p line=& ping -n 2 127.0.0.1 > nul";
        #[cfg(not(windows))]
        let active_script =
            "IFS= read line; printf 'info depth 1\\nbestmove 010401\\n'; IFS= read line; sleep 2";
        #[cfg(windows)]
        let waiting_script = "set /p line=";
        #[cfg(not(windows))]
        let waiting_script = "IFS= read line; sleep 2";

        let board = Board::builder()
            .placed([
                [Some(0), Some(1), Some(2), Some(3), None],
                [None; 5],
                [None; 5],
                [None; 5],
                [None; 5],
            ])
            .build();
        let mut bowls = vec![Bowl::default(); 6];
        bowls[1].fill(vec![4 as Tile]);
        let game = GameState::builder()
            .boards(vec![board, Board::default()])
            .bowls(bowls)
            .bag(Bag::default())
            .set_seed(42)
            .build()
            .unwrap();

        let launches = [launch(active_script), launch(waiting_script)];
        let mut processes = launches
            .iter()
            .map(|launch| EngineProcess::spawn_launch(launch).unwrap())
            .collect::<Vec<_>>();
        let controls = [TimeControl::Fixed(1_000), TimeControl::Fixed(1_000)];
        let result = play_uai_game(
            &mut processes,
            &launches,
            game,
            &controls,
            false,
            Duration::from_secs(5),
        )
        .unwrap();

        match result {
            super::GameResult::Completed(game) => {
                assert!(game.is_game_over());
                assert_eq!(game.get_winner(), 0);
                assert_eq!(game.get_boards()[0].count_horizontal_lines(), 1);
            }
            super::GameResult::Forfeit { failure, .. } => {
                panic!("unexpected forfeit: {failure:?}")
            }
        }
    }

    #[test]
    fn play_uai_game_recovers_once_from_an_engine_error() {
        let marker = format!(
            "azul-uai-recovery-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let current_dir = std::env::temp_dir();

        #[cfg(windows)]
        let script = format!(
            "$marker = '{marker}'; while (($line = [Console]::In.ReadLine()) -ne $null) {{ switch -Wildcard ($line) {{ 'uai' {{ 'id name Recovery'; 'id author Test'; 'uaiok' }} 'isready' {{ 'readyok' }} 'go*' {{ if (Test-Path -LiteralPath $marker) {{ 'bestmove 010401' }} else {{ New-Item -ItemType File -Path $marker | Out-Null; 'error transient' }} }} 'quit' {{ exit 0 }} }} }}"
        );
        #[cfg(not(windows))]
        let script = format!(
            "while IFS= read -r line; do case \"$line\" in uai) printf 'id name Recovery\\nid author Test\\nuaiok\\n';; isready) printf 'readyok\\n';; go*) if [ -e \"{marker}\" ]; then printf 'bestmove 010401\\n'; else : > \"{marker}\"; printf 'error transient\\n'; fi;; quit) exit 0;; esac; done"
        );

        let board = Board::builder()
            .placed([
                [Some(0), Some(1), Some(2), Some(3), None],
                [None; 5],
                [None; 5],
                [None; 5],
                [None; 5],
            ])
            .build();
        let mut bowls = vec![Bowl::default(); 6];
        bowls[1].fill(vec![4 as Tile]);
        let game = GameState::builder()
            .boards(vec![board, Board::default()])
            .bowls(bowls)
            .bag(Bag::default())
            .set_seed(42)
            .build()
            .unwrap();

        #[cfg(windows)]
        let launches = [
            EngineLaunch::new(
                String::from("powershell.exe"),
                vec![
                    String::from("-NoProfile"),
                    String::from("-Command"),
                    script.clone(),
                ],
                Some(current_dir.to_string_lossy().into_owned()),
            ),
            EngineLaunch::new(
                String::from("powershell.exe"),
                vec![
                    String::from("-NoProfile"),
                    String::from("-Command"),
                    script.clone(),
                ],
                Some(current_dir.to_string_lossy().into_owned()),
            ),
        ];
        #[cfg(not(windows))]
        let launches = [
            launch_in_dir(&script, Some(current_dir.to_string_lossy().into_owned())),
            launch_in_dir(&script, Some(current_dir.to_string_lossy().into_owned())),
        ];
        let mut processes = launches
            .iter()
            .map(|launch| EngineProcess::spawn_launch(launch).unwrap())
            .collect::<Vec<_>>();
        let controls = [TimeControl::Fixed(5_000), TimeControl::Fixed(5_000)];
        let result = play_uai_game(
            &mut processes,
            &launches,
            game,
            &controls,
            true,
            Duration::from_secs(1),
        )
        .unwrap();

        let _ = fs::remove_file(current_dir.join(&marker));
        match result {
            super::GameResult::Completed(game) => assert!(game.is_game_over()),
            super::GameResult::Forfeit { failure, .. } => {
                panic!("unexpected forfeit after recovery: {failure:?}")
            }
        }
    }

    #[test]
    fn play_uai_game_rejects_an_illegal_bestmove() {
        #[cfg(windows)]
        let script = "set /p line=& echo bestmove 990499& set /p line=& ping -n 2 127.0.0.1 > nul";
        #[cfg(not(windows))]
        let script = "IFS= read line; printf 'bestmove 990499\\n'; IFS= read line; sleep 2";

        let game = GameState::new(2, 42).unwrap();
        let launches = [launch(script), launch("sleep 2")];
        let mut processes = launches
            .iter()
            .map(|launch| EngineProcess::spawn_launch(launch).unwrap())
            .collect::<Vec<_>>();
        let controls = [TimeControl::Fixed(1_000), TimeControl::Fixed(1_000)];
        let result = play_uai_game(
            &mut processes,
            &launches,
            game,
            &controls,
            false,
            Duration::from_secs(5),
        )
        .unwrap();

        match result {
            super::GameResult::Forfeit { failure, .. } => {
                assert_eq!(failure.reason, super::EngineFailureKind::IllegalMove);
            }
            super::GameResult::Completed(_) => panic!("illegal move completed the game"),
        }
    }

    #[test]
    fn play_uai_game_enforces_the_move_deadline() {
        #[cfg(windows)]
        let slow_script = "set /p line=& ping -n 3 127.0.0.1 > nul& echo bestmove 010401";
        #[cfg(not(windows))]
        let slow_script = "IFS= read line; sleep 1; printf 'bestmove 010401\\n'";

        let game = GameState::new(2, 42).unwrap();
        let launches = [launch(slow_script), launch("sleep 2")];
        let mut processes = launches
            .iter()
            .map(|launch| EngineProcess::spawn_launch(launch).unwrap())
            .collect::<Vec<_>>();
        let controls = [TimeControl::Fixed(10), TimeControl::Fixed(10)];
        let result = play_uai_game(
            &mut processes,
            &launches,
            game,
            &controls,
            false,
            Duration::from_secs(5),
        )
        .unwrap();

        match result {
            super::GameResult::Forfeit { failure, .. } => {
                assert_eq!(failure.reason, super::EngineFailureKind::Timeout);
            }
            super::GameResult::Completed(_) => panic!("slow engine completed the game"),
        }
    }

    #[test]
    fn sends_new_game_position_and_go_commands() {
        #[cfg(windows)]
        let script = "set /p line=& echo !line!";
        #[cfg(not(windows))]
        let script = "IFS= read line; printf '%s\\n' \"$line\"";

        let mut game = GameState::new(2, 42).unwrap();
        game.setup_next_round();
        let expected_fen = game.to_azul_fen();
        let expected_fen = expected_fen.strip_suffix('\n').unwrap();

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        send_new_game(&mut process).unwrap();
        assert_eq!(
            process.recv_stdout(Duration::from_secs(1)).unwrap(),
            Some(String::from("newgame"))
        );

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        send_position(&mut process, &game).unwrap();
        assert_eq!(
            process.recv_stdout(Duration::from_secs(1)).unwrap(),
            Some(format!("position fen {expected_fen}"))
        );

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        process.send_line("go").unwrap();
        assert_eq!(
            process.recv_stdout(Duration::from_secs(1)).unwrap(),
            Some(String::from("go"))
        );

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        send_go_movetime(&mut process, Duration::from_millis(500)).unwrap();
        assert_eq!(
            process.recv_stdout(Duration::from_secs(1)).unwrap(),
            Some(String::from("go movetime 500"))
        );

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        send_go_clock(
            &mut process,
            Duration::from_secs(10),
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(
            process.recv_stdout(Duration::from_secs(1)).unwrap(),
            Some(String::from("go clock 10000 2000"))
        );
    }
}
