use azul_movegen::{
    Bag, Board, Bowl, GameState, Tile,
    board::{BOARD_DIMENSION, BonusTypes},
};
use rand::{Rng, SeedableRng, rng, rngs::SmallRng};

use crate::format::ProtocolFormat;

/// Attempting to parse an invalid AzulFEN or AzulFEN component will produce this error.
#[derive(Debug)]
pub struct ParseGameStateError;

/// Constructs a value from an AzulFEN string or component.
pub trait FromAzulFEN: Sized {
    /// Parses an AzulFEN representation.
    fn from_azul_fen(fen: &str) -> Result<Self, ParseGameStateError>;
}

/// Serializes a value to its AzulFEN representation.
pub trait ToAzulFEN {
    /// Returns the AzulFEN representation.
    fn to_azul_fen(&self) -> String;
}

impl FromAzulFEN for Bowl {
    /// Creates a bowl from the given AzulFEN bowl component.
    /// It is important to note that the bowl component is not an entire FEN.
    /// See `interface/azulfen.md` in the repository for the format specification.
    fn from_azul_fen(bowl_fen: &str) -> Result<Self, ParseGameStateError> {
        if bowl_fen.chars().nth(0).ok_or(ParseGameStateError)? == '-' {
            Ok(Bowl::default())
        } else {
            Ok(Bowl::from_tiles(
                bowl_fen
                    .chars()
                    .map(|c| c.to_string().parse::<Tile>().or(Err(ParseGameStateError)))
                    .collect::<Result<Vec<_>, ParseGameStateError>>()?,
            ))
        }
    }
}

impl FromAzulFEN for Board {
    /// Generates a board matching the given board component of a given AzulFEN.
    /// It is important to note that the board component is not an entire FEN.
    /// See `interface/azulfen.md` in the repository for the format specification.
    fn from_azul_fen(board_fen: &str) -> Result<Self, ParseGameStateError> {
        let mut builder = Board::builder();
        let parts: Vec<_> = board_fen.split_whitespace().collect();
        match parts.as_slice() {
            [
                placed_parts,
                held,
                bonus_rows,
                bonus_cols,
                bonus_tile_types,
                score,
                penalties,
            ] => {
                // Decode the wall using run-length counts for empty positions.
                let mut placed = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
                let mut y = 0;
                let mut x = 0;
                for p in placed_parts.chars() {
                    if let Ok(step) = p.to_string().parse::<usize>() {
                        x += step;
                    } else if p == '-' {
                        placed[y][x] = Some(Board::get_tile_type_at_pos(y, x));
                        x += 1;
                    }
                    if x >= BOARD_DIMENSION {
                        y += 1;
                        x = 0;
                    }
                }
                builder = builder.placed(placed);

                // Decode each pattern line as a tile type and tile count.
                let mut holds = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
                for (i, h) in held.chars().collect::<Vec<_>>().chunks(2).enumerate() {
                    let tile_type = h[0]
                        .to_string()
                        .parse::<Tile>()
                        .or(Err(ParseGameStateError))?;
                    let tile_count = h[1]
                        .to_string()
                        .parse::<Tile>()
                        .or(Err(ParseGameStateError))?;
                    if tile_count == 0 {
                        continue;
                    }
                    for n in 0..tile_count {
                        holds[i][n] = Some(tile_type);
                    }
                }
                builder = builder.holds(holds);

                // Decode collected row, column, and tile-type bonuses.
                builder = builder.bonuses(BonusTypes {
                    rows: bonus_rows
                        .chars()
                        .map(|c| c == '1')
                        .collect::<Vec<_>>()
                        .try_into()
                        .or(Err(ParseGameStateError))?,
                    columns: bonus_cols
                        .chars()
                        .map(|c| c == '1')
                        .collect::<Vec<_>>()
                        .try_into()
                        .or(Err(ParseGameStateError))?,
                    tile_types: bonus_tile_types
                        .chars()
                        .map(|c| c == '1')
                        .collect::<Vec<_>>()
                        .try_into()
                        .or(Err(ParseGameStateError))?,
                });

                // Decode the score and stored penalty-tile count.
                builder = builder.score(score.parse().or(Err(ParseGameStateError))?);
                builder = builder.penalties(penalties.parse().or(Err(ParseGameStateError))?);
            }
            _ => return Err(ParseGameStateError),
        };
        Ok(builder.build())
    }
}

impl FromAzulFEN for GameState {
    /// Parses the given AzulFEN into a gamestate.
    /// Will error if the given AzulFEN is invalid.
    /// See `interface/azulfen.md` in the repository for the format specification.
    fn from_azul_fen(azul_fen: &str) -> Result<Self, ParseGameStateError> {
        let mut sections = azul_fen.split("| ");

        let board_fens = sections.next().ok_or(ParseGameStateError)?.trim();
        let mut board_fens: Vec<_> = board_fens.split(";").map(|f| f.trim()).collect();
        // AzulFEN terminates every board component with ';', leaving an empty final component.
        board_fens.pop();
        let board_fens = board_fens;
        let boards = board_fens
            .into_iter()
            .map(Board::from_azul_fen)
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;

        let bowl_fens = sections.next().ok_or(ParseGameStateError)?;
        let bowls = bowl_fens
            .trim()
            .split_ascii_whitespace()
            .map(Bowl::from_azul_fen)
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;

        let bag_fen = sections.next().ok_or(ParseGameStateError)?;
        let items = bag_fen
            .chars()
            .map(|c| c.to_string().parse::<Tile>().or(Err(ParseGameStateError)))
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;
        let mut rng = SmallRng::from_seed(rand::rng().random());
        let bag = Bag::new(items, &mut rng);

        let active_player_and_first_token = sections.next().ok_or(ParseGameStateError)?;
        let (active_player, first_token_owner) = match active_player_and_first_token
            .split_whitespace()
            .collect::<Vec<_>>()
            .as_slice()
        {
            [active_player, first_token_owner] => (
                active_player
                    .parse::<usize>()
                    .or(Err(ParseGameStateError))?,
                first_token_owner.parse::<usize>().map(Some).unwrap_or(None),
            ),
            _ => return Err(ParseGameStateError),
        };
        Ok(GameState::builder()
            .active_player(active_player)
            .boards(boards)
            .bowls(bowls)
            .bag(bag)
            .first_token_owner(first_token_owner)
            .build())
    }
}

impl ToAzulFEN for GameState {
    /// Returns the AzulFEN encoding for this game state.
    /// See `interface/azulfen.md` in the repository for the format specification.
    fn to_azul_fen(&self) -> String {
        // Serialize board components.
        let mut azul_fen = String::new();
        for board in self.boards().iter() {
            azul_fen.push_str(&board.fmt_uci_like());
            azul_fen.push(' ');
        }

        // Serialize factory bowls and the centre area.
        azul_fen.push_str("| ");
        for bowl in self.bowls().iter() {
            azul_fen.push_str(&bowl.fmt_uci_like());
            azul_fen.push(' ');
        }

        // Serialize the remaining tile bag.
        azul_fen.push_str("| ");
        azul_fen.push_str(&self.bag().fmt_uci_like());

        // Serialize turn and first-player-token metadata.
        azul_fen.push_str(" | ");
        azul_fen.push_str(&self.active_player().to_string());
        azul_fen.push(' ');
        azul_fen.push_str(&if let Some(t) = self.first_token_owner() {
            t.to_string()
        } else {
            "-".to_string()
        });

        azul_fen.push('\n');
        azul_fen
    }
}
