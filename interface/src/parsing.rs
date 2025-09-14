use azul_movegen::{Board, GameState};

/// Attempting to parse an invalid AzulFEN or AzulFEN component will produce this error.
#[derive(Debug)]
pub struct ParseGameStateError;

trait FromAzulFEN {
    pub fn from_azul_fen(fen: &str) -> Result<Self, ParseGameStateError>;
}

trait ToAzulFEN {
    pub fn to_azul_fen(fen: &str) -> String;
}

impl FromAzulFEN for Board {
    /// Generates a board matching the given board component of a given AzulFEN.
    /// It is important to note that the board component is not an entire FEN.
    /// See the [AzulFEN protocol specification](crate::protocol) for details on the format.
    fn from_azul_fen(board_fen: &str) -> Result<Self, ParseGameStateError> {
        let mut board = Board::default();
        let parts: Vec<_> = board_fen.split_whitespace().collect();
        match parts.as_slice() {
            [
                placed,
                held,
                bonus_rows,
                bonus_cols,
                bonus_tile_types,
                score,
                penalties,
            ] => {
                // Placed
                let mut y = 0;
                let mut x = 0;
                for p in placed.chars() {
                    if let Ok(step) = p.to_string().parse::<usize>() {
                        x += step;
                    } else if p == '-' {
                        board.holds[y][x] = Some(Board::get_tile_type_at_pos(y, x));
                        x += 1;
                    }
                    if x >= BOARD_DIMENSION {
                        y += 1;
                        x = 0;
                    }
                }

                // Held
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
                        board.holds[i][n] = Some(tile_type);
                    }
                }

                // Bonuses
                board.bonuses = BonusTypes {
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
                };

                // Score and penalties
                board.score = score.parse().or(Err(ParseGameStateError))?;
                board.penalties = penalties.parse().or(Err(ParseGameStateError))?;
            }
            _ => return Err(ParseGameStateError),
        };
        Ok(board)
    }
}
