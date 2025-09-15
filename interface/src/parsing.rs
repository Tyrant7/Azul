use azul_movegen::{Board, Bowl, GameState};

/// Attempting to parse an invalid AzulFEN or AzulFEN component will produce this error.
#[derive(Debug)]
pub struct ParseGameStateError;

pub trait FromAzulFEN {
    fn from_azul_fen(fen: &str) -> Result<Self, ParseGameStateError>;
}

pub trait ToAzulFEN {
    fn to_azul_fen(&self) -> String;
}

impl FromAzulFEN for Bowl {
    /// Creates a bowl from the given AzulFEN bowl component.
    /// It is important to note that the bowl component is not an entire FEN.
    /// See the [AzulFEN protocol specification](crate::protocol) for details on the format.
    fn from_azul_fen(bowl_fen: &str) -> Result<Self, ParseGameStateError> {
        if bowl_fen.chars().nth(0).ok_or(ParseGameStateError)? == '-' {
            Ok(Bowl { tiles: Vec::new() })
        } else {
            Ok(Bowl {
                tiles: bowl_fen
                    .chars()
                    .map(|c| c.to_string().parse::<Tile>().or(Err(ParseGameStateError)))
                    .collect::<Result<Vec<_>, ParseGameStateError>>()?,
            })
        }
    }
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

impl FromAzulFEN for GameState {
    /// Parses the given AzulFEN into a gamestate.
    /// Will error if the given AzulFEN is invalid.
    /// See the [AzulFEN protocol specification](crate::protocol) for details on the format.
    fn from_azul_fen(azul_fen: &str) -> Result<Self, ParseGameStateError> {
        let mut sections = azul_fen.split("| ");

        let board_fens = sections.next().ok_or(ParseGameStateError)?.trim();
        let mut board_fens: Vec<_> = board_fens.split(";").map(|f| f.trim()).collect();
        // Last FEN will always be empty since we split at ";" and each board ends with one
        board_fens.pop();
        let board_fens = board_fens;
        let boards = board_fens
            .into_iter()
            .map(Board::from_board_fen)
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;

        let bowl_fens = sections.next().ok_or(ParseGameStateError)?;
        let bowls = bowl_fens
            .trim()
            .split_ascii_whitespace()
            .map(Bowl::from_bowl_fen)
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;

        let bag_fen = sections.next().ok_or(ParseGameStateError)?;
        let items = bag_fen
            .chars()
            .map(|c| c.to_string().parse::<Tile>().or(Err(ParseGameStateError)))
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;
        let bag = Bag::new(items);

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
        Ok(GameState {
            active_player,
            boards,
            bowls,
            bag,
            first_token_owner,
        })
    }
}

impl ToAzulFEN for GameState {
    /// Returns the AzulFEN encoding for this game state.
    /// See the [AzulFEN protocol specification](crate::protocol) for details on the format.
    fn to_azul_fen(&self) -> String {
        // Boards
        let mut azul_fen = String::new();
        for board in self.boards.iter() {
            azul_fen.push_str(&board.fmt_uci_like());
            azul_fen.push(' ');
        }

        // Bowls
        azul_fen.push_str("| ");
        for bowl in self.bowls.iter() {
            azul_fen.push_str(&bowl.fmt_uci_like());
            azul_fen.push(' ');
        }

        // Bag
        azul_fen.push_str("| ");
        azul_fen.push_str(&self.bag.fmt_uci_like());

        // Active player and first player token
        azul_fen.push_str(" | ");
        azul_fen.push_str(&self.active_player.to_string());
        azul_fen.push(' ');
        azul_fen.push_str(&if let Some(t) = self.first_token_owner {
            t.to_string()
        } else {
            "-".to_string()
        });

        azul_fen.push('\n');
        azul_fen
    }
}
