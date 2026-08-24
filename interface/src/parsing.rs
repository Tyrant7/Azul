use crate::format::ProtocolFormat;
use azul_movegen::{
    Bag, Board, Bowl, GameState, Tile,
    board::{BOARD_DIMENSION, BonusTypes},
};

const RNG_STATE_PREFIX: &str = "xoshiro256plusplus:";

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

        let bag_fen = sections.next().ok_or(ParseGameStateError)?.trim();
        let items = bag_fen
            .chars()
            .map(|c| c.to_string().parse::<Tile>().or(Err(ParseGameStateError)))
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;
        let bag = Bag::from_items(items);

        let metadata = sections.next().ok_or(ParseGameStateError)?;
        let metadata = metadata.split_whitespace().collect::<Vec<_>>();
        let (active_player, first_token_owner) = match metadata.as_slice() {
            [active_player, first_token_owner, ..] => (
                active_player
                    .parse::<usize>()
                    .or(Err(ParseGameStateError))?,
                first_token_owner.parse::<usize>().map(Some).unwrap_or(None),
            ),
            _ => return Err(ParseGameStateError),
        };
        let mut seed = None;
        let mut rng_state = None;
        match metadata.as_slice() {
            [_, _] => {}
            [_, _, state] if state.starts_with(RNG_STATE_PREFIX) => {
                rng_state = Some(decode_rng_state(state)?);
            }
            [_, _, seed_token] => {
                seed = Some(seed_token.parse::<u64>().or(Err(ParseGameStateError))?);
            }
            [_, _, seed_token, state] => {
                seed = Some(seed_token.parse::<u64>().or(Err(ParseGameStateError))?);
                rng_state = Some(decode_rng_state(state)?);
            }
            _ => return Err(ParseGameStateError),
        }
        let mut builder = GameState::builder()
            .active_player(active_player)
            .boards(boards)
            .bowls(bowls)
            .bag(bag)
            .first_token_owner(first_token_owner);
        if let Some(seed) = seed {
            builder = builder.set_seed(seed);
        }
        if let Some(rng_state) = rng_state {
            builder = builder.set_rng_state(rng_state);
        }
        builder.build().map_err(|_| ParseGameStateError)
    }
}

impl ToAzulFEN for GameState {
    /// Returns the AzulFEN encoding for this game state.
    ///
    /// The metadata includes the optional game seed and current random state,
    /// allowing exact snapshot restoration.
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

        // Serialize turn, token owner, optional seed, and current RNG state.
        azul_fen.push_str(" | ");
        azul_fen.push_str(&self.active_player().to_string());
        azul_fen.push(' ');
        azul_fen.push_str(&if let Some(t) = self.first_token_owner() {
            t.to_string()
        } else {
            "-".to_string()
        });
        azul_fen.push(' ');
        azul_fen.push_str(
            &self
                .seed()
                .map_or_else(|| "-".to_string(), |seed| seed.to_string()),
        );
        azul_fen.push(' ');
        azul_fen.push_str(RNG_STATE_PREFIX);
        azul_fen.push_str(&encode_rng_state(&self.rng_state()));

        azul_fen.push('\n');
        azul_fen
    }
}

/// Encodes serialized RNG bytes as a compact hexadecimal AzulFEN token.
fn encode_rng_state(state: &[u8]) -> String {
    state.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Decodes a tagged hexadecimal RNG state token.
fn decode_rng_state(token: &str) -> Result<Vec<u8>, ParseGameStateError> {
    let encoded = token
        .strip_prefix(RNG_STATE_PREFIX)
        .ok_or(ParseGameStateError)?;
    if encoded.len() != 64 {
        return Err(ParseGameStateError);
    }
    (0..encoded.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&encoded[index..index + 2], 16).or(Err(ParseGameStateError))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{FromAzulFEN, ToAzulFEN};
    use azul_movegen::GameState;

    #[test]
    fn seeded_azulfen_round_trips_two_through_four_players() {
        for players in 2..=4 {
            let seed = players as u64 * 100;
            let mut original = GameState::new(players, seed).unwrap();
            original.setup_next_round();
            let fen = original.to_azul_fen();
            assert!(fen.contains(&format!("| 0 - {seed} xoshiro256plusplus:")));
            let parsed = GameState::from_azul_fen(&fen).unwrap();

            assert_eq!(parsed.boards().len(), players);
            assert_eq!(parsed.bowls().len(), players * 2 + 2);
            assert_eq!(*parsed.seed(), Some(seed));
            assert_eq!(parsed.rng_state(), original.rng_state());
            assert_eq!(parsed.to_azul_fen(), fen);
        }
    }

    #[test]
    fn azulfen_seed_is_optional_for_backward_compatibility() {
        let original = GameState::new(2, 1234).unwrap();
        let fen = original.to_azul_fen();
        let without_seed = format!(
            "{} | 0 -\n",
            fen.trim_end().split(" | 0 - ").next().unwrap()
        );

        let parsed = GameState::from_azul_fen(&without_seed).unwrap();

        assert_eq!(*parsed.seed(), None);
    }

    #[test]
    fn azulfen_round_trip_reproduces_future_randomness() {
        let mut original = GameState::new(2, 777).unwrap();
        original.setup_next_round();
        let fen = original.to_azul_fen();
        let mut parsed = GameState::from_azul_fen(&fen).unwrap();

        original.setup_next_round();
        parsed.setup_next_round();

        assert_eq!(parsed.bag().items(), original.bag().items());
        for (parsed_bowl, original_bowl) in parsed.bowls().iter().zip(original.bowls()) {
            assert_eq!(parsed_bowl.tiles(), original_bowl.tiles());
        }
        assert_eq!(parsed.rng_state(), original.rng_state());
        assert_eq!(parsed.to_azul_fen(), original.to_azul_fen());
    }
}
