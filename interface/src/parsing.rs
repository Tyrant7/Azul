//! AzulFEN serialization and deserialization for movegen values.

use crate::format::ProtocolFormat;
use azul_movegen::{
    Bag, Board, Bowl, GameState, Tile,
    board::{BOARD_DIMENSION, BonusTypes},
};

/// The currently supported AzulFEN wire-format version.
pub const AZULFEN_VERSION: &str = "azulfen:v1";

const RNG_STATE_PREFIX: &str = "xoshiro256plusplus:";

/// Attempting to parse an invalid AzulFEN or AzulFEN component will produce this error.
#[derive(Debug)]
pub struct ParseGameStateError;

/// Constructs a value from an [AzulFEN](../azulfen.md) string or component.
pub trait FromAzulFEN: Sized {
    /// Parses an AzulFEN representation.
    ///
    /// Implementations may accept a complete position or a component when
    /// the target type represents one part of a position.
    fn from_azul_fen(fen: &str) -> Result<Self, ParseGameStateError>;
}

/// Serializes a value to its [AzulFEN](../azulfen.md) representation.
pub trait ToAzulFEN {
    /// Returns the AzulFEN representation.
    ///
    /// The returned string is suitable for persistence or use in the UAI
    /// `position fen` command.
    fn to_azul_fen(&self) -> String;
}

impl FromAzulFEN for Bowl {
    /// Creates a bowl from the given AzulFEN bowl component.
    /// It is important to note that the bowl component is not an entire FEN.
    /// See the [AzulFEN format specification](../azulfen.md).
    fn from_azul_fen(bowl_fen: &str) -> Result<Self, ParseGameStateError> {
        if bowl_fen == "-" {
            Ok(Bowl::default())
        } else if bowl_fen.is_empty()
            || !bowl_fen.is_ascii()
            || !bowl_fen.bytes().all(|byte| (b'0'..=b'4').contains(&byte))
        {
            Err(ParseGameStateError)
        } else {
            let mut previous = None;
            let mut tiles = Vec::with_capacity(bowl_fen.len());
            for byte in bowl_fen.bytes() {
                if previous.is_some_and(|tile| tile > byte) {
                    return Err(ParseGameStateError);
                }
                previous = Some(byte);
                tiles.push((byte - b'0') as Tile);
            }
            Ok(Bowl::from_tiles(tiles))
        }
    }
}

impl FromAzulFEN for Board {
    /// Generates a board matching the given board component of a given AzulFEN.
    /// It is important to note that the board component is not an entire FEN.
    /// See the [AzulFEN format specification](../azulfen.md).
    fn from_azul_fen(board_fen: &str) -> Result<Self, ParseGameStateError> {
        let parts: Vec<_> = board_fen.split(' ').collect();
        if parts.len() != 8 || parts.iter().any(|part| part.is_empty()) {
            return Err(ParseGameStateError);
        }
        let [
            placed_parts,
            held,
            bonus_rows,
            bonus_cols,
            bonus_tile_types,
            score,
            penalties,
            penalty_tiles,
        ] = parts.as_slice()
        else {
            return Err(ParseGameStateError);
        };

        // Decode the wall using run-length counts for empty positions.
        let placed_rows: Vec<_> = placed_parts.split('/').collect();
        if placed_rows.len() != BOARD_DIMENSION {
            return Err(ParseGameStateError);
        }
        let mut placed = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
        for (y, placed_row) in placed_rows.iter().enumerate() {
            let mut x = 0;
            for character in placed_row.chars() {
                match character {
                    '-' => {
                        if x >= BOARD_DIMENSION {
                            return Err(ParseGameStateError);
                        }
                        placed[y][x] = Some(Board::get_tile_type_at_pos(y, x));
                        x += 1;
                    }
                    character if character.is_ascii_digit() => {
                        let step = character.to_digit(10).unwrap() as usize;
                        if step == 0 || x + step > BOARD_DIMENSION {
                            return Err(ParseGameStateError);
                        }
                        x += step;
                    }
                    _ => return Err(ParseGameStateError),
                }
            }
            if x != BOARD_DIMENSION {
                return Err(ParseGameStateError);
            }
        }

        // Decode each pattern line as a tile-type/count pair.
        if held.len() != BOARD_DIMENSION * 2 || !held.is_ascii() {
            return Err(ParseGameStateError);
        }
        let mut holds = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
        for (row_idx, row) in holds.iter_mut().enumerate().take(BOARD_DIMENSION) {
            let tile_type = held.as_bytes()[row_idx * 2];
            let tile_count = held.as_bytes()[row_idx * 2 + 1];
            if !(b'0'..=b'4').contains(&tile_type) || !(b'0'..=b'5').contains(&tile_count) {
                return Err(ParseGameStateError);
            }
            let tile_type = (tile_type - b'0') as Tile;
            let tile_count = (tile_count - b'0') as usize;
            if tile_count > row_idx + 1 || (tile_count == 0 && tile_type != 0) {
                return Err(ParseGameStateError);
            }
            for hold in row.iter_mut().take(tile_count) {
                *hold = Some(tile_type);
            }
        }

        // Decode collected row, column, and tile-type bonuses.
        let bonuses = BonusTypes {
            rows: parse_bonus_field(bonus_rows)?,
            columns: parse_bonus_field(bonus_cols)?,
            tile_types: parse_bonus_field(bonus_tile_types)?,
        };

        // Decode the score, occupied penalty spaces, and physical penalty tiles.
        let score = parse_decimal(score)?;
        let penalties = parse_decimal(penalties)?;
        let penalty_tiles = parse_decimal(penalty_tiles)?;
        if penalty_tiles > penalties {
            return Err(ParseGameStateError);
        }

        Ok(Board::builder()
            .placed(placed)
            .holds(holds)
            .bonuses(bonuses)
            .score(score)
            .penalties(penalties)
            .penalty_tiles(penalty_tiles)
            .build())
    }
}

/// Parses a non-negative decimal AzulFEN field.
fn parse_decimal(field: &str) -> Result<usize, ParseGameStateError> {
    if field.is_empty() || !field.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(ParseGameStateError);
    }
    field.parse().map_err(|_| ParseGameStateError)
}

/// Parses one canonical five-bit bonus field.
fn parse_bonus_field(field: &str) -> Result<[bool; BOARD_DIMENSION], ParseGameStateError> {
    if field.len() != BOARD_DIMENSION || !field.bytes().all(|byte| matches!(byte, b'0' | b'1')) {
        return Err(ParseGameStateError);
    }
    field
        .bytes()
        .map(|byte| byte == b'1')
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| ParseGameStateError)
}

impl FromAzulFEN for GameState {
    /// Parses the given AzulFEN into a gamestate.
    /// Will error if the given AzulFEN is invalid.
    /// See the [AzulFEN format specification](../azulfen.md).
    fn from_azul_fen(azul_fen: &str) -> Result<Self, ParseGameStateError> {
        let line = azul_fen.strip_suffix('\n').ok_or(ParseGameStateError)?;
        if line.contains('\n') || line.contains('\r') {
            return Err(ParseGameStateError);
        }
        let body = line
            .strip_prefix(AZULFEN_VERSION)
            .and_then(|body| body.strip_prefix(' '))
            .ok_or(ParseGameStateError)?;
        let sections: Vec<_> = body.split(" | ").collect();
        if sections.len() != 4
            || sections[0].is_empty()
            || sections[1].is_empty()
            || sections[3].is_empty()
        {
            return Err(ParseGameStateError);
        }

        let board_section = sections[0].strip_suffix(" ;").ok_or(ParseGameStateError)?;
        let board_fens: Vec<_> = board_section.split(" ; ").collect();
        if board_fens.iter().any(|fen| fen.is_empty()) {
            return Err(ParseGameStateError);
        }
        let boards = board_fens
            .into_iter()
            .map(Board::from_azul_fen)
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;

        let bowl_fens: Vec<_> = sections[1].split(' ').collect();
        if bowl_fens.iter().any(|fen| fen.is_empty()) {
            return Err(ParseGameStateError);
        }
        let bowls = bowl_fens
            .into_iter()
            .map(Bowl::from_azul_fen)
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;

        let bag_fen = sections[2];
        if !bag_fen.is_ascii() || !bag_fen.bytes().all(|byte| (b'0'..=b'4').contains(&byte)) {
            return Err(ParseGameStateError);
        }
        let items = bag_fen.bytes().map(|byte| (byte - b'0') as Tile).collect();
        let bag = Bag::from_items(items);

        let metadata: Vec<_> = sections[3].split(' ').collect();
        if metadata.len() != 5 || metadata.iter().any(|field| field.is_empty()) {
            return Err(ParseGameStateError);
        }
        let active_player = parse_decimal(metadata[0])?;
        let first_token_owner = match metadata[1] {
            "-" => None,
            owner => Some(parse_decimal(owner)?),
        };
        let seed = match metadata[2] {
            "-" => None,
            seed => Some(parse_decimal(seed)? as u64),
        };
        let rng_state = decode_rng_state(metadata[3])?;
        let discarded_tiles = parse_decimal(metadata[4])?;
        let mut builder = GameState::builder()
            .active_player(active_player)
            .boards(boards)
            .bowls(bowls)
            .bag(bag)
            .first_token_owner(first_token_owner);
        if let Some(seed) = seed {
            builder = builder.set_seed(seed);
        }
        builder = builder
            .set_rng_state(rng_state)
            .discarded_tiles(discarded_tiles);
        builder.build().map_err(|_| ParseGameStateError)
    }
}

impl ToAzulFEN for GameState {
    /// Returns the AzulFEN encoding for this game state.
    ///
    /// The metadata includes the optional game seed, current random state, and
    /// discard count, allowing exact snapshot restoration and tile accounting.
    /// See the [AzulFEN format specification](../azulfen.md).
    fn to_azul_fen(&self) -> String {
        // Serialize board components.
        let mut azul_fen = String::from(AZULFEN_VERSION);
        azul_fen.push(' ');
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
        azul_fen.push(' ');
        azul_fen.push_str(&self.discarded_tiles().to_string());

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
    if encoded.len() != 64 || !encoded.is_ascii() {
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
    use super::{AZULFEN_VERSION, FromAzulFEN, ToAzulFEN};
    use azul_movegen::{Bag, Board, Bowl, GameState, Row};

    #[test]
    fn seeded_azulfen_round_trips_two_through_four_players() {
        for players in 2..=4 {
            let seed = players as u64 * 100;
            let mut original = GameState::new(players, seed).unwrap();
            original.setup_next_round();
            let fen = original.to_azul_fen();
            assert!(fen.starts_with("azulfen:v1 "));
            assert!(fen.ends_with('\n'));
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
    fn unseeded_snapshots_round_trip_with_exact_rng_state() {
        let original = GameState::builder()
            .boards(vec![Board::default(); 2])
            .bowls(vec![Bowl::default(); 6])
            .bag(Bag::default())
            .build()
            .unwrap();
        let fen = original.to_azul_fen();
        let parsed = GameState::from_azul_fen(&fen).unwrap();

        assert_eq!(*parsed.seed(), None);
        assert_eq!(parsed.rng_state(), original.rng_state());
        assert_eq!(parsed.to_azul_fen(), fen);
    }

    #[test]
    fn azulfen_requires_version_and_complete_metadata() {
        let original = GameState::new(2, 1234).unwrap();
        let fen = original.to_azul_fen();
        let unversioned = fen.strip_prefix(&format!("{AZULFEN_VERSION} ")).unwrap();
        let unknown_version = fen.replacen(AZULFEN_VERSION, "azulfen:v2", 1);
        let missing_newline = fen.trim_end();

        assert!(GameState::from_azul_fen(unversioned).is_err());
        assert!(GameState::from_azul_fen(&unknown_version).is_err());
        assert!(GameState::from_azul_fen(missing_newline).is_err());
    }

    #[test]
    fn azulfen_rejects_malformed_board_and_metadata_fields() {
        let mut original = GameState::new(2, 1234).unwrap();
        original.setup_next_round();
        let fen = original.to_azul_fen();

        let malformed_wall = fen.replacen("5/5/5/5/5", "5/5/5/5/6", 1);
        let malformed_holds = fen.replacen("0000000000", "0090000000", 1);
        let malformed_bonus = fen.replacen("00000", "0000x", 1);
        let malformed_metadata = fen.replacen("\n", " extra\n", 1);

        for malformed in [
            malformed_wall,
            malformed_holds,
            malformed_bonus,
            malformed_metadata,
        ] {
            assert!(
                GameState::from_azul_fen(&malformed).is_err(),
                "accepted malformed FEN: {malformed}"
            );
        }
    }

    #[test]
    fn component_parsers_reject_noncanonical_values() {
        for bowl in ["", "-0", "210", "5", "0x"] {
            assert!(Bowl::from_azul_fen(bowl).is_err(), "accepted bowl {bowl}");
        }

        for board in [
            "5/5/5/5/6 0000000000 00000 00000 00000 0 0 0",
            "5/5/5/5/5 0090000000 00000 00000 00000 0 0 0",
            "5/5/5/5/5 0000000000 0000x 00000 00000 0 0 0",
            "5/5/5/5/5 0000000000 00000 00000 00000 0 0 1",
        ] {
            assert!(
                Board::from_azul_fen(board).is_err(),
                "accepted board {board}"
            );
        }
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

    #[test]
    fn azulfen_round_trip_preserves_penalty_and_discard_tracking() {
        let mut board = Board::default();
        board.hold_tiles(2, 2, Row::Floor, 1).unwrap();
        let original = GameState::builder()
            .boards(vec![board, Board::default()])
            .bowls(vec![Bowl::default(); 6])
            .bag(Bag::default())
            .first_token_owner(Some(0))
            .set_seed(42)
            .discarded_tiles(7)
            .build()
            .unwrap();

        let fen = original.to_azul_fen();
        let parsed = GameState::from_azul_fen(&fen).unwrap();

        assert_eq!(*parsed.boards()[0].penalties(), 3);
        assert_eq!(*parsed.boards()[0].penalty_tiles(), 2);
        assert_eq!(*parsed.discarded_tiles(), 7);
        assert_eq!(parsed.to_azul_fen(), fen);
    }
}
