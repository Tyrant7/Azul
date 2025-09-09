use crate::protocol::{ParseGameStateError, ProtocolFormat};

/// The alias type for tiles. Since held and placed tiles have no unique properties beyond needing
/// to be differentiable, `usize` was used for the underlying type for tiles.
pub type Tile = usize;

/// A structure for holding groups of tiles according to Azul's bowl rules.
#[derive(Debug, Default)]
pub struct Bowl {
    tiles: Vec<Tile>,
}

impl Bowl {
    /// Creates a bowl from the given FEN portion.
    /// TODO: Link FEN docs for bowls
    pub fn from_bowl_fen(bowl_fen: &str) -> Result<Self, ParseGameStateError> {
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

    /// Assigns this bowl's tiles.
    pub fn fill(&mut self, tiles: Vec<Tile>) {
        self.tiles = tiles;
        self.tiles.sort();
    }

    /// Extends this bowl's tiles.
    pub fn extend(&mut self, tiles: &Vec<Tile>) {
        self.tiles.extend(tiles);
        self.tiles.sort();
    }

    /// Returns the tiles of the given type from this bowl, as well as the remaining tiles. Calling this function
    /// clears this bowl's stored tiles.
    ///
    /// # Arguments
    /// * `tile_type`: the type of tile to select from this bowl.
    ///
    /// # Returns
    /// A tuple containing a `Vec<Tile>` of the tiles matching the given type, and of the remaining tiles, in that order.
    pub fn take_tiles(&mut self, tile_type: Tile) -> (Vec<Tile>, Vec<Tile>) {
        let mut take = Vec::new();
        let mut keep = Vec::new();
        for &tile in self.tiles.iter() {
            if tile == tile_type {
                take.push(tile);
            } else {
                keep.push(tile);
            }
        }
        self.tiles.clear();
        (take, keep)
    }

    /// Returns a `Vec<Tile>` of all unique tile types owned by this bowl.
    pub fn get_tile_types(&self) -> Vec<Tile> {
        let mut tiles = self.tiles.clone();
        tiles.dedup();
        tiles
    }
}

impl Clone for Bowl {
    fn clone(&self) -> Self {
        Self {
            tiles: self.tiles.clone(),
        }
    }
}

impl ProtocolFormat for Bowl {
    fn fmt_human(&self) -> String {
        if self.tiles.is_empty() {
            return String::from("-");
        }
        self.tiles.iter().map(|t| t.to_string()).collect()
    }

    fn fmt_uci_like(&self) -> String {
        self.fmt_human()
    }
}

/// This struct represents a move in gameplay.
/// # Properties
/// * `bowl`: the index of the selected bowl.
/// * `tile_type`: the type of tile taken from the bowl.
/// * `row`: The row wished to hold the tiles taken from the selected bowl.
#[derive(Debug, PartialEq)]
pub struct Move {
    pub bowl: usize,
    pub tile_type: Tile,
    pub row: Row,
}

/// This enum represents a row where tiles can be placed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Row {
    /// The floor row is always a valid position to place tiles, however floor tiles incur penalties and are not scored.
    /// When no valid `Wall` rows remaing, tiles must be placed on the floor row.
    Floor,
    /// Tiles may only be placed on the wall in valid rows. The parameter `usize` represents the index from top to bottom.
    Wall(usize),
}

impl std::fmt::Display for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                Row::Floor => "-".to_string(),
                Row::Wall(i) => i.to_string(),
            }
        )
    }
}

/// Attempting to play a move which is not valid will produce this error.
#[derive(Debug)]
pub struct IllegalMoveError;
