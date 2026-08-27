use crate::{Tile, row::Row};

/// A selection of one tile type from one bowl and its destination on the active board.
///
/// Bowl indices and tile types are zero-based. [`Row::Wall`] also uses a zero-based
/// row index; [`Row::Floor`] represents the penalty area.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Move {
    pub bowl: usize,
    pub tile_type: Tile,
    pub row: Row,
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let row = match self.row {
            Row::Floor => 0,
            Row::Wall(row) => row + 1,
        };
        write!(f, "{:02}{:02}{:02}", self.bowl, self.tile_type, row)
    }
}

/// Returned when a move is not present in the current game's legal move list.
#[derive(Debug)]
pub struct IllegalMoveError;
