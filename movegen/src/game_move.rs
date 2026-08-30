use std::fmt;

use crate::{Tile, row::Row};

/// A selection of one tile type from one source bowl and its destination on the active board.
///
/// Factory indices and tile types are zero-based. The centre is represented by
/// [`BowlChoice::Centre`]. [`Row::Wall`] also uses a zero-based row index;
/// [`Row::Floor`] represents the penalty area.
#[derive(Debug, Clone, PartialEq)]
pub struct Move {
    pub bowl: BowlChoice,
    pub tile_type: Tile,
    pub row: Row,
}

/// Identifies the centre or one of the factory bowls as a move source.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BowlChoice {
    #[default]
    Centre,
    Factory(usize),
}

impl Default for Move {
    /// Creates a move from the centre to the floor using tile type zero.
    fn default() -> Self {
        Self {
            bowl: BowlChoice::Centre,
            tile_type: 0,
            row: Row::Floor,
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let row = match self.row {
            Row::Floor => 0,
            Row::Wall(row) => row + 1,
        };
        let bowl = match self.bowl {
            BowlChoice::Centre => 0,
            BowlChoice::Factory(idx) => idx.checked_add(1).ok_or(fmt::Error)?,
        };
        write!(f, "{:02}{:02}{:02}", bowl, self.tile_type, row)
    }
}

/// Returned when a move is not present in the current game's legal move list.
#[derive(Debug)]
pub struct IllegalMoveError;
