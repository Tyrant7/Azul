use crate::{Tile, row::Row};

/// A move in gameplay.
/// # Properties
/// * `bowl`: the index of the selected bowl.
/// * `tile_type`: the type of tile taken from the bowl.
/// * `row`: The row wished to hold the tiles taken from the selected bowl.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Move {
    pub bowl: usize,
    pub tile_type: Tile,
    pub row: Row,
}

/// Attempting to play a move which is not valid will produce this error.
#[derive(Debug)]
pub struct IllegalMoveError;
