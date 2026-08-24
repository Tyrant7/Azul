use super::{BOARD_DIMENSION, Board};
use crate::Tile;

/// Tracks which row, column, and tile-type completion bonuses have been collected.
#[derive(Debug, Clone, Copy, Default)]
pub struct BonusTypes {
    pub rows: [bool; BOARD_DIMENSION],
    pub columns: [bool; BOARD_DIMENSION],
    pub tile_types: [bool; BOARD_DIMENSION],
}

/// Builder for constructing a [`Board`] with explicit state.
#[derive(Default)]
pub struct BoardBuilder {
    holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    bonuses: BonusTypes,
    penalties: usize,
    penalty_tiles: Option<usize>,
    score: usize,
}

impl BoardBuilder {
    /// Sets the pattern-line contents.
    pub fn holds(mut self, holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION]) -> Self {
        self.holds = holds;
        self
    }

    /// Sets the wall contents.
    pub fn placed(mut self, placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION]) -> Self {
        self.placed = placed;
        self
    }

    /// Sets the collected completion bonuses.
    pub fn bonuses(mut self, bonuses: BonusTypes) -> Self {
        self.bonuses = bonuses;
        self
    }

    /// Sets the total number of occupied penalty spaces.
    pub fn penalties(mut self, penalties: usize) -> Self {
        self.penalties = penalties;
        self
    }

    /// Sets the number of penalty spaces occupied by physical tiles.
    pub fn penalty_tiles(mut self, penalty_tiles: usize) -> Self {
        self.penalty_tiles = Some(penalty_tiles);
        self
    }

    /// Sets the current score.
    pub fn score(mut self, score: usize) -> Self {
        self.score = score;
        self
    }

    /// Builds a board from the configured fields.
    pub fn build(self) -> Board {
        Board {
            holds: self.holds,
            placed: self.placed,
            bonuses: self.bonuses,
            penalties: self.penalties,
            penalty_tiles: self.penalty_tiles.unwrap_or(self.penalties),
            score: self.score,
        }
    }
}
