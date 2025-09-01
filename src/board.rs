use crate::movegen::{IllegalMoveError, Tile};

const BOARD_DIMENSION: usize = 5;

#[derive(Debug)]
pub struct Board {
    holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
}

impl Board {
    pub fn new() -> Board {
        Board {
            holds: [[None; BOARD_DIMENSION]; BOARD_DIMENSION],
            placed: [[None; BOARD_DIMENSION]; BOARD_DIMENSION],
        }
    }

    pub fn add_tiles(
        &mut self,
        tile_type: Tile,
        tile_count: usize,
        row_idx: usize,
    ) -> Result<(), IllegalMoveError> {
        // Validate row and existing tiles in that row
        let row = self.holds.get_mut(row_idx).ok_or(IllegalMoveError)?;
        if let Some(t) = row.first().unwrap()
            && *t != tile_type
        {
            return Err(IllegalMoveError);
        }

        // Add tiles to that row, overflowing extra to the penalty section
        let row_capacity = row_idx + 1;
        for row in row.iter_mut().take(tile_count.min(row_capacity)) {
            *row = Some(tile_type);
        }

        let overflow = tile_count.saturating_sub(row_capacity);
        for _ in 0..overflow {
            self.add_penalty_tile();
        }

        Ok(())
    }

    pub fn add_penalty_tile(&mut self) {
        todo!()
    }
}

impl Clone for Board {
    fn clone(&self) -> Self {
        let mut board = Board::new();
        board.holds = self.holds.clone();
        board.placed = self.holds.clone();
        board
    }
}
