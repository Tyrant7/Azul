use crate::movegen::{IllegalMoveError, Tile};

const BOARD_DIMENSION: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct Board {
    holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    penalties: usize,
    score: usize,
}

impl Board {
    pub fn new() -> Board {
        Board {
            holds: [[None; BOARD_DIMENSION]; BOARD_DIMENSION],
            placed: [[None; BOARD_DIMENSION]; BOARD_DIMENSION],
            penalties: 0,
            score: 0,
        }
    }

    pub fn get_active_tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        self.holds
            .iter()
            .flatten()
            .chain(self.placed.iter().flatten())
            .filter_map(|&t| t)
    }

    pub fn hold_tiles(
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
            self.penalties += 1;
        }

        Ok(())
    }

    pub fn place_holds(&mut self) {
        for (i, row) in self.holds.iter().enumerate() {
            let tiles_in_row = row.iter().filter(|tile| tile.is_some()).count();

            // We have enough tiles to place in this row, let's determine the position
            if tiles_in_row > i {
                let tile_type = row[0].unwrap();
                let tile_col = Board::get_tile_place_col(tile_type, i);
                *self
                    .placed
                    .get_mut(i)
                    .expect("Invalid row")
                    .get_mut(tile_col)
                    .expect("Invalid column") = Some(tile_type);

                // TODO: Score newly placed tile
            }
        }

        // Let's also apply our penalties
        self.score -= self.penalties;
        self.penalties = 0;
    }

    fn get_tile_place_col(tile_type: Tile, row_idx: usize) -> usize {
        // Tiles simply cycle by index
        // 0 1 2 3 4
        // 4 0 1 2 3
        // 3 4 0 1 2
        // ...
        (tile_type + row_idx) % BOARD_DIMENSION
    }
}
