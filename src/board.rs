use crate::movegen::{IllegalMoveError, Tile};

const BOARD_DIMENSION: usize = 5;

fn count_in_direction(
    placed: &[[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    mut row: isize,
    mut col: isize,
    drow: isize,
    dcol: isize,
) -> usize {
    let mut count = 0;
    loop {
        row += drow;
        col += dcol;
        if row < 0 || col < 0 {
            break;
        }
        if let Some(Some(_)) = placed.get(row as usize).and_then(|r| r.get(col as usize)) {
            count += 1;
        } else {
            break;
        }
    }
    count
}

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
        for (row_idx, row) in self.holds.iter().enumerate() {
            let tiles_in_row = row.iter().filter(|tile| tile.is_some()).count();

            // We have enough tiles to place in this row, let's determine the position
            if tiles_in_row > row_idx {
                let tile_type = row[0].unwrap();
                let col_idx = Board::get_tile_place_col(tile_type, row_idx);
                *self
                    .placed
                    .get_mut(row_idx)
                    .expect("Invalid row")
                    .get_mut(col_idx)
                    .expect("Invalid column") = Some(tile_type);

                // Score newly placed tile
                // We'll walk horizontal and vertically, counting the lengths of each group
                let h_line = 1
                    + count_in_direction(&self.placed, row_idx as isize, col_idx as isize, 0, 1)
                    + count_in_direction(&self.placed, row_idx as isize, col_idx as isize, 0, -1);
                let v_line = 1
                    + count_in_direction(&self.placed, row_idx as isize, col_idx as isize, 1, 0)
                    + count_in_direction(&self.placed, row_idx as isize, col_idx as isize, -1, 0);

                // If the tile is alone, don't double-count it
                self.score += if h_line == 1 && v_line == 1 {
                    1
                } else {
                    // Otherwise, we count the score for axes with more tiles than one
                    (if h_line > 1 { h_line } else { 0 }) + (if v_line > 1 { v_line } else { 0 })
                };
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
