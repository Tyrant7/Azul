use crate::{Tile, game_move::IllegalMoveError, row::Row};

/// The width and height of the place area of the board. A single constant is used as
/// all boards must be a square.
pub const BOARD_DIMENSION: usize = 5;

/// The score bonus given when a board row has been completely filled.
const ROW_BONUS: usize = 2;

/// The score bonus given when a board colmun has been completely filled.
const COLUMN_BONUS: usize = 7;

/// The score bonus given when all boardspaces for a given tile type have been filled.
const TILE_TYPE_BONUS: usize = 10;

/// A player's board.
#[derive(Debug, Clone, Copy, Default)]
pub struct Board {
    holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    bonuses: BonusTypes,
    penalties: usize,
    score: usize,
}

impl Board {
    /// Creates a new `BoardBuilder`.
    pub fn builder() -> BoardBuilder {
        BoardBuilder::default()
    }

    getters! {
        holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
        placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
        bonuses: BonusTypes,
        penalties: usize,
        score: usize,
    }

    /// Returns an iterator over all tiles on this board.
    /// Includes both the held and placed tiles.
    pub fn get_active_tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        self.holds
            .iter()
            .flatten()
            .chain(self.placed.iter().flatten())
            .filter_map(|&t| t)
    }

    /// Returns a vec of all rows which do not yet contain the given tile type, both within
    /// the held and placed positions.
    pub fn get_valid_rows_for_tile_type(&self, tile_type: Tile) -> Vec<Row> {
        let mut valid_rows = Vec::new();
        for (row_idx, hold) in self.holds.iter().enumerate() {
            // If we have a different tile held in this row
            if hold.iter().any(|t| t.is_some_and(|x| x != tile_type)) {
                continue;
            }
            // Or if we have this type of tile already placed somewhere in this row
            if self
                .placed
                .get(row_idx)
                .expect("Invalid row")
                .get(Board::get_tile_place_col(tile_type, row_idx))
                .expect("Invalid columnn")
                .is_some_and(|t| t == tile_type)
            {
                continue;
            }
            valid_rows.push(Row::Wall(row_idx));
        }
        // We can always soak a penalty if we want
        valid_rows.push(Row::Floor);
        valid_rows
    }

    /// Adds the given count of tiles of the given type to the hold positions at the given row index.
    /// Also accepts a penalty to apply to this board.
    /// ## Notes:
    /// - The penalty should only include special cases such as accepting the central tile, and not
    ///   cases such as overflow, which are handled by this method.
    /// - For the sake of simplicity, penalties are measured in tiles, and not score value.
    pub fn hold_tiles(
        &mut self,
        tile_type: Tile,
        tile_count: usize,
        row_idx: Row,
        penalty: usize,
    ) -> Result<(), IllegalMoveError> {
        // If we wanted to put the tiles straight to the floor we'll just soak the penalty
        let row_idx = match row_idx {
            Row::Floor => {
                self.penalties += tile_count;
                return Ok(());
            }
            Row::Wall(idx) => idx,
        };

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

        // We'll also deduct points in certain cases like if we took from the centre first
        self.penalties += penalty;

        Ok(())
    }

    /// Handles all end-of-round actions for this board, including:
    /// - Freeing the tiles in each completed held row
    /// - Adding appropriate tiles to the placed positions
    /// - Ordinary tile scoring
    /// - Bonus scoring and tracking collected bonuses
    /// - Penalty application and penalty resets
    pub fn place_holds(&mut self) {
        for (row_idx, row) in self.holds.iter_mut().enumerate() {
            let tiles_in_row = row.iter().filter(|tile| tile.is_some()).count();

            // We have enough tiles to place in this row
            if tiles_in_row > row_idx {
                // Let's determine the position
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
                let h_line =
                    1 + Board::count_in_direction(
                        &self.placed,
                        row_idx as isize,
                        col_idx as isize,
                        0,
                        1,
                    ) + Board::count_in_direction(
                        &self.placed,
                        row_idx as isize,
                        col_idx as isize,
                        0,
                        -1,
                    );
                let v_line =
                    1 + Board::count_in_direction(
                        &self.placed,
                        row_idx as isize,
                        col_idx as isize,
                        1,
                        0,
                    ) + Board::count_in_direction(
                        &self.placed,
                        row_idx as isize,
                        col_idx as isize,
                        -1,
                        0,
                    );

                // If the tile is alone, don't double-count it
                self.score += if h_line == 1 && v_line == 1 {
                    1
                } else {
                    // Otherwise, we count the score for axes with more tiles than one
                    (if h_line > 1 { h_line } else { 0 }) + (if v_line > 1 { v_line } else { 0 })
                };

                // Now we'll clear the hold for this row
                for tile in row.iter_mut() {
                    *tile = None;
                }
            }
        }

        // Let's apply bonuses that we haven't collected yet
        self.apply_uncollected_bonuses();

        // Let's also apply our penalties
        self.score = self
            .score
            .saturating_sub(Board::get_penalty_point_value(self.penalties));
        self.penalties = 0;
    }

    /// Grants this board score for each bonus it satisfies that has not yet been collected,
    /// then marks such bonuses as collected.
    fn apply_uncollected_bonuses(&mut self) {
        // Start with rows
        for (i, claimed) in self.bonuses.rows.iter_mut().enumerate() {
            if *claimed {
                continue;
            }
            // We haven't collected this bonus yet but this row has been filled,
            // so we'll collect that
            if self.placed[i].iter().all(|x| x.is_some()) {
                self.score += ROW_BONUS;
                *claimed = true;
            }
        }

        // Then columns
        for (i, claimed) in self.bonuses.columns.iter_mut().enumerate() {
            if *claimed {
                continue;
            }
            if self.placed.iter().all(|row| row.get(i).is_some()) {
                self.score += COLUMN_BONUS;
                *claimed = true;
            }
        }

        // And finally, tile types
        for (i, claimed) in self.bonuses.tile_types.iter_mut().enumerate() {
            if *claimed {
                continue;
            }
            if self
                .placed
                .iter()
                .flatten()
                .filter_map(|&t| t)
                .filter(|&t| t == i)
                .count()
                == BOARD_DIMENSION
            {
                self.score += TILE_TYPE_BONUS;
                *claimed = true;
            }
        }
    }

    /// Counts the number of complete horizontal lines in the placed section of this board.
    pub fn count_horizontal_lines(&self) -> usize {
        self.placed
            .iter()
            .filter(|row| row.iter().all(|x| x.is_some()))
            .count()
    }

    /// Score getter
    pub fn get_score(&self) -> usize {
        self.score
    }

    /// Returns the type of tile that can be placed at `row` and `col` on this board.
    pub fn get_tile_type_at_pos(row: usize, col: usize) -> Tile {
        ((col + BOARD_DIMENSION - row) % BOARD_DIMENSION) as Tile
    }

    /// Gets the index of the column where a tile in a given row of a given type should be placed.
    ///
    /// If we consider the board from a top view, tiles simply cycle by index and type:
    /// - 0 1 2 3 4
    /// - 4 0 1 2 3
    /// - 3 4 0 1 2
    /// - ...
    fn get_tile_place_col(tile_type: Tile, row_idx: usize) -> usize {
        (tile_type + row_idx) % BOARD_DIMENSION
    }

    /// Returns the number of penalty points associated with the given number of penalty tiles.  
    fn get_penalty_point_value(penalty_tiles: usize) -> usize {
        [1, 1, 2, 2, 2, 3, 3].iter().take(penalty_tiles).sum()
    }

    /// Counts the number of tiles in any given direction (`drow` and `dcol`) from a source `row` and `col`.
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
}

/// Struct for nicely packaging bonus types together for a board.
/// Each property simply represents whether or not the bonus for that
/// row, column, or tile type has been collected.
#[derive(Debug, Clone, Copy, Default)]
pub struct BonusTypes {
    pub rows: [bool; BOARD_DIMENSION],
    pub columns: [bool; BOARD_DIMENSION],
    pub tile_types: [bool; BOARD_DIMENSION],
}

/// TODO: docstrings for this
#[derive(Default)]
pub struct BoardBuilder {
    holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    bonuses: BonusTypes,
    penalties: usize,
    score: usize,
}

impl BoardBuilder {
    pub fn holds(mut self, holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION]) -> Self {
        self.holds = holds;
        self
    }

    pub fn placed(mut self, placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION]) -> Self {
        self.placed = placed;
        self
    }

    pub fn bonuses(mut self, bonuses: BonusTypes) -> Self {
        self.bonuses = bonuses;
        self
    }

    pub fn penalties(mut self, penalties: usize) -> Self {
        self.penalties = penalties;
        self
    }

    pub fn score(mut self, score: usize) -> Self {
        self.score = score;
        self
    }

    pub fn build(self) -> Board {
        Board {
            holds: self.holds,
            placed: self.placed,
            bonuses: self.bonuses,
            penalties: self.penalties,
            score: self.score,
        }
    }
}
