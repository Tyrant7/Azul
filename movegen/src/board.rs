use crate::{Tile, game_move::IllegalMoveError, row::Row};

/// The width and height of an Azul wall.
pub const BOARD_DIMENSION: usize = 5;

/// The score bonus for completing a wall row.
const ROW_BONUS: usize = 2;

/// The score bonus for completing a wall column.
const COLUMN_BONUS: usize = 7;

/// The score bonus for placing all five instances of a tile type on the wall.
const TILE_TYPE_BONUS: usize = 10;

mod builder;
pub use builder::{BoardBuilder, BonusTypes};

/// One player's pattern lines, wall, bonuses, penalties, and score.
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

    /// Returns an iterator over all tiles currently held on pattern lines or placed on the wall.
    pub fn get_active_tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        self.holds
            .iter()
            .flatten()
            .chain(self.placed.iter().flatten())
            .filter_map(|&t| t)
    }

    /// Returns every valid destination for `tile_type`, including the floor.
    ///
    /// A wall row is valid when its pattern line is empty or already holds the
    /// requested type and the corresponding wall position lacks that type.
    pub fn get_valid_rows_for_tile_type(&self, tile_type: Tile) -> Vec<Row> {
        let mut valid_rows = Vec::new();
        for (row_idx, hold) in self.holds.iter().enumerate() {
            // A pattern line cannot mix tile types.
            if hold.iter().any(|t| t.is_some_and(|x| x != tile_type)) {
                continue;
            }
            // A tile type may appear only once in each wall row.
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
        // The floor is always available, even when a wall row is legal.
        valid_rows.push(Row::Floor);
        valid_rows
    }

    /// Adds `tile_count` tiles of `tile_type` to a pattern line or the floor.
    ///
    /// `row` identifies the destination. Overflow from a wall pattern line and
    /// the explicit `penalty` are recorded as penalty tiles. `penalty` is for
    /// special penalties such as taking the first-player token; it is measured
    /// in tiles rather than score points.
    pub fn hold_tiles(
        &mut self,
        tile_type: Tile,
        tile_count: usize,
        row_idx: Row,
        penalty: usize,
    ) -> Result<(), IllegalMoveError> {
        // Tiles sent directly to the floor do not enter a pattern line.
        let row_idx = match row_idx {
            Row::Floor => {
                self.penalties += tile_count;
                return Ok(());
            }
            Row::Wall(idx) => idx,
        };

        // Validate the row index and the existing pattern-line tile type.
        let row = self.holds.get_mut(row_idx).ok_or(IllegalMoveError)?;
        if let Some(t) = row.first().unwrap()
            && *t != tile_type
        {
            return Err(IllegalMoveError);
        }

        // Fill the pattern line and send overflow to the penalty area.
        let row_capacity = row_idx + 1;
        for row in row.iter_mut().take(tile_count.min(row_capacity)) {
            *row = Some(tile_type);
        }

        let overflow = tile_count.saturating_sub(row_capacity);
        for _ in 0..overflow {
            self.penalties += 1;
        }

        // Apply explicit penalties, such as the first-player-token penalty.
        self.penalties += penalty;

        Ok(())
    }

    /// Resolves completed pattern lines at the end of a round.
    ///
    /// This places one tile from each completed line onto the wall, scores newly
    /// placed tiles and newly earned bonuses, applies penalty tiles, and clears
    /// the resolved lines and penalties.
    pub fn place_holds(&mut self) {
        for (row_idx, row) in self.holds.iter_mut().enumerate() {
            let tiles_in_row = row.iter().filter(|tile| tile.is_some()).count();

            // A pattern line is complete when it contains more tiles than its zero-based row index.
            if tiles_in_row > row_idx {
                // Determine the fixed wall position for this tile type and row.
                let tile_type = row[0].unwrap();
                let col_idx = Board::get_tile_place_col(tile_type, row_idx);
                *self
                    .placed
                    .get_mut(row_idx)
                    .expect("Invalid row")
                    .get_mut(col_idx)
                    .expect("Invalid column") = Some(tile_type);

                // Score the newly placed tile by counting contiguous horizontal and vertical lines.
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

                // An isolated tile scores once rather than once per axis.
                self.score += if h_line == 1 && v_line == 1 {
                    1
                } else {
                    // Otherwise, add each non-isolated axis length.
                    (if h_line > 1 { h_line } else { 0 }) + (if v_line > 1 { v_line } else { 0 })
                };

                // The remaining pattern-line tiles are discarded after one reaches the wall.
                for tile in row.iter_mut() {
                    *tile = None;
                }
            }
        }

        // Apply bonuses that have been completed since the previous round.
        self.apply_uncollected_bonuses();

        // Convert penalty tiles to points and reset the penalty count.
        self.score = self
            .score
            .saturating_sub(Board::get_penalty_point_value(self.penalties));
        self.penalties = 0;
    }

    /// Grants this board score for each bonus it satisfies that has not yet been collected,
    /// then marks such bonuses as collected.
    fn apply_uncollected_bonuses(&mut self) {
        // Check row bonuses.
        for (i, claimed) in self.bonuses.rows.iter_mut().enumerate() {
            if *claimed {
                continue;
            }
            // Award an unclaimed bonus when its row is complete.
            if self.placed[i].iter().all(|x| x.is_some()) {
                self.score += ROW_BONUS;
                *claimed = true;
            }
        }

        // Check column bonuses.
        for (i, claimed) in self.bonuses.columns.iter_mut().enumerate() {
            if *claimed {
                continue;
            }
            if self.placed.iter().all(|row| row[i].is_some()) {
                self.score += COLUMN_BONUS;
                *claimed = true;
            }
        }

        // Check tile-type bonuses.
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

    /// Counts complete horizontal lines in the wall.
    pub fn count_horizontal_lines(&self) -> usize {
        self.placed
            .iter()
            .filter(|row| row.iter().all(|x| x.is_some()))
            .count()
    }

    /// Returns the board's current score.
    pub fn get_score(&self) -> usize {
        self.score
    }

    /// Returns the tile type assigned to a zero-based wall position.
    pub fn get_tile_type_at_pos(row: usize, col: usize) -> Tile {
        ((col + BOARD_DIMENSION - row) % BOARD_DIMENSION) as Tile
    }

    /// Returns the zero-based wall column for a tile type in a pattern-line row.
    ///
    /// If we consider the board from a top view, tiles simply cycle by index and type:
    /// - 0 1 2 3 4
    /// - 4 0 1 2 3
    /// - 3 4 0 1 2
    /// - ...
    fn get_tile_place_col(tile_type: Tile, row_idx: usize) -> usize {
        (tile_type + row_idx) % BOARD_DIMENSION
    }

    /// Returns the penalty score for up to seven floor tiles.
    ///
    /// The board stores penalty tiles as a count; the floor scoring table has
    /// seven entries, so additional tiles do not add further points here.
    fn get_penalty_point_value(penalty_tiles: usize) -> usize {
        [1, 1, 2, 2, 2, 3, 3].iter().take(penalty_tiles).sum()
    }

    /// Counts contiguous placed tiles from a source position in one direction.
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
