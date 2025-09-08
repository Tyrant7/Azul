use crate::{
    bowl::{IllegalMoveError, Row, Tile},
    protocol::{ParseGameStateError, ProtocolFormat},
};

pub const BOARD_DIMENSION: usize = 5;

const ROW_BONUS: usize = 2;
const COLUMN_BONUS: usize = 7;
const TILE_TYPE_BONUS: usize = 10;

#[derive(Debug, Clone, Copy)]
pub struct Board {
    holds: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION],
    bonuses: BonusTypes,
    penalties: usize,
    score: usize,
}

impl Board {
    pub fn new() -> Self {
        Board {
            holds: [[None; BOARD_DIMENSION]; BOARD_DIMENSION],
            placed: [[None; BOARD_DIMENSION]; BOARD_DIMENSION],
            bonuses: BonusTypes {
                rows: [false; BOARD_DIMENSION],
                columns: [false; BOARD_DIMENSION],
                tile_types: [false; BOARD_DIMENSION],
            },
            penalties: 0,
            score: 0,
        }
    }

    pub fn from_board_fen(board_fen: &str) -> Result<Self, ParseGameStateError> {
        let mut board = Board::new();
        let parts: Vec<_> = board_fen.split_whitespace().collect();
        match parts.as_slice() {
            [
                placed,
                held,
                bonus_rows,
                bonus_cols,
                bonus_tile_types,
                score,
                penalties,
            ] => {
                // Placed
                let mut y = 0;
                let mut x = 0;
                for p in placed.chars() {
                    if let Ok(step) = p.to_string().parse::<usize>() {
                        x += step;
                    } else if p == '-' {
                        board.holds[y][x] = Some(Board::get_tile_type_at_pos(y, x));
                        x += 1;
                    }
                    if x >= BOARD_DIMENSION {
                        y += 1;
                        x = 0;
                    }
                }

                // Held
                for (i, h) in held.chars().collect::<Vec<_>>().chunks(2).enumerate() {
                    let tile_type = h[0]
                        .to_string()
                        .parse::<Tile>()
                        .or(Err(ParseGameStateError))?;
                    let tile_count = h[1]
                        .to_string()
                        .parse::<Tile>()
                        .or(Err(ParseGameStateError))?;
                    if tile_count == 0 {
                        continue;
                    }
                    for n in 0..tile_count {
                        board.holds[i][n] = Some(tile_type);
                    }
                }

                // Bonuses
                board.bonuses = BonusTypes {
                    rows: bonus_rows
                        .chars()
                        .map(|c| c == '1')
                        .collect::<Vec<_>>()
                        .try_into()
                        .or(Err(ParseGameStateError))?,
                    columns: bonus_cols
                        .chars()
                        .map(|c| c == '1')
                        .collect::<Vec<_>>()
                        .try_into()
                        .or(Err(ParseGameStateError))?,
                    tile_types: bonus_tile_types
                        .chars()
                        .map(|c| c == '1')
                        .collect::<Vec<_>>()
                        .try_into()
                        .or(Err(ParseGameStateError))?,
                };

                // Score and penalties
                board.score = score.parse().or(Err(ParseGameStateError))?;
                board.penalties = penalties.parse().or(Err(ParseGameStateError))?;
            }
            _ => return Err(ParseGameStateError),
        };
        Ok(board)
    }

    pub fn get_active_tiles(&self) -> impl Iterator<Item = Tile> + '_ {
        self.holds
            .iter()
            .flatten()
            .chain(self.placed.iter().flatten())
            .filter_map(|&t| t)
    }

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

    pub fn count_horizontal_lines(&self) -> usize {
        self.placed
            .iter()
            .filter(|row| row.iter().all(|x| x.is_some()))
            .count()
    }

    pub fn get_score(&self) -> usize {
        self.score
    }

    fn get_tile_place_col(tile_type: Tile, row_idx: usize) -> usize {
        // Tiles simply cycle by index
        // 0 1 2 3 4
        // 4 0 1 2 3
        // 3 4 0 1 2
        // ...
        (tile_type + row_idx) % BOARD_DIMENSION
    }

    fn get_tile_type_at_pos(row: usize, col: usize) -> Tile {
        ((col + BOARD_DIMENSION - row) % BOARD_DIMENSION) as Tile
    }

    fn get_penalty_point_value(penalty_tiles: usize) -> usize {
        [1, 1, 2, 2, 2, 3, 3].iter().take(penalty_tiles).sum()
    }

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

impl ProtocolFormat for Board {
    fn fmt_human(&self) -> String {
        let mut output = String::new();
        for ((h_idx, hold), row) in self.holds.iter().enumerate().zip(self.placed) {
            output.push_str(&(h_idx + 1).to_string());
            output.push_str(&"  ".repeat(BOARD_DIMENSION - h_idx));
            for h in 0..h_idx + 1 {
                if let Some(h) = hold.get(h).and_then(|x| *x) {
                    output.push_str(&h.to_string());
                    output.push(' ');
                } else {
                    output.push_str(". ");
                }
            }
            output.push_str(" | ");
            for p in 0..BOARD_DIMENSION {
                if let Some(p) = row.get(p).and_then(|x| *x) {
                    output.push_str(&p.to_string());
                    output.push(' ');
                } else {
                    output.push_str(". ");
                }
            }
            output.push('\n');
        }
        output.push_str(&format!("score: {}\n", self.score));
        output.push_str(&format!("penalties: {}", self.penalties));
        output.push('\n');
        output.push('\n');
        output
    }

    fn fmt_uci_like(&self) -> String {
        // Format according to AzulFEN specifications
        let mut output = String::new();

        // Placed
        let mut counter = 0;
        for row in self.placed {
            for tile in row {
                if tile.is_some() {
                    if counter > 0 {
                        output.push_str(&counter.to_string());
                    }
                    output.push('-');
                    counter = 0;
                } else {
                    counter += 1;
                }
            }
            if counter > 0 {
                output.push_str(&counter.to_string());
            }
            counter = 0;
            output.push('/');
        }
        output.pop();

        // Holds
        output.push(' ');
        for row in &self.holds {
            let mut tiles = row.iter().flatten();
            if let Some(t) = tiles.next() {
                let count = 1 + tiles.count();
                output.push_str(&t.to_string());
                output.push_str(&count.to_string());
            } else {
                output.push_str("00");
            }
        }

        // Bonuses
        output.push(' ');
        for row in self.bonuses.rows {
            output.push_str(&if row { 1 } else { 0 }.to_string());
        }
        output.push(' ');
        for column in self.bonuses.columns {
            output.push_str(&if column { 1 } else { 0 }.to_string());
        }
        output.push(' ');
        for tile_type in self.bonuses.tile_types {
            output.push_str(&if tile_type { 1 } else { 0 }.to_string());
        }

        // Score and penalties
        output.push(' ');
        output.push_str(&self.score.to_string());
        output.push(' ');
        output.push_str(&self.penalties.to_string());

        // End marker
        output.push_str(" ;");
        output
    }
}

#[derive(Debug, Clone, Copy)]
struct BonusTypes {
    pub rows: [bool; BOARD_DIMENSION],
    pub columns: [bool; BOARD_DIMENSION],
    pub tile_types: [bool; BOARD_DIMENSION],
}
