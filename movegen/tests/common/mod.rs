use azul_movegen::board::BOARD_DIMENSION;
use azul_movegen::{Board, Tile};

/// Creates an empty five-by-five wall or pattern-line grid.
pub fn empty_grid() -> [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION] {
    [[None; BOARD_DIMENSION]; BOARD_DIMENSION]
}

/// Creates a board with the supplied wall contents.
pub fn board_with_placed(placed: [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION]) -> Board {
    Board::builder().placed(placed).build()
}

/// Fills one wall row with the tile types assigned to its positions.
pub fn full_row(row: usize) -> [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION] {
    let mut placed = empty_grid();
    for col in 0..BOARD_DIMENSION {
        placed[row][col] = Some(Board::get_tile_type_at_pos(row, col));
    }
    placed
}

/// Fills one wall column with the tile types assigned to its positions.
pub fn full_column(col: usize) -> [[Option<Tile>; BOARD_DIMENSION]; BOARD_DIMENSION] {
    let mut placed = empty_grid();
    for row in 0..BOARD_DIMENSION {
        placed[row][col] = Some(Board::get_tile_type_at_pos(row, col));
    }
    placed
}
