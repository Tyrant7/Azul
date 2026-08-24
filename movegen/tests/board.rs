mod common;

use azul_movegen::board::{BOARD_DIMENSION, BonusTypes};
use azul_movegen::{Board, Row};
use common::{board_with_placed, empty_grid, full_column, full_row};

#[test]
fn default_board_has_empty_state() {
    let board = Board::default();

    assert_eq!(board.holds(), &empty_grid());
    assert_eq!(board.placed(), &empty_grid());
    assert_eq!(board.bonuses().rows, [false; BOARD_DIMENSION]);
    assert_eq!(board.bonuses().columns, [false; BOARD_DIMENSION]);
    assert_eq!(board.bonuses().tile_types, [false; BOARD_DIMENSION]);
    assert_eq!(*board.penalties(), 0);
    assert_eq!(*board.score(), 0);
}

#[test]
fn builder_sets_all_board_fields() {
    let mut holds = empty_grid();
    holds[2][0] = Some(3);
    let mut placed = empty_grid();
    placed[1][1] = Some(2);
    let bonuses = BonusTypes {
        rows: [true, false, false, false, false],
        columns: [false, true, false, false, false],
        tile_types: [false, false, true, false, false],
    };

    let board = Board::builder()
        .holds(holds)
        .placed(placed)
        .bonuses(bonuses)
        .penalties(3)
        .score(12)
        .build();

    assert_eq!(*board.holds(), holds);
    assert_eq!(*board.placed(), placed);
    assert_eq!(board.bonuses().rows, bonuses.rows);
    assert_eq!(board.bonuses().columns, bonuses.columns);
    assert_eq!(board.bonuses().tile_types, bonuses.tile_types);
    assert_eq!(*board.penalties(), 3);
    assert_eq!(*board.score(), 12);
}

#[test]
fn tile_positions_follow_the_cyclic_wall_pattern() {
    let expected = [
        [0, 1, 2, 3, 4],
        [4, 0, 1, 2, 3],
        [3, 4, 0, 1, 2],
        [2, 3, 4, 0, 1],
        [1, 2, 3, 4, 0],
    ];

    for (row, expected_row) in expected.iter().enumerate() {
        for (col, &tile_type) in expected_row.iter().enumerate() {
            assert_eq!(Board::get_tile_type_at_pos(row, col), tile_type);
        }
    }
}

#[test]
fn active_tiles_include_holds_then_placed_tiles() {
    let mut holds = empty_grid();
    holds[0][0] = Some(2);
    let mut placed = empty_grid();
    placed[1][1] = Some(3);
    let board = Board::builder().holds(holds).placed(placed).build();

    assert_eq!(board.get_active_tiles().collect::<Vec<_>>(), vec![2, 3]);
}

#[test]
fn valid_rows_exclude_conflicting_holds_and_completed_tile_positions() {
    let mut holds = empty_grid();
    holds[1][0] = Some(2);
    let board = Board::builder().holds(holds).build();

    let valid_for_two = board.get_valid_rows_for_tile_type(2);
    let valid_for_three = board.get_valid_rows_for_tile_type(3);
    assert!(valid_for_two.contains(&Row::Wall(1)));
    assert!(!valid_for_three.contains(&Row::Wall(1)));
    assert!(valid_for_three.contains(&Row::Floor));

    let mut placed = empty_grid();
    placed[0][0] = Some(0);
    let board = board_with_placed(placed);
    let valid = board.get_valid_rows_for_tile_type(0);
    assert!(!valid.contains(&Row::Wall(0)));
    assert_eq!(valid.len(), BOARD_DIMENSION);
}

#[test]
fn hold_tiles_fills_pattern_lines_and_rejects_conflicts() {
    let mut board = Board::default();

    board.hold_tiles(2, 3, Row::Wall(2), 0).unwrap();
    assert_eq!(
        board.holds()[2].to_vec(),
        vec![Some(2), Some(2), Some(2), None, None]
    );
    assert_eq!(*board.penalties(), 0);

    assert!(board.hold_tiles(3, 1, Row::Wall(2), 0).is_err());
    assert!(
        board
            .hold_tiles(3, 1, Row::Wall(BOARD_DIMENSION), 0)
            .is_err()
    );
}

#[test]
fn hold_tiles_sends_overflow_and_floor_tiles_to_penalties() {
    let mut board = Board::default();
    board.hold_tiles(1, 3, Row::Wall(0), 0).unwrap();

    assert_eq!(
        board.holds()[0].to_vec(),
        vec![Some(1), None, None, None, None]
    );
    assert_eq!(*board.penalties(), 2);

    board.hold_tiles(4, 3, Row::Floor, 0).unwrap();
    assert_eq!(*board.penalties(), 5);
}

#[test]
fn place_holds_scores_an_isolated_tile_and_clears_the_pattern_line() {
    let mut board = Board::default();
    board.hold_tiles(0, 1, Row::Wall(0), 0).unwrap();

    board.place_holds();

    assert_eq!(board.placed()[0][0], Some(0));
    assert!(board.holds()[0].iter().all(Option::is_none));
    assert_eq!(board.get_score(), 1);
}

#[test]
fn place_holds_scores_horizontal_vertical_and_combined_lines() {
    let mut horizontal_placed = empty_grid();
    horizontal_placed[0][1] = Some(1);
    let mut horizontal_holds = empty_grid();
    horizontal_holds[0][0] = Some(0);
    let mut horizontal = Board::builder()
        .placed(horizontal_placed)
        .holds(horizontal_holds)
        .build();
    horizontal.place_holds();
    assert_eq!(horizontal.get_score(), 2);

    let mut vertical_placed = empty_grid();
    vertical_placed[1][0] = Some(4);
    let mut vertical_holds = empty_grid();
    vertical_holds[0][0] = Some(0);
    let mut vertical = Board::builder()
        .placed(vertical_placed)
        .holds(vertical_holds)
        .build();
    vertical.place_holds();
    assert_eq!(vertical.get_score(), 2);

    let mut both_placed = empty_grid();
    both_placed[0][1] = Some(1);
    both_placed[1][0] = Some(4);
    let mut both_holds = empty_grid();
    both_holds[0][0] = Some(0);
    let mut both = Board::builder()
        .placed(both_placed)
        .holds(both_holds)
        .build();
    both.place_holds();
    assert_eq!(both.get_score(), 4);
}

#[test]
fn place_holds_awards_row_column_and_tile_type_bonuses_once() {
    let mut row_placed = full_row(0);
    row_placed[0][4] = None;
    let mut row_holds = empty_grid();
    row_holds[0][0] = Some(4);
    let mut row_board = Board::builder().placed(row_placed).holds(row_holds).build();
    row_board.place_holds();
    assert_eq!(row_board.get_score(), 7);
    assert!(row_board.bonuses().rows[0]);
    row_board.place_holds();
    assert_eq!(row_board.get_score(), 7);

    let mut column_placed = full_column(0);
    column_placed[0][0] = None;
    let mut column_holds = empty_grid();
    column_holds[0][0] = Some(0);
    let mut column_board = Board::builder()
        .placed(column_placed)
        .holds(column_holds)
        .build();
    column_board.place_holds();
    assert_eq!(column_board.get_score(), 12);
    assert!(column_board.bonuses().columns[0]);

    let mut tile_type_placed = empty_grid();
    for row in 1..BOARD_DIMENSION {
        tile_type_placed[row][row] = Some(0);
    }
    let mut tile_type_holds = empty_grid();
    tile_type_holds[0][0] = Some(0);
    let mut tile_type_board = Board::builder()
        .placed(tile_type_placed)
        .holds(tile_type_holds)
        .build();
    tile_type_board.place_holds();
    assert_eq!(tile_type_board.get_score(), 11);
    assert!(tile_type_board.bonuses().tile_types[0]);
}

#[test]
fn place_holds_applies_penalties_with_saturating_score() {
    let mut board = Board::builder().score(6).build();
    board.hold_tiles(0, 3, Row::Floor, 0).unwrap();

    board.place_holds();

    assert_eq!(board.get_score(), 2);
    assert_eq!(*board.penalties(), 0);
}

#[test]
fn horizontal_line_count_matches_complete_wall_rows() {
    let mut placed = full_row(0);
    placed[2] = [Some(3), Some(4), Some(0), Some(1), Some(2)];
    let board = board_with_placed(placed);

    assert_eq!(board.count_horizontal_lines(), 2);
}
