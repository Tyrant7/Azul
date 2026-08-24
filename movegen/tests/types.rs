use azul_movegen::{Move, Row};

#[test]
fn row_variants_have_expected_value_semantics() {
    assert_eq!(Row::default(), Row::Floor);
    assert_eq!(Row::Wall(2), Row::Wall(2));
    assert_ne!(Row::Wall(2), Row::Wall(3));
}

#[test]
fn move_default_targets_the_floor_of_the_centre() {
    assert_eq!(
        Move::default(),
        Move {
            bowl: 0,
            tile_type: 0,
            row: Row::Floor,
        }
    );
}
