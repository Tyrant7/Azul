use azul_movegen::{BowlChoice, Move, Row};

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
            bowl: BowlChoice::Centre,
            tile_type: 0,
            row: Row::Floor,
        }
    );
}

#[test]
fn move_display_uses_the_six_digit_protocol_format() {
    assert_eq!(
        Move {
            bowl: BowlChoice::Factory(3),
            tile_type: 1,
            row: Row::Wall(1),
        }
        .to_string(),
        "040102"
    );
    assert_eq!(
        Move {
            bowl: BowlChoice::Centre,
            tile_type: 4,
            row: Row::Floor,
        }
        .to_string(),
        "000400"
    );
}
