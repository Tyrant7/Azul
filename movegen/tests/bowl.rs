use azul_movegen::Bowl;

#[test]
fn fill_sorts_tiles_and_unique_types_are_sorted() {
    let bowl = Bowl::from_tiles(vec![3, 1, 3, 0, 2]);

    assert_eq!(bowl.get_tiles(), &vec![0, 1, 2, 3, 3]);
    assert_eq!(bowl.get_tile_types(), vec![0, 1, 2, 3]);
}

#[test]
fn extend_adds_and_resorts_tiles() {
    let mut bowl = Bowl::from_tiles(vec![1, 4]);
    bowl.extend(&vec![3, 0, 3]);

    assert_eq!(bowl.get_tiles(), &vec![0, 1, 3, 3, 4]);
}

#[test]
fn take_tiles_returns_selected_and_remaining_then_clears_bowl() {
    let mut bowl = Bowl::from_tiles(vec![0, 2, 2, 4]);

    let (selected, remaining) = bowl.take_tiles(2);

    assert_eq!(selected, vec![2, 2]);
    assert_eq!(remaining, vec![0, 4]);
    assert!(bowl.get_tiles().is_empty());
}

#[test]
fn take_tiles_without_a_match_returns_all_tiles_as_remaining() {
    let mut bowl = Bowl::from_tiles(vec![0, 2]);

    let (selected, remaining) = bowl.take_tiles(4);

    assert!(selected.is_empty());
    assert_eq!(remaining, vec![0, 2]);
    assert!(bowl.get_tiles().is_empty());
}

#[test]
fn clone_is_independent() {
    let original = Bowl::from_tiles(vec![1, 2]);
    let mut clone = original.clone();

    clone.fill(vec![3]);

    assert_eq!(original.get_tiles(), &vec![1, 2]);
    assert_eq!(clone.get_tiles(), &vec![3]);
}
