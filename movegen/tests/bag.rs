use azul_movegen::Bag;
use azul_movegen::Xoshiro256PlusPlus;

/// Creates a reproducible xoshiro RNG for bag tests.
fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::from_seed_u64(seed)
}

#[test]
fn new_is_seeded_and_iterator_draws_from_the_end() {
    let mut first_rng = rng(7);
    let mut second_rng = rng(7);
    let items = (0..10).collect::<Vec<_>>();
    let mut first = Bag::new(items.clone(), &mut first_rng);
    let second = Bag::new(items, &mut second_rng);

    assert_eq!(first.items(), second.items());

    let mut expected_draws = second.items().clone();
    expected_draws.reverse();
    assert_eq!(first.by_ref().collect::<Vec<_>>(), expected_draws);
    assert!(first.next().is_none());
}

#[test]
fn restock_replaces_contents_and_uses_the_supplied_rng() {
    let mut first_rng = rng(11);
    let mut second_rng = rng(11);
    let mut first = Bag::<usize>::default();
    let mut second = Bag::<usize>::default();

    first.restock(vec![4, 5, 6, 7], &mut first_rng);
    second.restock(vec![4, 5, 6, 7], &mut second_rng);

    assert_eq!(first.items(), second.items());
    assert_eq!(first.items().len(), 4);
    assert!(first.items().iter().all(|item| (4..=7).contains(item)));
}

#[test]
fn default_bag_is_empty() {
    let mut bag = Bag::<usize>::default();
    assert!(bag.items().is_empty());
    assert!(bag.next().is_none());
}

#[test]
fn from_items_preserves_draw_order() {
    let mut bag = Bag::from_items(vec![1, 2, 3]);

    assert_eq!(bag.next(), Some(3));
    assert_eq!(bag.next(), Some(2));
    assert_eq!(bag.next(), Some(1));
}
