# Movegen

The `movegen` crate owns Azul's validated game state, board rules, bowls, tile
bag, move representation, and legal move generation. It supports two through
four players.

`GameState::new(players, seed)` initializes a deterministic game using
xoshiro256++. The state exposes its serialized current RNG state through
`GameState::rng_state()`; `GameStateBuilder::set_rng_state` restores that state
for exact continuation. `GameState::get_tile_count()` tracks the complete
100-tile conservation invariant, including the discard pile. The `Bag::new` and `Bag::restock` constructors shuffle
their inputs, while `Bag::from_items` preserves an already serialized draw
order.

The integration tests in `tests/` cover public construction, transitions,
move legality, scoring, bag behavior, and seeded reproducibility.
