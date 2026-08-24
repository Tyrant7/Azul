# Movegen

`movegen` is the rules engine and mutable state model for Azul. 

## Responsibilities

The crate exposes the following main types from
[`src/lib.rs`](src/lib.rs):

- `GameState`: complete mutable game state and turn transitions.
- `Board`: one player's pattern lines, wall, score, bonuses, and penalties.
- `Bowl`: a factory display or the centre area.
- `Bag<T>`: a shuffled draw container.
- `Move`: a tile-selection action and its destination row.
- `Row`: either a numbered pattern line or the floor.
- `Tile`: the tile-type alias used throughout the rules engine.

The implementation is split by responsibility:

| Module | Responsibility |
| --- | --- |
| [`board.rs`](src/board.rs) | Pattern lines, wall placement, scoring, bonuses, and penalties. |
| [`board/builder.rs`](src/board/builder.rs) | Explicit `Board` construction for tests and state loading. |
| [`bag.rs`](src/bag.rs) | Generic shuffled bags and draw-order preservation. |
| [`bowl.rs`](src/bowl.rs) | Sorted tile collections and tile-type extraction. |
| [`gamestate.rs`](src/gamestate.rs) | Player state, round setup, legal moves, transitions, and game end. |
| [`gamestate/builder.rs`](src/gamestate/builder.rs) | Explicit `GameState` construction and validation. |
| [`game_move.rs`](src/game_move.rs) | Move representation and illegal-move errors. |
| [`row.rs`](src/row.rs) | Pattern-line and floor destinations. |

## Game lifecycle

Only two-, three-, and four-player games are supported. Construction returns a
`Result` so unsupported player counts and structurally invalid builder states
can be rejected safely.

```rust
use azul_movegen::GameState;

let mut game = GameState::new(2, 42).expect("valid player count");
game.setup_next_round();

while !game.is_game_over() {
    if game.round_over() {
        game.setup_next_round();
        continue;
    }

    let choice = game
        .get_valid_moves()
        .into_iter()
        .next()
        .expect("a non-empty round has a legal move");
    game.make_move(&choice).expect("move came from the legal set");
}

let winner = game.get_winner();
println!("winner: player {winner}");
```

The intended transition sequence is:

1. Create a `GameState` with a player count and seed.
2. Call `setup_next_round()` to resolve completed pattern lines, apply
   penalties, restock when required, and fill factory bowls.
3. Read `get_valid_moves()` for the active player.
4. Apply one choice with `make_move()`.
5. Repeat moves until `round_over()` is true, then start the next round.
6. Stop after `is_game_over()` becomes true and inspect `get_winner()`.

There are `2 * players + 2` bowls: the centre at index `0`, followed by the
factory bowls. Factory bowls receive four tiles during round setup. The centre
has no factory capacity limit.

## State and invariants

The rules engine maintains these important relationships:

- A pattern line with index `n` has capacity `n + 1` and contains at most one
  tile type.
- A wall position accepts only its prescribed tile type, and a tile type can
  appear at most once in a wall row.
- `Bowl` operations preserve sorted tile order. `take_tiles` returns the
  selected tiles and the remaining tiles without changing their multiplicity.
- `Bag::new` and `Bag::restock` shuffle their inputs. `Bag::from_items`
  preserves a serialized draw order exactly.
- A complete game accounts for 100 physical tiles. `GameState::get_tile_count`
  counts tiles in the bag, bowls, boards, and discard pile.
- Penalty spaces and physical penalty tiles are tracked separately because the
  first-player token occupies a penalty space but is not itself a tile.

The explicit builders are useful for tests, parsers, and controlled fixtures.
They validate the relationships owned by `GameState`, such as supported player
counts, bowl counts, active-player indexes, and RNG-state encoding. Callers
constructing arbitrary `Board` values remain responsible for supplying a
rule-valid wall and pattern-line layout.

## Determinism and snapshots

`GameState::new(players, seed)` initializes the xoshiro256++ random stream and
uses it to shuffle the initial bag. The numeric seed identifies the initial
game or episode. It is different from the current generator state after draws
and shuffles.

For exact continuation, use `GameState::rng_state()` and restore it with
`GameStateBuilder::set_rng_state`. The interface persists this state in
AzulFEN; see [`../interface/azulfen.md`](../interface/azulfen.md) for the
interchange format. A loaded snapshot also preserves the bag draw order and
discard accounting.

The RNG is intended for deterministic game simulation, replay, and training
environments. It is not a source of cryptographic randomness.

## Testing

Run the full workspace suite from the repository root:

```text
cargo test --workspace
```

The integration tests use only public APIs and are organized by behavior:

- [`tests/bag.rs`](tests/bag.rs): seeded shuffling, draw order, restocking,
  and replacement behavior.
- [`tests/bowl.rs`](tests/bowl.rs): sorting, extraction, tile types,
  extension, and cloning.
- [`tests/board.rs`](tests/board.rs): holds, wall placement, scoring,
  bonuses, penalties, and wall helpers.
- [`tests/gamestate.rs`](tests/gamestate.rs): construction validation,
  player counts, rounds, legal moves, RNG determinism, tile conservation,
  game end, and winner selection.
- [`tests/types.rs`](tests/types.rs): `Row` and `Move` value semantics.

When adding a rule, add a focused regression test for its boundary behavior
and update the public documentation if the state or transition contract
changes.
