# Azul

This repository is an experimental Rust implementation of the board game Azul, together with the beginnings of an engine interface. The workspace is organized so that the rules engine can be reused by command-line tools, UAI-compatible engines, self-play workers, and—eventually—a reinforcement-learning trainer.

## Workspace layout

```text
azul/
├── movegen/         Core Azul rules and game-state representation
├── interface/       CLI, UAI direction, move parsing, and AzulFEN I/O
├── random_engine/   Placeholder engine executable
├── Cargo.toml       Workspace definition
└── TODO.md          Development and reinforcement-learning roadmap
```

### `movegen`

[`movegen/`](movegen/) is the rules library. It owns:

- `Board`: held rows, placed tiles, penalties, scores, and bonuses.
- `Bowl`: a factory or centre bowl containing tile types.
- `Bag`: a shuffled tile container.
- `Move`: a bowl, tile type, and destination row.
- `GameState`: players, bowls, bag, active player, round transitions, legal moves, and terminal scoring.

The crate exposes the main types from [`movegen/src/lib.rs`](movegen/src/lib.rs), while the implementation is split across the modules under [`movegen/src/`](movegen/src/). Its short crate-specific notes are in [`movegen/README.md`](movegen/README.md).

The normal game flow is:

1. Create a `GameState` for two, three, or four players and handle the construction result.
2. Call `setup_next_round()` to place completed holds, apply penalties, and fill the factory bowls.
3. Read `get_valid_moves()` for the active player.
4. Apply one move with `make_move()`.
5. When `round_over()` is true, call `setup_next_round()` again.
6. Stop when `is_game_over()` is true and use `get_winner()`.

### `interface`

[`interface/`](interface/) is the integration layer. It currently provides:

- Clap-based engine and match configuration.
- Human-readable board and game-state formatting.
- Six-digit move parsing (`bowl`, `tile type`, `row`).
- AzulFEN serialization and deserialization.
- The initial shape of the Universal Azul Interface (UAI) protocol.

See the [`interface/README.md`](interface/README.md) for command-line configuration, [`interface/protocol.md`](interface/protocol.md) for the UAI direction, and [`interface/azulfen.md`](interface/azulfen.md) for the state interchange format.

The interface executable parses CLI configuration, can spawn configured child processes with managed stdin/stdout/stderr, performs the UAI startup sequence, enforces time controls, recovers eligible engine failures when requested, and runs a two-to-four-player UAI game when all configured engines use UAI. It also retains a local human-input mode. Tournaments, logging/resource limits, and advanced UAI options remain to be implemented.

### `random_engine`

[`random_engine/`](random_engine/) is reserved for a simple engine that can be used as a baseline opponent and as a smoke-test process for the interface. Its executable is currently only a placeholder.

## State and protocol boundaries

The rules engine should remain independent of process management, command-line parsing, and machine-learning code. The intended dependency direction is:

```text
rules / movegen  ←  interface and engines  ←  tournaments, self-play, and training tools
```

AzulFEN v1 is the current persistence and interchange format. The interface formats a `GameState` as board sections, bowl sections, a bag section, and complete active-player/token metadata. Versioned snapshots include the optional initial seed, current xoshiro256++ state, penalty-tile tracking, and discard count, so loading them reproduces future shuffles and tile accounting exactly. Parsing is strict and accepts only canonical `azulfen:v1` snapshots; see [`interface/azulfen.md`](interface/azulfen.md) for the grammar. UAI defines the external command and move protocol around that state. Both formats are still evolving; protocol changes should be documented in their respective files and covered by round-trip tests.

## Building and testing

From the repository root:

```bash
cargo check --workspace
cargo test --workspace
```

Run the interface help or executable with:

```bash
cargo run -p interface -- --help
cargo run -p interface -- --engine "path=PATH proto=human tc=60" "path=PATH proto=human tc=60"
```

The current CLI requires at least two `--engine` configurations. Consult [`interface/README.md`](interface/README.md) for the available configuration fields. Run the random engine with:

```bash
cargo run -p random_engine
```

## Development direction

The immediate engineering priorities are to make the rules deterministic and thoroughly tested, complete the UAI process protocol, and expose a stable environment API. Those foundations will support parallel self-play and a reproducible reinforcement-learning system. The detailed feature roadmap is maintained in [`TODO.md`](TODO.md).
