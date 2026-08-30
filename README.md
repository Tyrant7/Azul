# Azul

This repository is an experimental Rust implementation of the board game Azul, together with an engine interface and a reinforcement-learning environment. The workspace is organized so that the rules engine can be reused by command-line tools, UAI-compatible engines, self-play workers, and future training systems.

## Workspace layout

```text
azul/
├── movegen/         Core Azul rules and game-state representation
├── interface/       CLI, UAI direction, move parsing, and AzulFEN I/O
├── random_engine/   Random legal-move UAI engine
├── rl_env/          Two-player reinforcement-learning environment
├── Cargo.toml       Workspace definition
└── TODO.md          Development and reinforcement-learning roadmap
```

### `movegen`

[`movegen/`](movegen/) is the rules library. It owns:

- `Board`: held rows, placed tiles, penalties, scores, and bonuses.
- `Bowl`: a sorted tile collection used by a factory or the centre.
- `Bag`: a shuffled tile container.
- `Move`: a typed centre/factory source, tile type, and destination row.
- `GameState`: player boards, one centre bowl, factory bowls, the bag, active player, round transitions, legal moves, and terminal scoring.

The crate exposes the main types from [`movegen/src/lib.rs`](movegen/src/lib.rs), while the implementation is split across the modules under [`movegen/src/`](movegen/src/). Its short crate-specific notes are in [`movegen/README.md`](movegen/README.md).

The normal game flow is:

1. Create a `GameState` for two, three, or four players and handle the construction result.
2. Call `setup_next_round()` to place completed holds, apply penalties, and fill the factory bowls. The centre is collected during play and is not restocked.
3. Read `get_valid_moves()` for the active player.
4. Apply one move with `make_move()`.
5. When `round_over()` is true, call `setup_next_round()` again.
6. Stop when `is_game_over()` is true and use `get_winner()`.

### `interface`

[`interface/`](interface/) is the integration layer. It currently provides:

- Clap-based engine and match configuration.
- Human-readable board and game-state formatting.
- Six-digit move parsing (`wire bowl`, `tile type`, `row`), where wire bowl `00` is the centre and `01` is factory bowl `0`.
- AzulFEN serialization and deserialization.
- The initial shape of the Universal Azul Interface (UAI) protocol.

See the [`interface/README.md`](interface/README.md) for command-line configuration, [`interface/protocol.md`](interface/protocol.md) for the UAI direction, and [`interface/azulfen.md`](interface/azulfen.md) for the state interchange format.

The interface executable parses CLI configuration, can spawn configured child processes with managed stdin/stdout/stderr, performs the UAI startup sequence, enforces time controls, applies diagnostics and process resource limits, recovers eligible engine failures when requested, and runs a single two-to-four-player UAI game when all configured engines use UAI. It also retains a local human-input mode. Tournament scheduling, result persistence/resume, opening books, and advanced UAI options remain to be implemented.

### `rl_env`

[`rl_env/`](rl_env/) provides a two-player environment with `reset`, `step`,
terminal/truncation status, rewards, legal-action masks, and replay-buffer
scaffolding. Observations are fixed-size and player-relative: the active
player's board is first, the centre is encoded separately from factory bowls,
and the action space reserves ten wire bowl slots for the four-player maximum.
The crate uses `tch`, so building it requires a compatible LibTorch
installation; the rules and interface crates can be tested independently.

### `random_engine`

[`random_engine/`](random_engine/) is a simple UAI engine that selects uniformly from the legal moves in the supplied AzulFEN position. It is useful as a baseline opponent and as a smoke-test process for the interface. It supports `uai`, `isready`, `newgame`, `position fen`, `go`, `stop`, `setoption`, and `quit`.

## State and protocol boundaries

The rules engine should remain independent of process management, command-line parsing, and machine-learning code. The intended dependency direction is:

```text
rules / movegen  ←  interface and engines  ←  tournaments, self-play, and training tools
```

AzulFEN v1 is the current persistence and interchange format. The interface formats a `GameState` as board sections, a centre bowl followed by factory-bowl fields, a bag section, and complete active-player/token metadata. Versioned snapshots include the optional initial seed, current xoshiro256++ state, penalty-tile tracking, and discard count, so loading them reproduces future shuffles and tile accounting exactly. Parsing is strict and accepts only canonical `azulfen:v1` snapshots; see [`interface/azulfen.md`](interface/azulfen.md) for the grammar. UAI defines the external command and move protocol around that state. Both formats are still evolving; protocol changes should be documented in their respective files and covered by round-trip tests.

## Building and testing

From the repository root:

```bash
cargo check --workspace
cargo test --workspace
```

The workspace commands also build `rl_env` and therefore require LibTorch for
the `tch` dependency. Without LibTorch, run the rules and interface checks
separately with `cargo test -p movegen` and `cargo test -p interface`.

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
