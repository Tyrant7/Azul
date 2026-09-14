# Azul

This repository is an experimental Rust implementation of the board game Azul, together with an engine interface and a reinforcement-learning environment. The workspace is organized so that the rules engine can be reused by command-line tools, UAI-compatible engines, self-play workers, and future training systems.

## Workspace layout

```text
azul/
├── movegen/         Core Azul rules and game-state representation
├── interface/       CLI, UAI direction, move parsing, and AzulFEN I/O
├── random_engine/   Random legal-move UAI engine
├── rl_env/          Two-player reinforcement-learning environment
├── rl_beam/         Policy-prior beam-search UAI engine and matchup tool
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
terminal/truncation status, rewards, legal-action masks, and a minimal
on-policy PPO trainer. `PpoTrainer` collects complete episodes into a
temporary rollout batch, computes GAE advantages and value targets, and performs
full-batch clipped PPO updates before discarding that batch; it does not use
an off-policy replay buffer. Observations are fixed-size and player-relative:
the active player's board is first, the centre is encoded separately from
factory bowls, and the two-player wire action space contains six source slots
(centre plus five factories), five tile types, and six destinations. The
policy scores state/action pairs and normalizes a categorical distribution over
the currently legal candidates; the fixed 180-action IDs remain the boundary
used by `step` and action masks.
The critic uses an HL-Gauss categorical value head: it predicts probabilities
over a fixed value support, trains against Gaussian-smoothed return targets,
and decodes the expected support value for GAE and PPO. The support and target
width are configurable through `critic_value_min`, `critic_value_max`,
`critic_value_bins`, and `critic_sigma_ratio` in `PpoConfig`.
The crate uses `tch`, so building it requires a compatible LibTorch
installation; the rules and interface crates can be tested independently.

The current trainer is intentionally a learning baseline rather than a full
training system. It has no minibatches, entropy bonus, or parallel rollout
workers. It uses generalized advantage estimation, supports optional
deterministic greedy evaluation against a frozen reference actor, and writes
scalar training diagnostics to `runs/azul_ppo` using TensorBoard event files.
Set `PpoConfig::evaluation_games` above zero and configure a reference actor to
enable evaluation; `evaluation_interval` controls how many PPO iterations occur
between evaluations. See
[`rl_env/src/ppo.rs`](rl_env/src/ppo.rs) for the algorithm and [`TODO.md`](TODO.md)
for the remaining training-system work.

### `rl_beam`

[`rl_beam/`](rl_beam/) exposes the fixed reference actor and critic through UAI
with a critic-guided alternating beam search. It also provides a direct
matchup binary for comparing greedy and beam policies over reproducibly seeded
games. See [`rl_beam/README.md`](rl_beam/README.md).

A minimal training session can be started from Rust with:

```rust
let config = rl_env::PpoConfig::default();
let mut trainer = rl_env::PpoTrainer::new(config)?;
let mut environment = rl_env::AzulEnv::new(0, None);
trainer.train(&mut environment, 10_000);
```

The workspace executable runs the same training loop and writes TensorBoard
events:

```bash
source scripts/activate-env.sh
python -m pip install tensorboard  # once, if TensorBoard is not installed
cargo run -p rl_env
tensorboard --logdir runs
```

## Playing and training

The `interface` executable is the easiest way to play a game. It accepts at
least two engine descriptors; human players use `proto=human`, while UAI
engines provide an executable path and a time control.

### Human play

Start a human-vs-human game with a reproducible opening:

```bash
cargo run -p interface -- \
  --engine "proto=human" "proto=human" \
  --out ./runs/manual-game.azl \
  --seed 42
```

For human moves, enter six digits in the form `BBTTDD`:

| Field | Meaning |
| --- | --- |
| `BB` | Wire bowl: `00` is the centre, `01` is factory 0, and so on. |
| `TT` | Tile type, from `00` through `04`. |
| `DD` | Destination: `00` is the floor, while `01` through `05` are wall rows 1 through 5. |

For example, `040102` takes tile type 1 from factory 3 and places it on wall
row 2. The interface prints the board after each accepted move and rejects
malformed or illegal moves.

### Play against the random engine

Build the interface and baseline engine, then start a human-vs-random game:

```bash
cargo build -p interface -p random_engine
cargo run -p interface -- \
  --engine "path=./target/debug/random_engine proto=uai tc=60+2" \
           "proto=human" \
  --out ./runs/random-game.azl \
  --seed 42
```

The order of the descriptors determines player numbers. In this example the
random engine is player 0 and the human is player 1. Swap the descriptors to
play first.

### Play against a trained actor

`rl_engine` loads an actor checkpoint and exposes it as a UAI engine. The actor
checkpoint is sufficient for play; the critic checkpoint is used during PPO
training and is not required by `rl_engine`.

```bash
source scripts/activate-env.sh
cargo build -p interface -p rl_engine
cargo run -p interface -- \
  --engine "path=./target/debug/rl_engine args=checkpoints/azul_actor.ot proto=uai tc=1+0" \
           "proto=human" \
  --out ./runs/rl-game.azl \
  --seed 42
```

The `checkpoints/azul_actor.ot` file is the actor produced by the included
training executable when training completes. To use another checkpoint, change
the path after `args=`. The checkpoint path must not contain spaces because
engine descriptors are currently whitespace-separated.

You can also run two engines against each other, for example:

```bash
cargo run -p interface -- \
  --engine "path=./target/debug/rl_engine args=checkpoints/azul_actor.ot proto=uai tc=1+0" \
           "path=./target/debug/random_engine proto=uai tc=1+0" \
  --out ./runs/engine-game.azl \
  --seed 42
```

To run the beam engine instead, use `rl_beam` and optionally pass the beam
width and depth after the checkpoint path:

```bash
cargo run -p interface -- \
  --engine "path=./target/debug/rl_beam args=checkpoints/reference_actor.ot proto=uai tc=1+0" \
           "proto=human" \
  --out ./runs/beam-game.azl \
  --seed 42
```

For a direct 1,000-game greedy-versus-beam comparison using the same actor
checkpoint:

```bash
cargo run -p rl_beam --bin matchup -- \
  checkpoints/reference_actor.ot 1000 4 2
```

Use `cargo run -p interface -- --help` for time controls, diagnostics, engine
recovery, and resource-limit options.

### Train a new actor

The workspace training executable currently runs the PPO baseline configured
in [`rl_env/src/main.rs`](rl_env/src/main.rs). Activate the project environment
first so `tch` can find the local PyTorch/LibTorch installation:

```bash
source scripts/activate-env.sh
cargo run -p rl_env
```

The run collects complete episodes, performs PPO updates, periodically evaluates
the greedy actor against a frozen reference actor, and writes the final actor
and critic to:

```text
checkpoints/azul_actor.ot
checkpoints/azul_critic.ot
```

Before starting training, place the completed actor checkpoint used as the
stable baseline at:

```text
checkpoints/reference_actor.ot
```

The training executable loads this file but does not create or overwrite it.
Keep it unchanged if you want evaluation results to remain comparable across
runs.

Training diagnostics are written under `runs/azul_ppo`. Install TensorBoard
once if needed and view them with:

```bash
python -m pip install tensorboard
tensorboard --logdir runs
```

The executable evaluates 16 games against the fixed reference every 10 PPO
iterations, then runs a final 500-game evaluation after training completes.
The final evaluation is printed as `final_evaluation` and is the preferred
win-rate comparison between training runs; it is printed to the terminal and
is not currently added to TensorBoard. Periodic evaluation results are written
to TensorBoard under `evaluation/greedy_win_rate` and `evaluation/games`.
Evaluation is intentionally less frequent than training because it can be
expensive. To change the training length, periodic evaluation cadence, or
final evaluation size, edit the values in `rl_env/src/main.rs`. For custom
applications, call
`PpoTrainer::set_reference_actor` before `PpoTrainer::train_with_callback`;
set `evaluation_games` above zero to enable periodic evaluation.

The current executable uses 1,000 rollout transitions, four full-batch PPO
passes, and `gamma = 0.99`. The current HL-Gauss defaults are a value support
of `[-6.5, 6.5]`, 128 bins, and `critic_sigma_ratio = 1.0`; the actor learning
rate defaults to `3e-4` and the critic learning rate to `2e-4`. These
experimental values are defined in [`rl_env/src/main.rs`](rl_env/src/main.rs) and
[`rl_env/src/ppo.rs`](rl_env/src/ppo.rs).

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
the `tch` dependency. When using the project virtual environment's PyTorch
installation, activate it first so both compilation and runtime library
loading are configured:

```bash
source scripts/activate-env.sh
cargo test --workspace
```

Without LibTorch, run the rules and interface checks separately with
`cargo test -p movegen` and `cargo test -p interface`.

Run the interface help or executable with:

```bash
cargo run -p interface -- --help
cargo run -p interface -- --engine "proto=human" "proto=human" --out ./runs/manual-game.azl
```

The current CLI requires at least two `--engine` configurations. Consult [`interface/README.md`](interface/README.md) for the available configuration fields. Run the random engine with:

```bash
cargo run -p random_engine
```

## Development direction

The rules and environment now expose the foundations for a reproducible
reinforcement-learning system, including player-relative observations,
legal-action masking, and a minimal PPO baseline. The next priorities are
deterministic evaluation, stronger environment tests, checkpointing, and
scalable rollout infrastructure. The detailed feature roadmap is maintained
in [`TODO.md`](TODO.md).
