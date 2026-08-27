# TODO

This roadmap reflects the current workspace on 2026-08-27. `movegen` is the
working rules library; `interface` has basic AzulFEN, move parsing, process
management, and a two-to-four-player UAI game loop; `random_engine` is a legal
random-move UAI baseline. Keep the dependency direction as:

```text
movegen  <-  interface and engines  <-  tournaments, self-play, and training
```

## Suggested next items

Do these in order. Each item should leave behind a runnable check or test.

1. **Add an end-to-end smoke test.** Build the random engine, start two to four
	 copies through `interface`, play a seeded game, and assert a terminal result.
2. **Separate interface setup from match execution.** Wire `--dry-run`,
	 `--check-engines`, engine names, `--out`, and clean startup/handshake errors;
	 remove the debug `Cli` dump and `expect`-based process failures.
3. **Implement the first real time-control path.** Start with fixed
	 per-move time (`st`), enforce a deadline around `go`, then add increment
	 clocks. Define timeout and forfeiture behavior in the protocol docs.
4. **Connect diagnostics and recovery.** Surface stderr, debug/log output,
	 child exit status, and protocol errors; make `--recover` restart a crashed
	 engine only at a well-defined game boundary.

## Phase 1: Usable engine harness

- Complete UAI command handling and document the command/response grammar.
- Add fixture-engine integration tests for handshake, readiness, full games,
	illegal moves, malformed responses, timeout, crash, stderr, and shutdown.
- Make human and UAI modes explicit instead of silently falling back to a
	local human game for mixed configurations.
- Add process resource limits where the platform supports them, or reject
	unsupported limits clearly.
- Tighten protocol parsing to match the documented whitespace and error rules.

## Phase 2: Rules and state robustness

- Validate builder and AzulFEN invariants: tile counts, wall/pattern-line
	consistency, penalties, active-player metadata, and impossible states.
- Add property/regression tests for legal moves, move application, round setup,
	scoring, game-over transitions, and tile conservation.
- Add exact seeded snapshot tests covering AzulFEN round trips, RNG state, and
	future draw order.
- Add deterministic random-play smoke games and benchmarks for legal moves,
	stepping, cloning, and serialization.

## Phase 3: Matches and tournaments

- Extract a reusable match runner from `interface/src/main.rs`.
- Implement seeded game scheduling for round-robin first, then gauntlet,
	random, and Swiss styles.
- Persist versioned game results with participants, seed, time control, moves,
	scores, winner, and failure reason.
- Add resume, `--max-games`, repeated matches, opening positions, side
	swapping, concurrency, and summaries incrementally.
- Track win rate, score difference, game length, illegal-action rate, timeout
	rate, and throughput.

## Phase 4: Reinforcement-learning foundation

- Define a stable environment API with `reset`, `step`, terminal/truncated
	status, rewards, metadata, deterministic seeds, and replayable episodes.
- Define one canonical action encoding for every legal move and expose a legal
	action mask; reject invalid actions rather than silently converting them.
- Define player perspective, round/game boundaries, first-player-token
	ownership, and observation encodings for player and centralized-critic use.
- Finish random and deterministic heuristic baselines before training a model.
- Add a short deterministic environment smoke run and golden observation/mask
	tests.

## Phase 5: Training and evaluation

- Choose the initial algorithm and model backend only after the environment
	contract is tested; keep CPU inference available for tests.
- Implement batched/vectorized environments, rollout storage, masked action
	selection, returns/advantages, and the selected optimizer/training loop.
- Version checkpoints with model, optimizer, scheduler, configuration, counters,
	dependency versions, repository revision, and RNG state.
- Build reproducible self-play and evaluation against random, heuristic,
	previous-checkpoint, and external UAI opponents.
- Add promotion gates, Elo or equivalent ratings, trajectory replay tools, and
	metrics for win rate, score, game length, illegal actions, and throughput.

## Maintenance and release gates

- Add CI for `cargo fmt --check`, `cargo clippy --workspace --all-targets`,
	`cargo test --workspace`, documentation builds, and deterministic smoke runs.
- Keep AzulFEN, UAI, environment, trajectory, and checkpoint formats versioned
	and documented independently.
- Record experiment provenance, generated-game provenance, and applicable model
	or data licenses.
- Profile allocations and memory before optimizing batching, self-play, or
	replay storage.
