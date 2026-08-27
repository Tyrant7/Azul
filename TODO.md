# TODO

## Interface
- Complete UAI command dispatch with diagnostics, recovery, and tournament integration
  - Define the per-engine command state machine: startup, ready, new game, position, search, move response, and shutdown.
  - Add the remaining `go` forms, including clock-aware commands if the UAI draft adopts them.
  - Preserve strict response handling: consume `info` updates, accept exactly one `bestmove`, and surface `error` responses with the active command.
  - Reject protocol responses that arrive out of phase, including duplicate terminal responses, unexpected EOF, and moves from the wrong player.
  - Add end-to-end tests using a deterministic fake UAI engine for normal play, malformed output, illegal moves, timeouts, and graceful `quit`.
- Implement structured logging and resource limits
  - Persist completed-game and forfeit records, including failure reason, player, restart attempts, and final clock state.
  - Extend recovery coverage across startup failures, crashes, EOF, broken pipes, and repeated restart exhaustion.
  - Add structured command/response logging with engine identity, player, game, turn, timestamps, and redaction rules for sensitive paths or arguments.
  - Wire `--debug`, `--log`, `--stderr`, and `--quiet` to the same diagnostics pipeline without contaminating protocol stdout.
  - Extend platform resource-limit tests to cover descendant processes, restart inheritance, and resource-limit forfeit reporting.

- Implement tournament scheduling, concurrency, resumable results, openings, and summaries
  - Convert engine configurations into deterministic pairings for gauntlet, round-robin, Swiss, and random styles.
  - Honor `--games`, `--rounds`, `--repeat`, `--max-games`, `--swap`, and the tournament seed in pairing and side assignment.
  - Run independent games concurrently while isolating processes, RNG seeds, logs, and result records per game.
  - Define a versioned results format containing participants, configuration, seed, outcome, scores, termination reason, and elapsed time.
  - Make `--out` atomic and make `--resume` validate configuration compatibility before continuing unfinished work.
  - Load and validate opening positions, then include the opening snapshot in reproducibility metadata.
  - Produce summaries for wins, draws, scores, failures, timeouts, game length, and throughput.

```text
movegen  <-  interface and engines  <-  tournaments, self-play, and training
```

### Environment contract
- Define a stable environment API with `reset`, `step`, terminal/truncated status, rewards, and episode metadata
- Define the observation space for one player and for a centralized critic
- Define a canonical action encoding for every legal move, including an explicit legal-action mask
- Define perspective handling so the active player, opponent boards, scores, and rewards are unambiguous
- Decide how invalid actions are handled: masked before inference, rejected by the environment, and never silently converted
- Expose round boundaries, game boundaries, first-player-token ownership, and player count to the environment
- Add batched and vectorized environments for parallel rollouts
- Add deterministic seeded resets and replayable episode seeds
### Rules and environment validation
- Build a comprehensive rules test suite, including property tests and regression tests for scoring and transitions
- Add golden tests for legal-action masks and observation encodings
- Test AzulFEN save/load as an exact environment snapshot, including RNG and turn metadata where required
- Test that random, scripted, and model policies cannot create illegal or impossible states
- Add short deterministic smoke episodes and a random-policy baseline
- Add performance benchmarks for reset, step, legal moves, cloning, serialization, and batched stepping

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
