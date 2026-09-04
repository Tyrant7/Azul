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

### Immediate goals

- Add batched and vectorized environments for parallel rollouts
- Minibatching
- Add greedy policy evaluations against strongest previous checkpoints and track winrate
- Look into 1. Nextlat 2. HL Gauss
- Deterministic reproducibility
- Add optimizer state, scheduler state, counters, configuration, and RNG state to save/load checkpoint and make training pause/resume possible
- Add promotion gates so new checkpoints must beat a reference agent before entering the opponent pool
- Document everything -> esepcially how to play against the various engine checkpoints

### Longterm goals

- Explore MCTS or AlphaZero-style search on top of the learned policy/value model.
- Add Elo or another rating system for checkpoint and opponent-pool comparisons
- Add deterministic evaluation suites separate from stochastic training
- Document how to reproduce a published run from a clean checkout
