# TODO

## Interface
- Use a custom markdown file for UAI
  - Once complete, change the links in the docstrings and review that
- Command line argument documentation
- Full docstrings
- Tests
- Complete child-process stdin/stdout/stderr wiring and lifecycle management
- Implement UAI command dispatch, handshakes, position updates, move requests, quit, and errors
- Implement time controls, deadlines, engine recovery, logging, and resource limits
- Implement tournament scheduling, concurrency, resumable results, openings, and summaries
- Make AzulFEN parsing strict, versioned, and round-trip tested

## Reinforcement learning system

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

### Baselines and agents
- Finish `random_engine` as a legal random-move baseline
- Add a deterministic heuristic baseline for measuring learning progress
- Define a policy/value agent interface independent of a particular neural-network framework
- Implement action selection with legal-action masking, temperature, exploration, and evaluation modes
- Choose and implement the initial learning algorithm (for example PPO, or policy/value self-play with MCTS)
- Support self-play with alternating player perspectives and correct credit assignment across turns and rounds
- Add optional opponent pools, fixed checkpoints, and exploitability-style evaluation

### Model and training runtime
- Choose the model representation and backend, with CPU inference available for tests and a GPU path where useful
- Implement observation encoding, policy logits, value prediction, and batched inference
- Implement loss functions, optimizer, gradient clipping, learning-rate schedules, and entropy/value-loss weighting
- Implement rollout workers and an actor/learner data path
- Support configurable parallel environments, inference batches, rollout length, and update frequency
- Add checkpoint save/load for model weights, optimizer state, scheduler state, counters, configuration, and RNG state
- Add checkpoint compatibility/versioning and a way to resume interrupted training
- Add replay or trajectory storage with episode IDs, observations, actions, masks, rewards, values, log-probabilities, and terminal flags
- Add generalized advantage estimation or the equivalent return/target calculation for the selected algorithm
- Add replay-buffer capacity, sampling, prioritization, persistence, and cleanup if the selected algorithm needs replay

### Self-play and evaluation
- Build a self-play runner that can generate reproducible games from a checkpoint
- Store trajectories and completed game results in a documented, versioned format
- Add head-to-head evaluation against random, heuristic, previous-checkpoint, and external UAI engines
- Track win rate, score, score difference, game length, illegal-action rate, and throughput
- Add Elo or another rating system for checkpoint and opponent-pool comparisons
- Add promotion gates so new checkpoints must beat a reference agent before entering the opponent pool
- Add deterministic evaluation suites separate from stochastic training
- Add tools to replay a trajectory and inspect observations, masks, rewards, and chosen actions

### Experiment management
- Define a single versioned training configuration covering seeds, environment, model, algorithm, rollout, evaluation, and output paths
- Record the repository revision, configuration, dependency versions, platform, and random seeds with every run
- Add structured metrics, episode logs, checkpoint metadata, and optional experiment tracking
- Add CLI commands for training, self-play generation, evaluation, checkpoint inspection, and replay
- Add graceful shutdown and periodic checkpointing for long-running jobs
- Add resource controls for worker count, memory, CPU threads, GPU selection, and storage limits
- Document how to reproduce a published run from a clean checkout

### Serving and engine integration
- Load a trained checkpoint into an engine process without importing training-only dependencies
- Expose the trained policy through the UAI protocol and the existing interface tournament runner
- Support inference-time limits, deterministic strength testing, temperature, and optional search
- Add model validation when loading checkpoints and clear protocol errors for unsupported positions
- Benchmark end-to-end move latency and throughput under realistic tournament concurrency

### Reliability and maintenance
- Add unit, integration, property, serialization, and end-to-end process tests for the full RL path
- Add CI checks for formatting, linting, tests, reproducibility smoke runs, and documentation links
- Profile memory and allocations in move generation, self-play, batching, and replay storage
- Document model/data licenses and the provenance of generated games
- Keep the rules engine, environment API, protocol, trajectory format, and model checkpoints versioned independently where practical
