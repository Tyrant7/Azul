# RL Beam Engine

`rl_beam` exposes the fixed reference actor through the interface crate's UAI
runtime. The actor generates candidate moves, while the matching reference
critic evaluates search states from the root player's perspective. The critic
may be scalar MSE or categorical; the loader detects the checkpoint head shape.
Opponent layers retain lower-valued states and root-player layers retain
higher-valued states as a basic alternating beam search.

## Play against the beam engine

Build the engine with the project PyTorch environment active:

```bash
source scripts/activate-env.sh
cargo build -p interface -p rl_beam
cargo run -p interface -- \
  --engine "path=./target/debug/rl_beam args=checkpoints/reference_actor.ot proto=uai tc=1+0" \
           "proto=human" \
  --out ./runs/beam-game.azl \
  --seed 42
```

The engine accepts optional positional arguments after the checkpoint path:

```text
rl_beam ACTOR_CHECKPOINT [BEAM_WIDTH] [DEPTH] [CRITIC_CHECKPOINT]
```

The default critic checkpoint is `checkpoints/reference_critic.ot`. The default
beam width is `4` and depth is `2` plies. The actor and critic should be the
matching reference checkpoints for this experiment.

## Compare greedy and beam policies

The matchup binary uses the same actor checkpoint for both policies, seeds
each game deterministically, and alternates which player receives beam search:

```bash
source scripts/activate-env.sh
cargo run -p rl_beam --bin matchup -- \
  checkpoints/reference_actor.ot 1000 4 2
```

The arguments are `ACTOR_CHECKPOINT`, `GAMES`, `BEAM_WIDTH`, and `DEPTH`.
The output reports beam wins, greedy baseline wins, and the beam win rate.
