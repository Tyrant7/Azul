# RL Beam Engine

`rl_beam` exposes the trained actor through the interface crate's UAI runtime
and selects moves with a policy-prior beam search. It keeps the highest-scoring
partial action sequences according to the actor's masked log-probabilities and
returns the first move from the best line. The current implementation is a
basic policy beam: it does not load the critic or perform minimax search.

## Play against the beam engine

Build the engine with the project PyTorch environment active:

```bash
source scripts/activate-env.sh
cargo build -p interface -p rl_beam
cargo run -p interface -- \
  --engine "path=./target/debug/rl_beam args=checkpoints/azul_actor.ot proto=uai tc=1+0" \
           "proto=human" \
  --out ./runs/beam-game.azl \
  --seed 42
```

The engine accepts optional positional arguments after the checkpoint path:

```text
rl_beam ACTOR_CHECKPOINT [BEAM_WIDTH] [DEPTH]
```

The defaults are beam width `4` and depth `2` plies.

## Compare greedy and beam policies

The matchup binary uses the same actor checkpoint for both policies, seeds
each game deterministically, and alternates which player receives beam search:

```bash
source scripts/activate-env.sh
cargo run -p rl_beam --bin matchup -- \
  checkpoints/azul_actor.ot 1000 4 2
```

The arguments are `ACTOR_CHECKPOINT`, `GAMES`, `BEAM_WIDTH`, and `DEPTH`.
The output reports beam wins, greedy baseline wins, and the beam win rate.
