# RL Engine

`rl_engine` exposes a trained `rl_env` actor through the interface crate's UAI
runtime. It selects the highest-scoring legal move for each two-player position.

Build and run it with an actor checkpoint:

```bash
cargo run -p rl_engine -- checkpoints/azul_actor.ot
```

The executable expects the checkpoint path as its first argument and then reads
UAI commands from standard input. It can be used as an engine by the interface
executable with a descriptor such as:

```text
--engine "path=./target/debug/rl_engine args=checkpoints/azul_actor.ot tc=1+0" \
         "path=./target/debug/random_engine tc=1+0"
```
