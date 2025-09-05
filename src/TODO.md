# TODO

## Movegen
- Finish and document UAI implementation (universal Azul interface)
    - Implement AzulFENs -> from and to fens for gamestate
- UAI and AzulFEN documentation and docstrings overall
- Implement tests
- Cut out all the CLI code and put that into a separate project, importing the rest
  as an Azul-movegen crate
- Figure out if I want to use separate repos for each crate or if I want to keep them separate
  but within the same repo
- Then from there we can build up the engine manager using two engines as separate processes

## Engine
- Research techniques to use
    - Likely try PPO first