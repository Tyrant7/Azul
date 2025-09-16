# Azul Interface

The Azul Interface has been created to facilitate running games between various 
Azul playing programs using the UAI (Universal Azul Interface) protocol. 

See the [protocol docs](/interface/protocol.md) for more info on UAI. 

## Instructions for Use

The Azul Interface supports variety of command line arguments, as listed below. 

Engine Config

-engine path=[path to executable]: path the the engine executable
-engine dir=[working directory]: path for engines to read/write to
-engine args=["any string"]: any command line arguments to pass to the engine
-engine name=[engine name]: display name of this engine
-engine proto=[uai or human]: protocol type to use for this engine
-engine limit-mem=N: optional per-engine memory cap
-engine limit-threads=N: optional per-engine thread restriction

Timing Settings

tc: time control -> set number of seconds, or number with increment
st: fixed milliseconds per move, cannot be used with tc

Tournament/Match Settings

-tournament [gauntlet | round-robin | swiss | random]: tournament style
-concurrency N: sets number of concurrent games
-out file.azl: Saves the game results to the specified file
-resume file.azl: resume a stopped tournament from results/log
-rounds N: Sets number of matches in the tournamemt
-games N: sets the number of games per match
-repeat: Repeats the tournament or match indefinitely
-max-games N: hard cap on total games even if `-repeat` enable
-seed N: RNG seed for reproducibility
-openings file.azl: load a set of starting positions/opening book for fair testing
-swap: ensure each engine plays both "first" and "second" positions equally

-timeout N: max seconds to wait for an engine to reply to the start command before forfeitting the match
-crash-mode [forfeit | restart]: what to do when an engine crashes mid-match

Debugging and logging

-version: prints interface version
-dry-run: parse config, validate engines exist, but don't start games
-check-engines: runs each engine with a handshake to confirm it's alive
-summary: prints human-readible results after each round/match
-debug: displays all engine input and output
-log: writes all engine communication to a log file
-stderr: shows error messages from the command line or engines
-quiet: surpress program output (only errors and final results are printed)
