# Azul Interface

The interface is the command-line harness for running Azul games between
engines and human players. It accepts engine and match configuration and is
designed around the [UAI protocol](protocol.md).

The current executable parses the options, starts configured child processes,
manages their standard streams, performs the UAI startup sequence, and runs a
two-to-four-player UAI game when all configured engines use UAI. It also retains
a local human-input game. Match scheduling, time controls, persistence,
and diagnostics/resource-limit options are parsed but are not all wired into
execution yet. The `--recover` option enables one bounded restart for a crashed
or protocol-erroring engine during a UAI game.

## Quick start

The `--engine` option requires at least two engine descriptors. Each descriptor
is a quoted, whitespace-separated list of `key=value` fields:

```text
cargo run -p interface -- \
  --engine "path=./target/debug/random_engine proto=uai tc=60+5" \
           "path=./target/debug/random_engine proto=uai tc=60+5" \
  --tournament round-robin \
  --games 10 \
  --out ./results.azl
```

The descriptor parser currently splits on whitespace and does not provide an
escaping or nested-quoting syntax. Quote the complete descriptor for the
shell, and keep each field's value free of spaces.

## Engine descriptors

`--engine` accepts two or more descriptors. `path` and `tc` are required for
each descriptor; the other fields are optional.

| Field | Value | Default | Description |
| --- | --- | --- | --- |
| `path` | executable path | none | Program to start for this engine. |
| `proto` | `uai` or `human` | `uai` | Protocol/interaction mode. |
| `tc` | `SECONDS` or `SECONDS+INCREMENT` | none | Incremental time control in seconds. It cannot be combined with `st`. |
| `st` | integer | none | Fixed time-per-move value, interpreted as milliseconds. It cannot be combined with `tc`. |
| `dir` | path | none | Working directory for the engine process. |
| `args` | one whitespace-free string | none | Additional argument text passed to the engine. |
| `name` | text | none | Display name for the engine. |
| `limit_mem` | unsigned integer | none | Per-engine memory limit value reserved for resource enforcement. |
| `limit_threads` | unsigned integer | none | Per-engine thread limit value reserved for resource enforcement. |

Example:

```text
--engine "path=engine-a proto=uai tc=60+2 dir=./runs name=Alpha" \
         "path=engine-b proto=uai st=500 name=Beta"
```

Unknown fields, missing `path`/time control, invalid numeric values, and using
both `tc` and `st` reject the command line.

## Match and tournament options

| Option | Value/default | Description |
| --- | --- | --- |
| `--tournament` | `gauntlet`, `round-robin`, `swiss`, or `random` | Selects the pairing strategy. |
| `--concurrency` | `N`, default `1` | Number of games intended to run concurrently. |
| `--out` | `PATH`, required | Output path for match results. |
| `--resume` | `PATH` | Resume a tournament from a saved results file. |
| `--rounds` | `N` | Number of rounds or matches to schedule. |
| `--games` | `N`, default `1` | Games per match. |
| `--repeat` | flag | Repeat the tournament or match. |
| `--max-games` | `N` | Hard cap on total games, including repeated runs. |
| `--seed` | unsigned `N` | Seed for reproducible tournament randomness. |
| `--openings` | `PATH` | Load starting positions or an opening book. |
| `--swap` | flag | Balance which engine receives each starting side. |
| `--timeout` | `N`, default `10` | Startup handshake/readiness timeout in seconds. Move deadlines come from each engine's `tc` or `st` setting. |
| `--recover` | flag | Restart a crashed or protocol-erroring engine once instead of immediately forfeiting. |

## Diagnostics and logging options

| Option | Description |
| --- | --- |
| `--version` | Print the interface version. |
| `--dry-run` | Parse configuration and validate setup without starting games. |
| `--check-engines` | Perform an engine handshake check. |
| `--summary` | Print results after rounds or matches. |
| `--debug` | Display engine input and output. |
| `--log` | Write engine commands, stdout, and stderr to a sibling `.log` file next to `--out`. |
| `--stderr` | Display engine stderr while preserving protocol stdout for the interface. |
| `--quiet` | Suppress normal output, leaving errors and final results. |

Clap also provides `--help`. Run `cargo run -p interface -- --help` to view
the generated option summary directly from the executable.
