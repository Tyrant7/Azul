# Universal Azul Interface (UAI)

UAI is the line-oriented protocol used by the Azul interface to communicate
with engine processes. It follows the shape of the Universal Chess Interface
(UCI), while using Azul-specific position and move formats.

This document is the project specification for UAI. The executable currently
contains the configuration, parsing, child-process stream, startup handshake,
basic turn-loop pieces, and bounded engine recovery. Match scheduling,
structured diagnostics, and resource limits remain implementation work.

## Transport

- The interface starts an engine as a child process.
- Commands are UTF-8 text lines written to the engine's standard input.
- Engine responses are UTF-8 text lines written to standard output.
- A command or response ends at the next newline. Leading and trailing spaces
  are ignored; fields are separated by one or more spaces.
- Diagnostic output must go to standard error so it cannot be mistaken for a
  protocol response.
- Commands that are not understood or cannot be parsed produce an `error`
  response when the sender expects a response.

## Command lifecycle

The interface and engine use the following lifecycle:

1. The interface starts the engine and sends `uai`.
2. The engine responds with its identity, optional options, and `uaiok`.
3. The interface sends `isready` and waits for `readyok` before starting a
   game.
4. The interface sends `newgame`, followed by a `position` command.
5. The interface sends `go movetime <milliseconds>` for fixed-time engines, or
  `go clock <remaining-milliseconds> <increment-milliseconds>` for
  clock-controlled engines.
6. The engine responds with `bestmove` and may send `info` lines first.
7. The interface sends another `position`/`go` pair, or sends `quit` to end
   the process.

An engine must finish any outstanding work before replying to `isready`.
`quit` should be handled promptly and does not require a response.

## Interface-to-engine commands

| Command | Required | Description |
| --- | --- | --- |
| `uai` | Yes | Requests engine identification and capability information. |
| `isready` | Yes | Requests `readyok` after pending initialization is complete. |
| `newgame` | Yes | Resets game-specific search and position state. |
| `position startpos` | Yes | Loads the standard empty Azul position. |
| `position fen <AzulFEN>` | Yes | Loads the position encoded by [`azulfen.md`](./azulfen.md). |
| `go` | Yes | Requests a move for the active player. |
| `go movetime <milliseconds>` | Yes | Requests a move with a fixed millisecond budget. |
| `go clock <remaining> <increment>` | Yes | Supplies the active engine's remaining clock and increment in milliseconds; the engine chooses its own allocation. |
| `stop` | Planned | Stops an in-progress search; the engine must still return `bestmove`. |
| `quit` | Yes | Requests process termination. |

The `position` command must be followed by a complete position. A future
`moves` suffix may be added if the interface needs to transmit a move history,
but current snapshots use complete AzulFEN states instead.

## Engine-to-interface responses

| Response | Required | Description |
| --- | --- | --- |
| `id name <text>` | Yes after `uai` | Reports the engine's display name. |
| `id author <text>` | Yes after `uai` | Reports the engine author or organization. |
| `option name <name> type <type> ...` | Optional | Describes a configurable engine option. |
| `uaiok` | Yes after `uai` | Completes the UAI handshake. |
| `readyok` | Yes after `isready` | Confirms the engine is ready. |
| `info ...` | Optional during `go` | Non-terminal search progress or evaluation update. The interface may ignore the line and must continue waiting for `bestmove`. |
| `bestmove <move>` | Yes after `go` | Returns the selected move in the [move format](#move-format). |
| `error <text>` | When needed | Terminal failure for the current command or position. The interface must abort that operation and must not treat the line as a move or readiness response; the match runner may restart the process once when recovery is enabled. |

An engine must emit exactly one `uaiok` for each completed `uai` handshake and
exactly one `bestmove` for each `go` request, unless the process exits or the
interface reports a fatal error.

### Response prefixes

The response prefix determines how the interface handles the rest of the line:

- `info ` is an asynchronous, non-terminal update. An engine may emit any
  number of `info` lines while processing `go`; they do not complete the
  request and do not replace `bestmove`.
- `error ` is a terminal failure response. The remainder of the line is a
  human-readable explanation. It applies to the current command or position,
  and the interface should stop waiting for that command's normal response.
  The match runner may then replace the process and retry once when recovery
  is enabled.

For example, `info depth 8 score 12` reports search progress, while `error
invalid position` reports that the engine could not accept the requested
position. Diagnostic logging that is unrelated to the protocol belongs on
standard error, not in an `info` or `error` response.

## Move format

A move is six decimal digits made from three two-digit, zero-based components:

```text
BBTTRR
```

| Component | Range | Meaning |
| --- | --- | --- |
| `BB` | `00` through the final bowl index | Factory bowl index. `00` is the centre. |
| `TT` | `00` through `04` | Tile type. |
| `RR` | `00` through `05` | Destination. `00` is the floor; `01` through `05` are wall rows `0` through `4`. |

For example, `040102` selects tile type `01` from bowl index `04` and sends it
to wall row index `01`. The move's legality depends on the current position;
the six-digit parser only decodes its fields.

## Positions

UAI positions use the project's AzulFEN representation. AzulFEN preserves the
boards, bowls, bag, active player, first-player-token owner, optional initial
seed, current xoshiro256++ state, and discarded-tile count. See the
[`AzulFEN specification`](./azulfen.md) for the grammar and round-trip rules.

The interface rejects malformed or invalid positions before asking an engine to
search them. The turn loop sends a complete authoritative position before each
`go` request and validates the returned move against the local state.

For a fixed per-move control (`st`), the interface sends that value as the
`movetime` upper bound on every turn. For an increment control (`tc=BASE+INC`),
it starts the player's clock at `BASE` seconds, sends the current remaining
clock and `INC` to the engine, subtracts the elapsed search time, and adds
`INC` seconds after a successful move. The engine chooses its own search
allocation within the interface's hard deadline. A response that arrives
after the deadline is a timeout, and `info` lines do not extend that deadline.

## Errors and termination

Errors are single-line responses:

```text
error <human-readable explanation>
```

Errors should identify the command or field that failed without exposing stack
traces or other diagnostic output on standard output. Fatal transport errors,
engine crashes, and timeouts are managed by the interface process. The
interface's `--timeout` setting is
used for startup handshake/readiness waits; move deadlines come from the
configured engine time control.

## Engine failure policy

During a game, timeouts, malformed responses, and illegal moves forfeit the
engine immediately. Crashes, EOF, broken pipes, and explicit `error` responses
are recoverable process failures when `--recover` is enabled. Recovery is
limited to one restart per engine per game: the replacement repeats the UAI
handshake, readiness check, `newgame`, and current AzulFEN before retrying the
same turn. If recovery is disabled, or the retry also fails, the engine
forfeits. An accepted move is never undone.

## Compatibility

This is a project-local draft. Changes to command names, response grammar, or
move/position encoding must be made here first and covered by interface tests.
The protocol is intentionally versionless until command dispatch is
implemented; a future handshake option can advertise a protocol version if
compatibility requirements emerge.
