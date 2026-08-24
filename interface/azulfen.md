# AzulFEN v1

AzulFEN is the versioned, line-oriented snapshot format used to persist an
Azul [`GameState`](../movegen/README.md) and transmit it through UAI. Version 1
is deliberately strict: fields use exact separators, numeric values are
ASCII decimal, tile types are `0` through `4`, and serialized output is the
canonical form accepted by the parser.

A complete AzulFEN contains exactly one final newline and has this shape:

```text
azulfen:v1 <board> ; [<board> ; ...] | <bowl> <bowl> ... | <bag> | <active-player> <token-owner> <seed> <rng-state> <discarded-tiles>\n
```

The parser requires the `azulfen:v1` prefix. Unknown versions, unversioned
snapshots, extra fields, malformed separators, and missing final newlines are
rejected. The serializer always emits the complete five-field metadata form.

## Boards

Each board is terminated by ` ;`. Multiple boards are separated by one space:

```text
<placed> <holds> <row-bonuses> <column-bonuses> <tile-bonuses> <score> <penalties> <penalty-tiles> ;
```

### Placed wall

The placed wall has five slash-separated rows. Digits represent runs of empty
spaces and `-` represents an occupied space. Each row must describe exactly
five spaces; a digit `0` is not allowed. An occupied tile's type is determined
by its wall position, so the tile type is not written explicitly.

```text
5/5/2-2/5/5
```

### Pattern-line holds

The holds field is exactly ten digits: one tile-type/count pair for each of
the five pattern lines. Tile types are `0` through `4`; the count must not
exceed the line capacity. An empty line is encoded as `00`.

```text
0011000000
```

### Bonuses and scores

The row, column, and tile-type bonus fields are each exactly five binary
digits, in that order. The final four fields are non-negative decimal values:

1. current score;
2. occupied penalty spaces;
3. physical penalty tiles.

The physical penalty-tile count may not exceed the occupied penalty-space
count because the first-player token can occupy a penalty space without being
a physical tile.

For example, an empty board with score `7`, one occupied penalty space, and one
physical penalty tile is:

```text
5/5/5/5/5 0000000000 00000 00000 00000 7 1 1 ;
```

## Bowls

The bowl section contains exactly `2 * players + 2` bowls, including the centre
at index `0`. A bowl is either `-` for empty or a non-empty, sorted sequence
of tile-type digits:

```text
0123 - 0011
```

Tile types outside `0` through `4`, unsorted sequences, embedded whitespace,
and a bare empty string are invalid. The centre may contain more tiles than a
factory bowl during play.

## Bag

The bag is a sequence of tile-type digits in draw order. It may be empty, but
every character must be a tile type from `0` through `4`:

```text
03440140321203
```

## Metadata

Metadata contains exactly five space-separated fields:

```text
<active-player> <token-owner> <seed> <rng-state> <discarded-tiles>
```

- `active-player` is a zero-based player index.
- `token-owner` is a zero-based player index or `-` when unclaimed.
- `seed` is the optional initial `u64` seed or `-` when unavailable.
- `rng-state` is `xoshiro256plusplus:` followed by exactly 64 hexadecimal
  digits, preserving the current 256-bit generator state.
- `discarded-tiles` is the number of physical tiles returned to the discard
  pile.

Example:

```text
0 - 12345 xoshiro256plusplus:<64 hexadecimal digits> 7
```

The current RNG state, rather than the initial seed, guarantees exact future
shuffle reproduction. The seed is retained as episode/game metadata.

## Round trips and compatibility

`GameState::to_azul_fen` emits canonical version-1 output, and
`GameState::from_azul_fen` accepts only that version. A successful parse must
round-trip byte-for-byte through serialization. Older unversioned snapshots
are intentionally not accepted by the strict v1 parser; callers should
migrate them before loading.

AzulFEN is a state snapshot, not a move history. It contains enough board,
bag, bowl, turn, discard, seed, and RNG information to continue a game from
the serialized state.
