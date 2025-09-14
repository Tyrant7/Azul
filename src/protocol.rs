/*
AzulFEN works as follows:

-
Boards:

Each board's placed tiles are broken down into their own FEN-style string where numbers represent N empty spaces,
"/" denotes a new line, and a - represents a tile in that position
e.x.  5/5/5/5/5 is an empty board
while 5/5/2-2/5/5 would have a single tile in the centre

Each row of a board's hold section can be encoded with two numbers. The first represents the tile type in that row,
and the second representing the number of tiles. The encodings for each row are written sequentially
e.x.  0042000000 corresponds to 2 tiles of type 4 in the second row

For each board, the collected bonuses also need to be known. Each bonus type is encoded individually, in the order of
[row, column, tile_type], and sequentially to one another, with a space in between where 0 is an uncollected bonus
and 1 is a collected bonus.
e.x.  00001 00000 00000 corresponds to having collected only the horizontal bonus for the final row

The score and penalty tiles for each board and encoded and single numbers at the end of the FEN
e.x.  10 3 corresponds to 10 score and 3 penalty tiles

And finally, each board FEN is separated by a semi-colon

Altogether a typical board FEN may look something like follows:
2-1-/-4/--3/5/4- 0011000013 00000 00000 00000 7 1 ;

-
Bowls:

The bowl's section is prefixed with a "|" character

Each bowl is encoded as a sequence of numbers corresponding to tile types, each with a space in between
An empty bowl is denoted with a "-"
e.x.  000234 - 1132 would correspond to three unique bowls, one centre, one empty, and one full

-
Bag:

The bag's section is prefixed with another "|" character

The bag is simply listed as a sequence of numbers corresponding to tile types
e.x.  03440140321203

-
Active player and first player token:

Finally, the active player and first player token owner and encoded at the end in order as two numbers,
once again, prefixed with a "|" character
e.x.  0 2 corresponds to the active player being player 0, and the first player token owner being player 2
If nobody owns the first player token, then "-" will be written in its place

-
In full, a complete AzulFEN may look something like the following

2-1-/-4/--3/5/4- 0011000013 00000 00000 00000 7 1 ;
1--1-/-4/1-3/4-/4- 0000220013 00000 00000 00000 10 0 ;
| 0123003 - - - 0123 0001
| 0133041230412404142
| 0 -

AzulFENs should be outputted on a single-line, with a newline as the final character
*/

use crate::bowl::{Move, Row, Tile};
use clap::{Parser, ValueEnum};
use std::num::ParseIntError;

#[derive(Parser)]
#[command(name = "azul-engine")]
#[command(about = "An azul engine")]
struct Cli {
    #[arg(long, value_enum, default_value_t = Protocol::Human)]
    protocol: Protocol,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Protocol {
    Human,
    UCILike,
}

impl Protocol {
    pub fn extract() -> Protocol {
        let cli = Cli::parse();
        println!("Running with protocol: {:?}", cli.protocol);
        cli.protocol
    }
}

pub trait ProtocolFormat {
    fn fmt_human(&self) -> String;
    fn fmt_uci_like(&self) -> String;

    fn fmt_protocol(&self, protocol: Protocol) -> String {
        match protocol {
            Protocol::Human => self.fmt_human(),
            Protocol::UCILike => self.fmt_uci_like(),
        }
    }
}

#[derive(Debug)]
pub struct ParseGameStateError;

#[derive(Debug)]
pub struct ParseMoveError;

impl From<ParseIntError> for ParseMoveError {
    fn from(_: ParseIntError) -> Self {
        ParseMoveError
    }
}

/*
Here we expect moves in the format of `bowl, tile_type, row` where each input is a two-digit number
ex. 040102 would correspond to the fourth bowl, first tile type, and second row of our own board
Note: Bowl 00 will always correspond to the centre area, and row 00 will always correspond to the penalty area
*/
pub fn parse_move(input: &str) -> Result<Move, ParseMoveError> {
    if input.len() != 6 {
        return Err(ParseMoveError);
    }
    let (bowl, other) = input.split_at(2);
    let (tile_type, row) = other.split_at(2);

    let bowl = bowl.parse::<usize>()?;
    let tile_type = tile_type.parse::<Tile>()?;
    let row = row.parse::<usize>()?;
    let row = if row == 0 {
        Row::Floor
    } else {
        Row::Wall(row - 1)
    };
    Ok(Move {
        bowl,
        tile_type,
        row,
    })
}
