use clap::{Parser, ValueEnum};
use std::num::ParseIntError;

use crate::{Tile, game_move::Move, row::Row};

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
