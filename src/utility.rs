use std::num::ParseIntError;

use crate::bowl::{Move, Tile};

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
Note: Bowl 00 will always correspond to the centre area
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
    Ok(Move {
        bowl,
        tile_type,
        row,
    })
}
