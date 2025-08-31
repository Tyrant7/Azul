use crate::{
    Board,
    bag::Bag,
    movegen::{Bowl, Tile},
};

pub struct GameState {
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
}

fn get_bowl_count(players: usize) -> usize {
    players * 2 + 1
}

fn get_tiles() -> Vec<Tile> {
    let mut tiles = Vec::new();
    for t in 0..6 {
        tiles.append(&mut vec![t as Tile; 20]);
    }
    tiles
}

impl GameState {
    pub fn new(players: usize) -> Self {
        let bag = Bag::new(get_tiles());
        let bowls = Vec::new();
        for bowl in 0..get_bowl_count(players) {
            bowls.push(Bowl::new());
        }

        GameState {
            boards: vec![Board::new(); players],
            bowls: vec![Bowl::new(); get_bowl_count(players)],
        }
    }
}
