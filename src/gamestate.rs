use crate::{
    BOWL_CAPACITY, Board,
    bag::Bag,
    movegen::{Bowl, Tile},
};

const TILE_TYPES: usize = 4;
const TILES_PER_TYPE: usize = 20;

pub struct GameState {
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    tiles_in_play: Vec<Tile>,
}

fn get_bowl_count(players: usize) -> usize {
    players * 2 + 1
}

fn get_default_tileset() -> Vec<Tile> {
    let mut tiles = Vec::new();
    for t in 0..TILE_TYPES + 1 {
        tiles.append(&mut vec![t as Tile; TILES_PER_TYPE]);
    }
    tiles
}

impl GameState {
    pub fn new(players: usize) -> Self {
        GameState {
            boards: vec![Board::new(); players],
            bowls: vec![Bowl::new(); get_bowl_count(players)],
            bag: Bag::new(get_default_tileset()),
            tiles_in_play: Vec::new(),
        }
    }

    fn get_tiles_not_in_play(&self) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for t in 0..5 {
            let in_play = self.tiles_in_play.iter().filter(|&&x| x == t).count();
            tiles.append(&mut vec![t as Tile; 20 - in_play]);
        }
        tiles
    }

    fn restock_bag(&mut self) {
        self.bag.restock(self.get_tiles_not_in_play());
    }
}
