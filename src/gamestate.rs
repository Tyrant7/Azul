use std::ffi::os_str::Display;

use crate::{
    BOWL_CAPACITY, Board,
    bag::Bag,
    movegen::{Bowl, Tile},
};

const TILE_TYPES: usize = 4;
const TILES_PER_TYPE: usize = 20;

#[derive(Debug)]
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

    pub fn setup(&mut self) {
        let (bowls, bag) = (&mut self.bowls, &mut self.bag);
        for bowl in bowls.iter_mut() {
            let mut next: Vec<Tile> = bag.take(BOWL_CAPACITY).collect();
            if next.len() < BOWL_CAPACITY {
                // Refill the bag with all tiles currently not in play
                let mut tiles = Vec::new();
                for t in 0..5 {
                    let in_play = self.tiles_in_play.iter().filter(|&&x| x == t).count();
                    tiles.append(&mut vec![t as Tile; 20 - in_play]);
                }
                bag.restock(tiles);
            }
            next.extend(bag.take(BOWL_CAPACITY - next.len()));
            bowl.fill(next.clone());
            self.tiles_in_play.append(&mut next);
        }
    }

    pub fn select_tiles(mut self, player_id: usize, bowl: usize, tile_type: Tile, row: usize) {
        let bowl = self.bowls.get_mut(bowl).expect("Invalid bowl");
        let tiles = bowl.take_tiles(tile_type);

        println!("tiles: {:?}", tiles);

        // Put the tiles into that row
    }
}
