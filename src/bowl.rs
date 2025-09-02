pub type Tile = usize;

#[derive(Debug)]
pub struct Bowl {
    tiles: Vec<Tile>,
}

impl Bowl {
    pub fn new() -> Bowl {
        Bowl { tiles: Vec::new() }
    }

    pub fn fill(&mut self, tiles: Vec<Tile>) {
        self.tiles = tiles;
    }

    pub fn take_tiles(&mut self, tile_type: Tile) -> Vec<Tile> {
        let mut take = Vec::new();
        let mut keep = Vec::new();
        for &tile in self.tiles.iter() {
            if tile == tile_type {
                take.push(tile);
            } else {
                keep.push(tile);
            }
        }
        self.tiles = keep;
        take
    }

    pub fn get_tile_types(&self) -> Vec<Tile> {
        let mut tiles = self.tiles.clone();
        tiles.sort();
        tiles.dedup();
        tiles
    }
}

impl Clone for Bowl {
    fn clone(&self) -> Self {
        Self {
            tiles: self.tiles.clone(),
        }
    }
}

#[derive(Debug)]
pub struct Move {
    pub bowl: usize,
    pub tile_type: Tile,
    pub row: usize,
}

impl PartialEq for Move {
    fn eq(&self, other: &Self) -> bool {
        self.bowl == other.bowl && self.tile_type == other.tile_type && self.row == other.row
    }
}

#[derive(Debug)]
pub struct IllegalMoveError;
