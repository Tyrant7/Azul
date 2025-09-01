pub type Tile = u32;

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
}

impl Clone for Bowl {
    fn clone(&self) -> Self {
        Self {
            tiles: self.tiles.clone(),
        }
    }
}
