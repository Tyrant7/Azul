const BOWL_CAPACITY: usize = 4;

pub type Tile = u32;

pub struct Bowl {
    tiles: [Tile; BOWL_CAPACITY],
}

impl Bowl {
    pub fn new(tiles: [Tile; BOWL_CAPACITY]) -> Bowl {
        Bowl { tiles }
    }
}
