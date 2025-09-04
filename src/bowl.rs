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
        self.tiles.sort();
    }

    pub fn extend(&mut self, tiles: &Vec<Tile>) {
        self.tiles.extend(tiles);
        self.tiles.sort();
    }

    pub fn take_tiles(&mut self, tile_type: Tile) -> (Vec<Tile>, Vec<Tile>) {
        let mut take = Vec::new();
        let mut keep = Vec::new();
        for &tile in self.tiles.iter() {
            if tile == tile_type {
                take.push(tile);
            } else {
                keep.push(tile);
            }
        }
        self.tiles.clear();
        (take, keep)
    }

    pub fn get_tile_types(&self) -> Vec<Tile> {
        let mut tiles = self.tiles.clone();
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

impl std::fmt::Display for Bowl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for tile in self.tiles.iter() {
            write!(f, "{}", tile)?;
        }
        if self.tiles.is_empty() {
            write!(f, "-")?;
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct Move {
    pub bowl: usize,
    pub tile_type: Tile,
    pub row: Row,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Row {
    Floor,
    Wall(usize),
}

impl std::fmt::Display for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                Row::Floor => "-".to_string(),
                Row::Wall(i) => i.to_string(),
            }
        )
    }
}

#[derive(Debug)]
pub struct IllegalMoveError;
