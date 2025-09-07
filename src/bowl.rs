use crate::protocol::ProtocolFormat;

pub type Tile = usize;

#[derive(Debug)]
pub struct Bowl {
    tiles: Vec<Tile>,
}

impl Bowl {
    pub fn new() -> Self {
        Bowl { tiles: Vec::new() }
    }

    pub fn from_bowl_fen(bowl_fen: &str) -> Self {
        if bowl_fen.chars().nth(0).expect("Invalid bowl FEN") == '-' {
            Bowl { tiles: Vec::new() }
        } else {
            Bowl {
                tiles: bowl_fen
                    .chars()
                    .map(|c| c.to_string().parse::<Tile>().expect("Invalid bowl FEN"))
                    .collect(),
            }
        }
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

impl ProtocolFormat for Bowl {
    fn fmt_human(&self) -> String {
        if self.tiles.is_empty() {
            return String::from("-");
        }
        self.tiles.iter().map(|t| t.to_string()).collect()
    }

    fn fmt_uci_like(&self) -> String {
        self.fmt_human()
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
