use crate::Tile;

/// A structure for holding groups of tiles according to Azul's bowl rules.
#[derive(Debug, Default)]
pub struct Bowl {
    tiles: Vec<Tile>,
}

impl Bowl {
    /// Assigns this bowl's tiles.
    pub fn fill(&mut self, tiles: Vec<Tile>) {
        self.tiles = tiles;
        self.tiles.sort();
    }

    /// Extends this bowl's tiles.
    pub fn extend(&mut self, tiles: &Vec<Tile>) {
        self.tiles.extend(tiles);
        self.tiles.sort();
    }

    /// Returns the tiles of the given type from this bowl, as well as the remaining tiles. Calling this function
    /// clears this bowl's stored tiles.
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

    /// Returns a `Vec<Tile>` of all unique tile types owned by this bowl.
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
