use crate::Tile;

/// A sorted collection of tiles held by a factory bowl or the centre.
#[derive(Debug, Default)]
pub struct Bowl {
    tiles: Vec<Tile>,
}

impl Bowl {
    /// Creates a bowl containing `tiles` in sorted order.
    pub fn from_tiles(tiles: Vec<Tile>) -> Self {
        let mut bowl = Bowl::default();
        bowl.fill(tiles);
        bowl
    }

    /// Replaces the bowl's tiles and sorts them by tile type.
    pub fn fill(&mut self, tiles: Vec<Tile>) {
        self.tiles = tiles;
        self.tiles.sort();
    }

    /// Adds `tiles` to the bowl and keeps the stored tiles sorted.
    pub fn extend(&mut self, tiles: &Vec<Tile>) {
        self.tiles.extend(tiles);
        self.tiles.sort();
    }

    /// Removes all tiles of `tile_type` and returns them with the other tiles.
    ///
    /// The bowl is empty after this call; callers can place the returned remaining
    /// tiles into another bowl as required by the game rules.
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

    /// Returns the distinct tile types in sorted order.
    pub fn get_tile_types(&self) -> Vec<Tile> {
        let mut tiles = self.tiles.clone();
        tiles.dedup();
        tiles
    }

    /// Returns all tiles currently in the bowl in sorted order.
    pub fn get_tiles(&self) -> &Vec<Tile> {
        &self.tiles
    }
}

impl Clone for Bowl {
    fn clone(&self) -> Self {
        Self {
            tiles: self.tiles.clone(),
        }
    }
}
