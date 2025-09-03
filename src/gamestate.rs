use crate::{
    BOWL_CAPACITY, Board,
    bag::Bag,
    bowl::{Bowl, IllegalMoveError, Move, Tile},
};

const TILE_TYPES: usize = 4;
const TILES_PER_TYPE: usize = 20;

const CENTRE_BOWL_IDX: usize = 0;

#[derive(Debug)]
pub struct GameState {
    active_player: usize,
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
}

fn get_bowl_count(players: usize) -> usize {
    players * 2 + 2
}

fn get_default_tileset() -> Vec<Tile> {
    let mut tiles = Vec::new();
    for t in 0..TILE_TYPES {
        tiles.append(&mut vec![t as Tile; TILES_PER_TYPE]);
    }
    tiles
}

impl GameState {
    pub fn new(players: usize) -> Self {
        GameState {
            active_player: 0,
            boards: vec![Board::new(); players],
            bowls: vec![Bowl::new(); get_bowl_count(players)],
            bag: Bag::new(get_default_tileset()),
            first_token_owner: None,
        }
    }

    pub fn setup(&mut self) {
        // Fill each bowl, skipping the centre
        let (bowls, bag) = (&mut self.bowls, &mut self.bag);
        for bowl in bowls.iter_mut().skip(1) {
            let mut next: Vec<Tile> = bag.take(BOWL_CAPACITY).collect();
            if next.len() < BOWL_CAPACITY {
                // Refill the bag with all tiles currently not in play
                let mut used_tiles = Vec::new();
                for board in &self.boards {
                    used_tiles.extend(board.get_active_tiles());
                }
                let mut unused_tiles = Vec::new();
                for t in 0..TILE_TYPES {
                    unused_tiles.append(&mut vec![
                        t as Tile;
                        TILES_PER_TYPE
                            - used_tiles
                                .iter()
                                .filter(|&&x| x == t as Tile)
                                .count()
                    ]);
                }
                bag.restock(unused_tiles);
            }
            next.extend(bag.take(BOWL_CAPACITY - next.len()));
            bowl.fill(next.clone());
        }

        // At the end of setup, the player with the first player's token goes first
        self.active_player = self.first_token_owner.unwrap_or_default();
        self.first_token_owner = None;
    }

    pub fn get_valid_moves(&self) -> Vec<Move> {
        let board = self.boards.get(self.active_player).expect("Invalid player");
        let mut moves = Vec::new();
        for (bowl_idx, bowl) in self.bowls.iter().enumerate() {
            for tile in bowl.get_tile_types() {
                for row in board.get_valid_rows_for_tile_type(tile) {
                    moves.push(Move {
                        bowl: bowl_idx,
                        tile_type: tile,
                        row,
                    });
                }
            }
        }
        moves
    }

    pub fn make_move(&mut self, choice: Move) -> Result<(), IllegalMoveError> {
        let valid_moves = self.get_valid_moves();
        if !valid_moves.contains(&choice) {
            return Err(IllegalMoveError);
        }

        // Get the tiles and update the bowls
        let tiles = self
            .bowls
            .get_mut(choice.bowl)
            .ok_or(IllegalMoveError)?
            .take_tiles(choice.tile_type);

        // A penalty is given if we're the first player to pick from the centre
        let penalty = if choice.bowl == CENTRE_BOWL_IDX && self.first_token_owner.is_none() {
            self.first_token_owner = Some(self.active_player);
            1
        } else {
            0
        };

        // Put the tiles into the appropriate row
        let active_board = self
            .boards
            .get_mut(self.active_player)
            .expect("Invalid player");
        active_board.hold_tiles(choice.tile_type, tiles.0.len(), choice.row, penalty)?;

        // Move the remaining tiles to the centre
        self.bowls
            .get_mut(CENTRE_BOWL_IDX)
            .expect("Invalid bowl")
            .extend(&tiles.1);

        // Cycle to the next player's turn
        self.active_player += 1;
        if self.active_player >= self.boards.len() {
            self.active_player = 0;
        }

        // If we only had one valid move, let's setup for the next round
        if valid_moves.len() == 1 {
            for board in self.boards.iter_mut() {
                board.place_holds();
            }
            self.setup();
        }
        Ok(())
    }
}
