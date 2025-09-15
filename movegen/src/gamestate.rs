use crate::{
    Board, Tile,
    bag::Bag,
    board::BOARD_DIMENSION,
    bowl::Bowl,
    game_move::{IllegalMoveError, Move},
};

/// The number of tiles of each type to be added to the bag at the beginning of the game, and to be
/// used for reference during round setup.
const TILES_PER_TYPE: usize = 20;

/// The number of tiles that each bowl is restocked to contain during the roubnd setup.
const BOWL_CAPACITY: usize = 4;

/// The index of the centre tile space. Is area is not technically a bowl in the original game, but for
/// simplicity of the code, this decision has been made here.
const CENTRE_BOWL_IDX: usize = 0;

/// Represents a complete gamestate for a given number of players.
/// Supports generation from and serialization to a custom AzulFEN [TODO: link].
#[derive(Debug)]
pub struct GameState {
    active_player: usize,
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
}

/// Bowl formula is given by 2n + 1, with an additional bowl for the centre space.
fn get_bowl_count(players: usize) -> usize {
    players * 2 + 2
}

/// Generates a default tileset for a game setup.
/// By default, [TILES_PER_TYPE] of each tile type are given.
fn get_default_tileset() -> Vec<Tile> {
    let mut tiles = Vec::new();
    // There should always be the same number of tiles as board width
    for t in 0..BOARD_DIMENSION {
        tiles.append(&mut vec![t as Tile; TILES_PER_TYPE]);
    }
    tiles
}

impl GameState {
    /// Creates a new gamestate for the given number of players.
    pub fn new(players: usize) -> Self {
        GameState {
            active_player: 0,
            boards: vec![Board::default(); players],
            bowls: vec![Bowl::default(); get_bowl_count(players)],
            bag: Bag::new(get_default_tileset()),
            first_token_owner: None,
        }
    }

    /// Creates a new `GameStateBuilder`.
    pub fn builder() -> GameStateBuilder {
        GameStateBuilder::default()
    }

    /// Performs a variety of tasks to setup the beginning of a round, including
    /// - Placing held tiles
    /// - Applying previous round penalties
    /// - Refilling bowls
    /// - Restocking the bag, if necessary
    /// - Determining the first player
    /// - Resetting the first player token holder
    pub fn setup_next_round(&mut self) {
        // Place each board's held tiles and apply penalties
        for board in self.boards.iter_mut() {
            board.place_holds();
        }

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
                for t in 0..BOARD_DIMENSION {
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

    /// Returns a list of all valid moves in the current gamestate.
    /// This list includes penalizing moves, such as placing tiles to the floor position.
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

    /// Makes a move, modifying the current gamestate.
    /// Will error if the given move is illegal.
    pub fn make_move(&mut self, choice: &Move) -> Result<(), IllegalMoveError> {
        let valid_moves = self.get_valid_moves();
        if !valid_moves.contains(choice) {
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
        Ok(())
    }

    /// Returns true if all bowls are empty, otherwise false.
    pub fn round_over(&self) -> bool {
        self.bowls.iter().all(|b| b.get_tile_types().is_empty())
    }

    /// Returns true if any player has completed a horizontal line on their board.
    pub fn is_game_over(&self) -> bool {
        self.boards.iter().any(|b| b.count_horizontal_lines() > 0)
    }

    /// Gets the index of the board with the highest score.
    /// In the case of a tie, the number of horizontal lines are used.
    /// If there is still a tie, the lower-indexed player will be returned.  
    pub fn get_winner(&self) -> usize {
        self.boards
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| (b.get_score(), b.count_horizontal_lines()))
            .unwrap()
            .0
    }
}

#[derive(Default)]
pub struct GameStateBuilder {
    active_player: usize,
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
}

impl GameStateBuilder {
    pub fn active_player(mut self, active_player: usize) -> Self {
        self.active_player = active_player;
        self
    }

    pub fn boards(mut self, boards: Vec<Board>) -> Self {
        self.boards = boards;
        self
    }

    pub fn bowls(mut self, bowls: Vec<Bowl>) -> Self {
        self.bowls = bowls;
        self
    }

    pub fn bag(mut self, bag: Bag<Tile>) -> Self {
        self.bag = bag;
        self
    }

    pub fn first_token_owner(mut self, first_token_owner: Option<usize>) -> Self {
        self.first_token_owner = first_token_owner;
        self
    }

    pub fn build(self) -> GameState {
        GameState {
            active_player: self.active_player,
            boards: self.boards,
            bowls: self.bowls,
            bag: self.bag,
            first_token_owner: self.first_token_owner,
        }
    }
}
