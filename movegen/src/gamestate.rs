use crate::{
    Board, Tile,
    bag::Bag,
    board::BOARD_DIMENSION,
    bowl::Bowl,
    game_move::{IllegalMoveError, Move},
};

/// The number of tiles of each type in a complete game set.
const TILES_PER_TYPE: usize = 20;

/// The number of tiles placed in each factory bowl during round setup.
const BOWL_CAPACITY: usize = 4;

/// The index used for the centre area, which is represented as a bowl in this model.
const CENTRE_BOWL_IDX: usize = 0;

mod builder;
pub use builder::GameStateBuilder;
use rand::{SeedableRng, rngs::SmallRng};

/// Complete mutable state for an Azul game.
///
/// The state contains one board per player, the factory bowls and centre area,
/// the tile bag, the active player, and the owner of the first-player token.
#[derive(Debug)]
pub struct GameState {
    active_player: usize,
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
    rng: SmallRng,
}

/// Returns the number of factory bowls plus the centre area for `players` players.
///
/// Azul uses `2n + 1` factory bowls; this model adds one bowl for the centre.
fn get_bowl_count(players: usize) -> usize {
    players * 2 + 2
}

/// Generates the standard game set with [`TILES_PER_TYPE`] tiles of each type.
fn get_default_tileset() -> Vec<Tile> {
    let mut tiles = Vec::new();
    // Azul has one tile type for each wall dimension.
    for t in 0..BOARD_DIMENSION {
        tiles.append(&mut vec![t as Tile; TILES_PER_TYPE]);
    }
    tiles
}

impl GameState {
    /// Creates a new game with empty bowls and boards for `players` players.
    ///
    /// Call [`GameState::setup_next_round`] before requesting or applying moves.
    pub fn new(players: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        GameState {
            active_player: 0,
            boards: vec![Board::default(); players],
            bowls: vec![Bowl::default(); get_bowl_count(players)],
            bag: Bag::new(get_default_tileset(), &mut rng),
            first_token_owner: None,
            rng,
        }
    }

    /// Creates a new `GameStateBuilder`.
    pub fn builder() -> GameStateBuilder {
        GameStateBuilder::default()
    }

    getters! {
        active_player: usize,
        boards: Vec<Board>,
        bowls: Vec<Bowl>,
        bag: Bag<Tile>,
        first_token_owner: Option<usize>,
    }

    /// Resolves the previous round and prepares the next one.
    ///
    /// This places completed pattern-line tiles, applies board scoring and
    /// penalties, fills the factory bowls, restocks the bag when necessary,
    /// selects the next active player, and clears first-player-token ownership.
    pub fn setup_next_round(&mut self) {
        // Resolve each board before refilling the bowls.
        for board in self.boards.iter_mut() {
            board.place_holds();
        }

        // Fill factory bowls; index zero is reserved for the centre.
        let (bowls, bag) = (&mut self.bowls, &mut self.bag);
        for bowl in bowls.iter_mut().skip(1) {
            let mut next: Vec<Tile> = bag.take(BOWL_CAPACITY).collect();
            if next.len() < BOWL_CAPACITY {
                // Rebuild the bag from tiles not currently held or placed.
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
                bag.restock(unused_tiles, &mut self.rng);
            }
            next.extend(bag.take(BOWL_CAPACITY - next.len()));
            bowl.fill(next.clone());
        }

        // The first-player-token owner starts the new round.
        self.active_player = self.first_token_owner.unwrap_or_default();
        self.first_token_owner = None;
    }

    /// Returns all legal moves for the active player.
    ///
    /// The list includes moves that send tiles to the floor and therefore incur
    /// penalties. An empty bowl contributes no moves.
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

    /// Applies `choice` to the current state.
    ///
    /// Returns [`IllegalMoveError`] when `choice` is not legal for the active
    /// player and current bowl contents.
    pub fn make_move(&mut self, choice: &Move) -> Result<(), IllegalMoveError> {
        let valid_moves = self.get_valid_moves();
        if !valid_moves.contains(choice) {
            return Err(IllegalMoveError);
        }

        // Remove the selected tile type from the chosen bowl.
        let tiles = self
            .bowls
            .get_mut(choice.bowl)
            .ok_or(IllegalMoveError)?
            .take_tiles(choice.tile_type);

        // The first player to take from the centre receives the token penalty.
        let penalty = if choice.bowl == CENTRE_BOWL_IDX && self.first_token_owner.is_none() {
            self.first_token_owner = Some(self.active_player);
            1
        } else {
            0
        };

        // Put the selected tiles into the active player's destination.
        let active_board = self
            .boards
            .get_mut(self.active_player)
            .expect("Invalid player");
        active_board.hold_tiles(choice.tile_type, tiles.0.len(), choice.row, penalty)?;

        // Move the other tiles from the selected bowl to the centre.
        self.bowls
            .get_mut(CENTRE_BOWL_IDX)
            .expect("Invalid bowl")
            .extend(&tiles.1);

        // Advance to the next player, wrapping at the end of the player list.
        self.active_player += 1;
        if self.active_player >= self.boards.len() {
            self.active_player = 0;
        }
        Ok(())
    }

    /// Returns `true` when no bowl contains any tiles.
    pub fn round_over(&self) -> bool {
        self.bowls.iter().all(|b| b.get_tile_types().is_empty())
    }

    /// Returns `true` when any player has completed a horizontal wall line.
    pub fn is_game_over(&self) -> bool {
        self.boards.iter().any(|b| b.count_horizontal_lines() > 0)
    }

    /// Returns the index of the board selected as the winner.
    ///
    /// Scores are compared first, then completed horizontal lines. If both
    /// values are equal, the current iterator-based implementation selects the
    /// later board index.
    pub fn get_winner(&self) -> usize {
        self.boards
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| (b.get_score(), b.count_horizontal_lines()))
            .unwrap()
            .0
    }
}
