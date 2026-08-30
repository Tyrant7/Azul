use crate::{
    Board, Tile,
    bag::Bag,
    board::BOARD_DIMENSION,
    bowl::Bowl,
    game_move::{
        BowlChoice::{self, Centre, Factory},
        IllegalMoveError, Move,
    },
};
use rand::{RngCore, SeedableRng};
use rand_xoshiro::rand_core::{SeedableRng as XoshiroSeedableRng, TryRng as XoshiroTryRng};

/// Adapts xoshiro256++ to the workspace's rand 0.9 traits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xoshiro256PlusPlus(rand_xoshiro::Xoshiro256PlusPlus);

impl Xoshiro256PlusPlus {
    /// Creates a xoshiro256++ generator from a 64-bit seed.
    pub fn from_seed_u64(seed: u64) -> Self {
        Self(<rand_xoshiro::Xoshiro256PlusPlus as XoshiroSeedableRng>::seed_from_u64(seed))
    }
}

impl RngCore for Xoshiro256PlusPlus {
    fn next_u32(&mut self) -> u32 {
        XoshiroTryRng::try_next_u32(&mut self.0).unwrap()
    }

    fn next_u64(&mut self) -> u64 {
        XoshiroTryRng::try_next_u64(&mut self.0).unwrap()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        XoshiroTryRng::try_fill_bytes(&mut self.0, dest).unwrap()
    }
}

impl SeedableRng for Xoshiro256PlusPlus {
    type Seed = [u8; 32];

    fn from_seed(seed: Self::Seed) -> Self {
        Self(<rand_xoshiro::Xoshiro256PlusPlus as XoshiroSeedableRng>::from_seed(seed))
    }
}

/// The number of tiles of each type in a complete game set.
const TILES_PER_TYPE: usize = 20;

/// The number of physical tiles in a complete Azul set.
pub const TOTAL_TILE_COUNT: usize = TILES_PER_TYPE * BOARD_DIMENSION;

/// The number of tiles placed in each factory bowl during round setup.
const BOWL_CAPACITY: usize = 4;

mod builder;
pub use builder::GameStateBuilder;

/// Describes why a game state could not be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GameStateError {
    /// The game must contain between two and four players.
    InvalidPlayerCount { players: usize },
    /// The active-player index is outside the board list.
    InvalidActivePlayer { active_player: usize },
    /// The first-player-token owner is outside the board list.
    InvalidFirstTokenOwner { player: usize },
    /// The number of bowls does not match the player count.
    InvalidBowlCount { expected: usize, actual: usize },
    /// The serialized random-generator state could not be decoded.
    InvalidRngState,
}

/// Complete mutable state for an Azul game.
///
/// The state contains one board per player, the factory bowls and centre area,
/// the tile bag, the active player, and the owner of the first-player token.
/// An optional seed records the seed used to initialize the game's random
/// stream. The current xoshiro state is available through
/// [`GameState::rng_state`] for exact snapshot and replay support. Physical
/// tiles returned to the discard pile are tracked separately.
#[derive(Debug)]
pub struct GameState {
    active_player: usize,
    boards: Vec<Board>,
    centre_bowl: Bowl,
    factory_bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
    rng: Xoshiro256PlusPlus,
    seed: Option<u64>,
    discarded_tiles: usize,
}

/// Returns the number of factory bowls plus the centre area for `players` players.
///
/// Azul uses `2n + 1` factory bowls
fn get_factory_bowl_count(players: usize) -> usize {
    players * 2 + 1
}

/// Validates the component relationships required by a playable game state.
fn validate_components(
    active_player: usize,
    boards: &[Board],
    factory_bowls: &[Bowl],
    first_token_owner: Option<usize>,
) -> Result<(), GameStateError> {
    if !(2..=4).contains(&boards.len()) {
        return Err(GameStateError::InvalidPlayerCount {
            players: boards.len(),
        });
    }
    let expected_bowls = get_factory_bowl_count(boards.len());
    if factory_bowls.len() != expected_bowls {
        return Err(GameStateError::InvalidBowlCount {
            expected: expected_bowls,
            actual: factory_bowls.len(),
        });
    }
    if active_player >= boards.len() {
        return Err(GameStateError::InvalidActivePlayer { active_player });
    }
    if let Some(player) = first_token_owner
        && player >= boards.len()
    {
        return Err(GameStateError::InvalidFirstTokenOwner { player });
    }
    Ok(())
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
    /// Only two-, three-, and four-player games are supported. Call
    /// [`GameState::setup_next_round`] before requesting or applying moves.
    pub fn new(players: usize, seed: u64) -> Result<Self, GameStateError> {
        if !(2..=4).contains(&players) {
            return Err(GameStateError::InvalidPlayerCount { players });
        }
        let mut rng = Xoshiro256PlusPlus::from_seed_u64(seed);
        Ok(GameState {
            active_player: 0,
            boards: vec![Board::default(); players],
            factory_bowls: vec![Bowl::default(); get_factory_bowl_count(players)],
            centre_bowl: Bowl::default(),
            bag: Bag::new(get_default_tileset(), &mut rng),
            first_token_owner: None,
            rng,
            seed: Some(seed),
            discarded_tiles: 0,
        })
    }

    /// Creates a new `GameStateBuilder`.
    pub fn builder() -> GameStateBuilder {
        GameStateBuilder::default()
    }

    ref_getters! {
        boards: Vec<Board>,
        factory_bowls: Vec<Bowl>,
        centre_bowl: Bowl,
        bag: Bag<Tile>,
    }

    value_getters! {
        first_token_owner: Option<usize>,
        active_player: usize,
        discarded_tiles: usize,
        seed: Option<u64>,
    }

    /// Returns the serialized current state of the random generator.
    ///
    /// The bytes are intended to be persisted with the game snapshot and
    /// restored through [`GameStateBuilder::set_rng_state`].
    pub fn rng_state(&self) -> Vec<u8> {
        bincode::serialize(&self.rng.0).expect("xoshiro state serialization cannot fail")
    }

    /// Returns the number of physical tiles still accounted for by the game.
    ///
    /// This includes tiles in the bag, bowls, boards, and discard pile. A
    /// complete game state accounts for all [`TOTAL_TILE_COUNT`] tiles.
    pub fn get_tile_count(&self) -> usize {
        self.bag.items().len()
            + self
                .factory_bowls
                .iter()
                .map(|bowl| bowl.get_tiles().len())
                .sum::<usize>()
            + self.centre_bowl.get_tiles().len()
            + self.boards.iter().map(Board::get_tile_count).sum::<usize>()
            + self.discarded_tiles
    }

    /// Resolves the previous round and prepares the next one.
    ///
    /// This places completed pattern-line tiles, applies board scoring and
    /// penalties, fills the factory bowls, restocks the bag when necessary,
    /// selects the next active player, and clears first-player-token ownership.
    pub fn setup_next_round(&mut self) {
        // Resolve each board before refilling the bowls.
        for board in self.boards.iter_mut() {
            self.discarded_tiles += board.place_holds();
        }

        // Fill factory bowls
        let (factory_bowls, bag) = (&mut self.factory_bowls, &mut self.bag);
        for bowl_idx in 0..factory_bowls.len() {
            let mut next: Vec<Tile> = bag.take(BOWL_CAPACITY).collect();
            if next.len() < BOWL_CAPACITY {
                // Rebuild the bag from tiles not currently held, placed, or dealt.
                let mut used_tiles = Vec::new();
                for board in &self.boards {
                    used_tiles.extend(board.get_active_tiles());
                }
                for bowl in factory_bowls.iter() {
                    used_tiles.extend(bowl.get_tiles().iter().copied());
                }
                used_tiles.extend(next.iter().copied());
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
                self.discarded_tiles = self.discarded_tiles.saturating_sub(unused_tiles.len());
                bag.restock(unused_tiles, &mut self.rng);
            }
            next.extend(bag.take(BOWL_CAPACITY - next.len()));
            factory_bowls[bowl_idx].fill(next);
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
        let mut moves = Vec::with_capacity((1 + self.factory_bowls.len()) * BOARD_DIMENSION);
        for tile in self.centre_bowl.get_tile_types() {
            board.for_each_valid_row_for_tile_type(tile, |row| {
                moves.push(Move {
                    bowl: Centre,
                    tile_type: tile,
                    row,
                })
            });
        }
        for (bowl_idx, bowl) in self.factory_bowls.iter().enumerate() {
            for tile in bowl.get_tile_types() {
                board.for_each_valid_row_for_tile_type(tile, |row| {
                    moves.push(Move {
                        bowl: Factory(bowl_idx),
                        tile_type: tile,
                        row,
                    })
                });
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
        let tiles = match choice.bowl {
            BowlChoice::Centre => self.centre_bowl.take_tiles(choice.tile_type),
            BowlChoice::Factory(idx) => self
                .factory_bowls
                .get_mut(idx)
                .ok_or(IllegalMoveError)?
                .take_tiles(choice.tile_type),
        };

        // The first player to take from the centre receives the token penalty.
        let penalty = if let BowlChoice::Centre = choice.bowl
            && self.first_token_owner.is_none()
        {
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
        self.centre_bowl.extend(&tiles.1);

        // Advance to the next player, wrapping at the end of the player list.
        self.active_player += 1;
        if self.active_player >= self.boards.len() {
            self.active_player = 0;
        }
        Ok(())
    }

    /// Returns `true` when no bowl contains any tiles.
    pub fn round_over(&self) -> bool {
        self.centre_bowl.get_tile_types().is_empty()
            && self
                .factory_bowls
                .iter()
                .all(|b| b.get_tile_types().is_empty())
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
