use rand::{Rng, rng};

use super::{GameState, GameStateError, Xoshiro256PlusPlus, validate_components};
use crate::{Bag, Board, Bowl, Tile};

/// Builder for constructing a [`GameState`] from explicit component state.
#[derive(Default)]
pub struct GameStateBuilder {
    active_player: usize,
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
    seed: Option<u64>,
    rng_state: Option<Vec<u8>>,
    discarded_tiles: usize,
}

impl GameStateBuilder {
    /// Sets the index of the active player.
    pub fn active_player(mut self, active_player: usize) -> Self {
        self.active_player = active_player;
        self
    }

    /// Sets the player boards.
    pub fn boards(mut self, boards: Vec<Board>) -> Self {
        self.boards = boards;
        self
    }

    /// Sets the factory bowls, including the centre at index zero.
    pub fn bowls(mut self, bowls: Vec<Bowl>) -> Self {
        self.bowls = bowls;
        self
    }

    /// Sets the tile bag.
    pub fn bag(mut self, bag: Bag<Tile>) -> Self {
        self.bag = bag;
        self
    }

    /// Sets the player holding the first-player token, if any.
    pub fn first_token_owner(mut self, first_token_owner: Option<usize>) -> Self {
        self.first_token_owner = first_token_owner;
        self
    }

    /// Sets the game seed.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the serialized current state of the random generator.
    pub fn set_rng_state(mut self, rng_state: Vec<u8>) -> Self {
        self.rng_state = Some(rng_state);
        self
    }

    /// Sets the number of physical tiles already returned to the discard pile.
    pub fn discarded_tiles(mut self, discarded_tiles: usize) -> Self {
        self.discarded_tiles = discarded_tiles;
        self
    }

    /// Builds a validated game state from the configured fields.
    ///
    /// Construction fails when the player count, bowl count, active-player
    /// index, first-player-token owner, or RNG state is invalid.
    pub fn build(self) -> Result<GameState, GameStateError> {
        validate_components(
            self.active_player,
            &self.boards,
            &self.bowls,
            self.first_token_owner,
        )?;
        let rng_seed = self.seed.unwrap_or_else(|| rng().random());
        let rng = match self.rng_state {
            Some(rng_state) => {
                if rng_state.iter().all(|byte| *byte == 0) {
                    return Err(GameStateError::InvalidRngState);
                }
                Xoshiro256PlusPlus(
                    bincode::deserialize(&rng_state)
                        .map_err(|_| GameStateError::InvalidRngState)?,
                )
            }
            None => Xoshiro256PlusPlus::from_seed_u64(rng_seed),
        };
        Ok(GameState {
            active_player: self.active_player,
            boards: self.boards,
            bowls: self.bowls,
            bag: self.bag,
            first_token_owner: self.first_token_owner,
            rng,
            seed: self.seed,
            discarded_tiles: self.discarded_tiles,
        })
    }
}
