use rand::{SeedableRng, rngs::SmallRng};

use super::{GameState, GameStateError, validate_components};
use crate::{Bag, Board, Bowl, Tile};

/// Builder for constructing a [`GameState`] from explicit component state.
#[derive(Default)]
pub struct GameStateBuilder {
    active_player: usize,
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
    seed: u64,
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
        self.seed = seed;
        self
    }

    /// Builds a validated game state from the configured fields.
    ///
    /// Construction fails when the player count, bowl count, active-player
    /// index, or first-player-token owner is invalid.
    pub fn build(self) -> Result<GameState, GameStateError> {
        validate_components(
            self.active_player,
            &self.boards,
            &self.bowls,
            self.first_token_owner,
        )?;
        Ok(GameState {
            active_player: self.active_player,
            boards: self.boards,
            bowls: self.bowls,
            bag: self.bag,
            first_token_owner: self.first_token_owner,
            rng: SmallRng::seed_from_u64(self.seed),
        })
    }
}
