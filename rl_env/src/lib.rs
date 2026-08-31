//! Reinforcement-learning environment and fixed-size Azul observations.

use azul_movegen::game_move::IllegalMoveError;
use azul_movegen::{Bag, Board, Bowl, BowlChoice, GameState, Move, Row, Tile, board};
use tch::Tensor;

mod net;
pub mod ppo;

pub use ppo::{PpoConfig, PpoMetrics, PpoTrainer};

const BOARD_SIZE: usize = board::BOARD_DIMENSION;
const MAX_PLAYERS: usize = 4;
const TILE_TYPES: usize = 5;
const MAX_BOWLS: usize = 2 * MAX_PLAYERS + 2;
const DESTINATIONS_PER_ACTION: usize = BOARD_SIZE + 1;
const PLAYER_COUNT_FEATURES: usize = MAX_PLAYERS - 1;
const FIRST_TOKEN_FEATURES: usize = MAX_PLAYERS + 1;
const SCORE_SCALE: f32 = 100.0;
const MAX_PENALTY_SPACES: f32 = 8.0;
const MAX_PENALTY_TILES: f32 = 7.0;
const REWARD_SCALE: f32 = 1.0;
const BOARD_FEATURES_PER_PLAYER: usize =
    BOARD_SIZE * BOARD_SIZE * TILE_TYPES + BOARD_SIZE * (TILE_TYPES + 1) + 2 + 1 + 3 * BOARD_SIZE;

/// Number of discrete actions in the fixed wire-bowl/tile/destination action space.
/// Wire bowl slot zero is the centre; unused slots for smaller games are masked.
pub const ACTION_SPACE_SIZE: usize = MAX_BOWLS * TILE_TYPES * DESTINATIONS_PER_ACTION;

/// Number of values in every encoded, active-player-relative observation.
/// The current layout contains 757 values, including padded four-player board
/// features and separate centre/factory bowl features.
pub const OBSERVATION_SIZE: usize = MAX_BOWLS * TILE_TYPES
    + MAX_PLAYERS * BOARD_FEATURES_PER_PLAYER
    + TILE_TYPES
    + PLAYER_COUNT_FEATURES
    + FIRST_TOKEN_FEATURES
    + 2;

/// Encodes a game state from the active player's perspective.
fn encode_gamestate(gamestate: &GameState) -> Tensor {
    let player_count = gamestate.get_boards().len();
    let active_player = gamestate.get_active_player();
    let board_order: Vec<usize> = (0..player_count)
        .map(|offset| (active_player + offset) % player_count)
        .collect();
    let ordered_boards: Vec<Board> = board_order
        .iter()
        .map(|&player| gamestate.get_boards()[player])
        .collect();

    let encoded_bowls = encode_bowls(gamestate.get_centre_bowl(), gamestate.get_factory_bowls());
    let encoded_boards = encode_boards(&ordered_boards);
    let encoded_bag = encode_bag(gamestate.get_bag());
    let mut player_count_encoding = vec![0.0_f32; PLAYER_COUNT_FEATURES];
    player_count_encoding[player_count - 2] = 1.0;

    // The first-token owner is encoded relative to the active player: 0 is None, 1 is active.
    let mut first_token_encoding = vec![0.0_f32; FIRST_TOKEN_FEATURES];
    let first_token_index = gamestate
        .get_first_token_owner()
        .map(|owner| 1 + (owner + player_count - active_player) % player_count)
        .unwrap_or(0);
    first_token_encoding[first_token_index] = 1.0;

    let round_over_encoding = [if gamestate.round_over() { 1.0_f32 } else { 0.0 }];
    let game_over_encoding = [if gamestate.is_game_over() {
        1.0_f32
    } else {
        0.0
    }];

    Tensor::cat(
        &[
            encoded_bowls,
            encoded_boards,
            encoded_bag,
            Tensor::from_slice(&player_count_encoding).to_device(get_device()),
            Tensor::from_slice(&first_token_encoding).to_device(get_device()),
            Tensor::from_slice(&round_over_encoding).to_device(get_device()),
            Tensor::from_slice(&game_over_encoding).to_device(get_device()),
        ],
        0,
    )
    .to_device(get_device())
}

/// Encodes each bowl as normalized tile-type counts, padding to four players.
fn encode_bowls(centre: &Bowl, factories: &[Bowl]) -> Tensor {
    let mut encoded_bowls = vec![0.0_f32; MAX_BOWLS * TILE_TYPES];
    for &tile_type in centre.get_tiles() {
        encoded_bowls[tile_type] += 1.0 / 20.0;
    }
    for (factory_index, bowl) in factories.iter().enumerate() {
        for &tile_type in bowl.get_tiles() {
            let index = (factory_index + 1) * TILE_TYPES + tile_type;
            encoded_bowls[index] += 1.0 / 4.0;
        }
    }
    Tensor::from_slice(&encoded_bowls).to_device(get_device())
}

fn encode_boards(boards: &[Board]) -> Tensor {
    let mut encoded_boards = Vec::<f32>::new();
    for board in boards {
        // Each placed cell is represented by a five-way tile-type one-hot vector.
        for row in board.get_placed() {
            for tile in row {
                for tile_type in 0..TILE_TYPES {
                    encoded_boards.push(if *tile == Some(tile_type) { 1.0 } else { 0.0 });
                }
            }
        }

        // Each pattern line gets its tile type and fullness, preserving row identity.
        for (row_index, row) in board.get_holds().iter().enumerate() {
            let tile_type = row.iter().flatten().next().copied();
            for candidate in 0..TILE_TYPES {
                encoded_boards.push(if tile_type == Some(candidate) {
                    1.0
                } else {
                    0.0
                });
            }
            let fullness =
                row.iter().filter(|tile| tile.is_some()).count() as f32 / (row_index + 1) as f32;
            encoded_boards.push(fullness);
        }

        // Keep both scoring-space occupancy and physical penalty-tile occupancy.
        encoded_boards.push((board.get_penalties() as f32 / MAX_PENALTY_SPACES).min(1.0));
        encoded_boards.push((board.get_penalty_tiles() as f32 / MAX_PENALTY_TILES).min(1.0));

        // Each board gets one score difference relative to the active player's board at index zero.
        let active_score = boards[0].get_score() as f32;
        encoded_boards.push((board.get_score() as f32 - active_score) / SCORE_SCALE);

        // Bonuses are five row, five column, and five tile-type flags (15 values total).
        let bonuses = board.get_bonuses();
        encoded_boards.extend(
            bonuses
                .rows
                .iter()
                .chain(bonuses.columns.iter())
                .chain(bonuses.tile_types.iter())
                .map(|&bonus| if bonus { 1.0 } else { 0.0 }),
        );
    }
    encoded_boards.extend(std::iter::repeat_n(
        0.0,
        (MAX_PLAYERS - boards.len()) * BOARD_FEATURES_PER_PLAYER,
    ));

    Tensor::from_slice(&encoded_boards).to_device(get_device())
}

fn encode_bag(bag: &Bag<Tile>) -> Tensor {
    let mut encoded_bag = vec![0.0_f32; TILE_TYPES];
    for &tile_type in bag.items() {
        encoded_bag[tile_type] += 1.0 / 20.0;
    }
    Tensor::from_slice(&encoded_bag).to_device(get_device())
}

/// Computes the active player's score delta relative to the strongest opponent.
fn calculate_reward(score_deltas: &[i64], active_player: usize) -> f32 {
    // Max score delta of opponents
    let max_opp_score = score_deltas
        .iter()
        .enumerate()
        .fold(i64::MIN, |acc, (player, &delta)| {
            if player == active_player {
                return acc;
            }
            delta.max(acc)
        });
    (score_deltas[active_player] - max_opp_score) as f32 * REWARD_SCALE
}

fn get_device() -> tch::Device {
    tch::Device::cuda_if_available()
}

/// Two-player Azul environment with a fixed action and observation interface.
pub struct AzulEnv {
    gamestate: GameState,
    max_steps: usize,
    steps: usize,
}

/// Result of applying one environment action.
pub struct StepResult {
    /// Observation after the action, from the new active player's perspective.
    pub next_state: Tensor,
    /// Reward for the player who took the action.
    pub reward: f32,
    /// Whether the game reached its terminal state.
    pub terminated: bool,
    /// Whether the environment stopped because its step limit was reached.
    pub truncated: bool,
}

impl AzulEnv {
    /// Creates a two-player environment with a seeded, playable first round.
    pub fn new(seed: u64, max_steps: usize) -> Self {
        let mut environment = AzulEnv {
            gamestate: GameState::new(2, seed).expect("two-player game state must be valid"),
            max_steps,
            steps: 0,
        };
        environment.gamestate.setup_next_round();
        environment
    }

    /// Resets the environment with an explicit seed and starts a playable round.
    pub fn seeded_reset(&mut self, seed: u64, max_steps: usize) -> Tensor {
        self.gamestate = GameState::new(2, seed).expect("two-player game state must be valid");
        self.steps = 0;
        self.max_steps = max_steps;
        self.gamestate.setup_next_round();
        encode_gamestate(&self.gamestate)
    }

    /// Resets the environment with a random seed and starts a playable round.
    pub fn reset(&mut self, max_steps: usize) -> Tensor {
        self.gamestate =
            GameState::new(2, rand::random()).expect("two-player game state must be valid");
        self.steps = 0;
        self.max_steps = max_steps;
        self.gamestate.setup_next_round();
        encode_gamestate(&self.gamestate)
    }

    /// Returns a fixed-size mask containing one for every currently legal action.
    pub fn action_mask(&self) -> Tensor {
        let mut mask = vec![0.0; ACTION_SPACE_SIZE];
        if !self.gamestate.is_game_over() && self.steps < self.max_steps {
            for choice in self.gamestate.get_valid_moves() {
                if let Some(action) = Self::action_for_move(&choice) {
                    mask[action] = 1.0;
                }
            }
        }
        Tensor::from_slice(&mask).to_device(get_device())
    }

    /// Applies an action, rejecting out-of-range and illegal moves without panicking.
    pub fn step(&mut self, action: usize) -> Result<StepResult, IllegalMoveError> {
        if self.gamestate.is_game_over() || self.steps >= self.max_steps {
            return Err(IllegalMoveError);
        }

        let choice = Self::map_action(action).ok_or(IllegalMoveError)?;
        let acting_player = self.gamestate.get_active_player();
        let before_scores: Vec<i64> = self
            .gamestate
            .get_boards()
            .iter()
            .map(|board| board.get_score() as i64)
            .collect();

        self.gamestate.make_move(&choice)?;
        self.steps += 1;

        let round_over = self.gamestate.round_over();
        if round_over {
            self.gamestate.setup_next_round();
        }

        let score_deltas: Vec<i64> = if round_over {
            self.gamestate
                .get_boards()
                .iter()
                .map(|board| board.get_score() as i64)
                .zip(before_scores)
                .map(|(after, before)| after - before)
                .collect()
        } else {
            vec![0; self.gamestate.get_boards().len()]
        };

        let terminated = self.gamestate.is_game_over();
        let truncated = !terminated && self.steps >= self.max_steps;

        Ok(StepResult {
            next_state: encode_gamestate(&self.gamestate),
            reward: calculate_reward(&score_deltas, acting_player),
            terminated,
            truncated,
        })
    }

    /// Converts a fixed action index into a move when its components are in range.
    fn map_action(action: usize) -> Option<Move> {
        if action >= ACTION_SPACE_SIZE {
            return None;
        }

        let bowl = match action / (TILE_TYPES * DESTINATIONS_PER_ACTION) {
            0 => BowlChoice::Centre,
            bowl => BowlChoice::Factory(bowl - 1),
        };
        let tile_type = (action / DESTINATIONS_PER_ACTION) % TILE_TYPES;
        let row = match action % DESTINATIONS_PER_ACTION {
            0 => Row::Floor,
            n => Row::Wall(n - 1),
        };
        Some(Move {
            bowl,
            tile_type,
            row,
        })
    }

    /// Converts a move to its canonical fixed action index.
    fn action_for_move(choice: &Move) -> Option<usize> {
        let row = match choice.row {
            Row::Floor => 0,
            Row::Wall(row) if row < BOARD_SIZE => row + 1,
            Row::Wall(_) => return None,
        };
        let bowl = match choice.bowl {
            BowlChoice::Centre => 0,
            BowlChoice::Factory(index) => index.checked_add(1)?,
        };
        if bowl >= MAX_BOWLS || choice.tile_type >= TILE_TYPES {
            return None;
        }
        Some((bowl * TILE_TYPES + choice.tile_type) * DESTINATIONS_PER_ACTION + row)
    }

    /// Returns the current game state used by the environment.
    pub fn get_gamestate(&self) -> &GameState {
        &self.gamestate
    }
}
