//! Reinforcement-learning environment and fixed-size Azul observations.

use azul_movegen::game_move::IllegalMoveError;
use azul_movegen::{Bag, Board, Bowl, BowlChoice, GameState, Move, Row, Tile, board};
use tch::Tensor;

mod net;
pub mod ppo;

pub use ppo::{PpoConfig, PpoMetrics, PpoTrainer};

const BOARD_SIZE: usize = board::BOARD_DIMENSION;
const MAX_PLAYERS: usize = 2;
const TILE_TYPES: usize = 5;
const MAX_BOWLS: usize = 2 * MAX_PLAYERS + 2;
const CENTRE_SLOT: usize = 0;
const FACTORY_SLOT_OFFSET: usize = 1;
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
pub const OBSERVATION_SIZE: usize = MAX_BOWLS * TILE_TYPES
    + MAX_PLAYERS * BOARD_FEATURES_PER_PLAYER
    + TILE_TYPES
    + PLAYER_COUNT_FEATURES
    + FIRST_TOKEN_FEATURES
    + 2;

/// Maps canonical RL factory slots to physical factory indices; the centre is
/// always kept separately in slot zero.
#[derive(Debug, Clone)]
struct FactoryOrder {
    canonical_to_physical: Vec<usize>,
    physical_to_canonical: Vec<usize>,
}

impl FactoryOrder {
    /// Builds a deterministic order with non-empty factories before empty ones.
    fn new(factories: &[Bowl]) -> Self {
        let mut canonical_to_physical: Vec<_> = (0..factories.len()).collect();
        canonical_to_physical.sort_by_key(|&physical_index| {
            let bowl = &factories[physical_index];
            (
                bowl.get_tiles().is_empty(),
                bowl_tile_counts(bowl),
                physical_index,
            )
        });

        let mut physical_to_canonical = vec![0; factories.len()];
        for (canonical_index, &physical_index) in canonical_to_physical.iter().enumerate() {
            physical_to_canonical[physical_index] = canonical_index;
        }

        Self {
            canonical_to_physical,
            physical_to_canonical,
        }
    }

    /// Returns the physical factory index for a canonical RL slot.
    fn physical_index(&self, canonical_index: usize) -> Option<usize> {
        self.canonical_to_physical.get(canonical_index).copied()
    }

    /// Returns the canonical RL slot for a physical factory index.
    fn canonical_index(&self, physical_index: usize) -> Option<usize> {
        self.physical_to_canonical.get(physical_index).copied()
    }
}

/// Returns a complete tile-count key for deterministic factory ordering.
fn bowl_tile_counts(bowl: &Bowl) -> [u8; TILE_TYPES] {
    let mut counts = [0; TILE_TYPES];
    for &tile_type in bowl.get_tiles() {
        counts[tile_type] += 1;
    }
    counts
}

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

    let factory_order = FactoryOrder::new(gamestate.get_factory_bowls());
    let encoded_bowls = encode_bowls(
        gamestate.get_centre_bowl(),
        gamestate.get_factory_bowls(),
        &factory_order,
    );
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

/// Encodes the centre in slot zero and canonical factories in later slots.
/// The centre remains first even when it is empty.
fn encode_bowls(centre: &Bowl, factories: &[Bowl], order: &FactoryOrder) -> Tensor {
    let mut encoded_bowls = vec![0.0_f32; MAX_BOWLS * TILE_TYPES];
    for &tile_type in centre.get_tiles() {
        encoded_bowls[tile_type] += 1.0 / 20.0;
    }
    for (canonical_index, &physical_index) in order.canonical_to_physical.iter().enumerate() {
        let bowl = &factories[physical_index];
        for &tile_type in bowl.get_tiles() {
            let index = (canonical_index + FACTORY_SLOT_OFFSET) * TILE_TYPES + tile_type;
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

pub fn get_device() -> tch::Device {
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
            let factory_order = FactoryOrder::new(self.gamestate.get_factory_bowls());
            for choice in self.gamestate.get_valid_moves() {
                if let Some(action) = Self::action_for_move(&choice, &factory_order) {
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

        let factory_order = FactoryOrder::new(self.gamestate.get_factory_bowls());
        let choice = Self::map_action(action, &factory_order).ok_or(IllegalMoveError)?;
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
    fn map_action(action: usize, factory_order: &FactoryOrder) -> Option<Move> {
        if action >= ACTION_SPACE_SIZE {
            return None;
        }

        let bowl = match action / (TILE_TYPES * DESTINATIONS_PER_ACTION) {
            CENTRE_SLOT => BowlChoice::Centre,
            canonical_index => BowlChoice::Factory(
                factory_order.physical_index(canonical_index - FACTORY_SLOT_OFFSET)?,
            ),
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
    fn action_for_move(choice: &Move, factory_order: &FactoryOrder) -> Option<usize> {
        let row = match choice.row {
            Row::Floor => 0,
            Row::Wall(row) if row < BOARD_SIZE => row + 1,
            Row::Wall(_) => return None,
        };
        let bowl = match choice.bowl {
            BowlChoice::Centre => CENTRE_SLOT,
            BowlChoice::Factory(index) => factory_order
                .canonical_index(index)?
                .checked_add(FACTORY_SLOT_OFFSET)?,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with_factories(factory_bowls: Vec<Bowl>) -> GameState {
        GameState::builder()
            .boards(vec![Board::default(); 2])
            .centre_bowl(Bowl::default())
            .factory_bowls(factory_bowls)
            .bag(Bag::<Tile>::default())
            .set_seed(1)
            .build()
            .expect("test game state should be valid")
    }

    #[test]
    fn factory_order_places_empty_bowls_last_and_is_reversible() {
        let factories = vec![
            Bowl::default(),
            Bowl::from_tiles(vec![2, 2]),
            Bowl::from_tiles(vec![0]),
            Bowl::default(),
            Bowl::from_tiles(vec![1]),
        ];
        let order = FactoryOrder::new(&factories);

        let first_empty = order
            .canonical_to_physical
            .iter()
            .position(|&physical| factories[physical].get_tiles().is_empty())
            .expect("there should be an empty factory");
        assert!(
            order.canonical_to_physical[first_empty..]
                .iter()
                .all(|&physical| factories[physical].get_tiles().is_empty())
        );

        for physical_index in 0..factories.len() {
            let canonical_index = order
                .canonical_index(physical_index)
                .expect("physical factory should have a canonical slot");
            assert_eq!(order.physical_index(canonical_index), Some(physical_index));
        }
    }

    #[test]
    fn permuting_physical_factories_preserves_observation_and_action_mask() {
        let factories = vec![
            Bowl::from_tiles(vec![0, 0]),
            Bowl::default(),
            Bowl::from_tiles(vec![3]),
            Bowl::from_tiles(vec![1, 1, 1]),
            Bowl::default(),
        ];
        let permuted = vec![
            factories[3].clone(),
            factories[0].clone(),
            factories[4].clone(),
            factories[2].clone(),
            factories[1].clone(),
        ];
        let first_state = state_with_factories(factories);
        let second_state = state_with_factories(permuted);
        let first_environment = AzulEnv {
            gamestate: first_state,
            max_steps: 100,
            steps: 0,
        };
        let second_environment = AzulEnv {
            gamestate: second_state,
            max_steps: 100,
            steps: 0,
        };

        assert!(
            encode_gamestate(first_environment.get_gamestate()).allclose(
                &encode_gamestate(second_environment.get_gamestate()),
                1e-6,
                1e-6,
                false,
            )
        );
        assert!(first_environment.action_mask().allclose(
            &second_environment.action_mask(),
            1e-6,
            1e-6,
            false
        ));
    }

    #[test]
    fn canonical_action_maps_back_to_the_physical_factory() {
        let factories = vec![
            Bowl::from_tiles(vec![0]),
            Bowl::default(),
            Bowl::from_tiles(vec![3]),
            Bowl::from_tiles(vec![1]),
            Bowl::default(),
        ];
        let order = FactoryOrder::new(&factories);
        let physical_move = Move {
            bowl: BowlChoice::Factory(2),
            tile_type: 3,
            row: Row::Floor,
        };
        let action = AzulEnv::action_for_move(&physical_move, &order)
            .expect("test move should have a valid action index");

        assert_eq!(AzulEnv::map_action(action, &order), Some(physical_move));
    }

    #[test]
    fn empty_centre_remains_in_the_first_rl_slot() {
        let factories = vec![
            Bowl::default(),
            Bowl::from_tiles(vec![2]),
            Bowl::from_tiles(vec![0]),
            Bowl::default(),
            Bowl::from_tiles(vec![1]),
        ];
        let order = FactoryOrder::new(&factories);
        let empty_centre = Bowl::default();
        let encoded = encode_bowls(&empty_centre, &factories, &order);
        let centre_move = Move {
            bowl: BowlChoice::Centre,
            tile_type: 0,
            row: Row::Floor,
        };
        let action = AzulEnv::action_for_move(&centre_move, &order)
            .expect("centre action should have a valid action index");

        assert_eq!(action, CENTRE_SLOT * TILE_TYPES * DESTINATIONS_PER_ACTION);
        assert_eq!(AzulEnv::map_action(action, &order), Some(centre_move));
        assert_eq!(encoded.double_value(&[CENTRE_SLOT as i64]), 0.0);
    }
}
