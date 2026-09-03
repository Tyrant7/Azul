//! Reinforcement-learning environment and fixed-size Azul observations.

use azul_movegen::game_move::IllegalMoveError;
use azul_movegen::{
    Bag, Board, Bowl, BowlChoice, GameState, Move, Row, TOTAL_TILE_COUNT, Tile, board,
};
use tch::Tensor;

mod metrics;
mod net;
mod policy;
pub mod ppo;

pub use metrics::PpoMetrics;
pub use policy::ActorPolicy;
pub use ppo::{PpoConfig, PpoTrainer};

const BOARD_SIZE: usize = board::BOARD_DIMENSION;
const PLAYER_COUNT: usize = 2;
const TILE_TYPES: usize = 5;
const FACTORY_BOWLS: usize = 2 * PLAYER_COUNT + 1;
const BOWL_SLOTS: usize = FACTORY_BOWLS + 1;
const CENTRE_SLOT: usize = 0;
const FACTORY_SLOT_OFFSET: usize = 1;
const DESTINATIONS_PER_ACTION: usize = BOARD_SIZE + 1;
const FIRST_TOKEN_FEATURES: usize = PLAYER_COUNT + 1;
const SCORE_SCALE: f32 = 100.0;
const MAX_PENALTY_SPACES: f32 = 8.0;
const MAX_PENALTY_TILES: f32 = 7.0;
const REWARD_SCALE: f32 = 0.1;
const BOARD_FEATURES_PER_PLAYER: usize =
    BOARD_SIZE * BOARD_SIZE * TILE_TYPES + BOARD_SIZE * (TILE_TYPES + 1) + 2 + 1 + 3 * BOARD_SIZE;
const ACTION_SOURCE_FEATURES: usize = BOWL_SLOTS;
const ACTION_DESTINATION_FEATURES: usize = DESTINATIONS_PER_ACTION;
const ACTION_SPEC_FEATURES: usize =
    ACTION_SOURCE_FEATURES + TILE_TYPES + ACTION_DESTINATION_FEATURES;
const DYNAMIC_ACTION_FEATURES: usize = 8;
const FACTORY_BOWL_CAPACITY: f32 = 4.0;
const MAX_TILE_TYPE_COUNT: f32 = 20.0;
const MAX_PENALTY_SCORE: f32 = 12.0;

/// Number of discrete actions in the fixed wire-bowl/tile/destination action space.
/// Wire bowl slot zero is the centre, followed by the five two-player factories.
pub const ACTION_SPACE_SIZE: usize = BOWL_SLOTS * TILE_TYPES * DESTINATIONS_PER_ACTION;

/// Number of one-hot values used to describe a candidate action to the actor.
pub const ACTION_FEATURE_SIZE: usize = ACTION_SPEC_FEATURES + DYNAMIC_ACTION_FEATURES;

/// Number of values in every encoded, active-player-relative observation.
pub const OBSERVATION_SIZE: usize = BOWL_SLOTS * TILE_TYPES
    + PLAYER_COUNT * BOARD_FEATURES_PER_PLAYER
    + TILE_TYPES
    + FIRST_TOKEN_FEATURES
    + 2;

/// Encodes the static source, tile-type, and destination features of an action.
/// State-dependent feature slots are left at zero by this state-independent helper.
pub fn encode_action_features(action: usize) -> Option<[f32; ACTION_FEATURE_SIZE]> {
    if action >= ACTION_SPACE_SIZE {
        return None;
    }

    let source = action / (TILE_TYPES * DESTINATIONS_PER_ACTION);
    let tile_type = (action / DESTINATIONS_PER_ACTION) % TILE_TYPES;
    let destination = action % DESTINATIONS_PER_ACTION;
    let mut features = [0.0; ACTION_FEATURE_SIZE];
    features[source] = 1.0;
    features[ACTION_SOURCE_FEATURES + tile_type] = 1.0;
    features[ACTION_SPEC_FEATURES - ACTION_DESTINATION_FEATURES + destination] = 1.0;
    Some(features)
}

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
pub fn encode_state(gamestate: &GameState) -> Tensor {
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
            Tensor::from_slice(&first_token_encoding).to_device(get_device()),
            Tensor::from_slice(&round_over_encoding).to_device(get_device()),
            Tensor::from_slice(&game_over_encoding).to_device(get_device()),
        ],
        0,
    )
    .to_device(get_device())
}

/// Returns legal moves with the same dynamic features used during training.
pub fn legal_move_features(gamestate: &GameState) -> Vec<(Move, [f32; ACTION_FEATURE_SIZE])> {
    let factory_order = FactoryOrder::new(gamestate.get_factory_bowls());
    gamestate
        .get_valid_moves()
        .into_iter()
        .filter_map(|choice| {
            let action = AzulEnv::action_for_move(&choice, &factory_order)?;
            let features = action_features_for_gamestate(gamestate, action, &factory_order)?;
            Some((choice, features))
        })
        .collect()
}

/// Encodes the centre in slot zero and canonical factories in later slots.
/// The centre remains first even when it is empty.
fn encode_bowls(centre: &Bowl, factories: &[Bowl], order: &FactoryOrder) -> Tensor {
    let mut encoded_bowls = vec![0.0_f32; BOWL_SLOTS * TILE_TYPES];
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
        (PLAYER_COUNT - boards.len()) * BOARD_FEATURES_PER_PLAYER,
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

/// Returns the penalty points represented by a board's scoring-space occupancy.
fn penalty_points(penalties: usize) -> usize {
    [1, 1, 2, 2, 2, 3, 3].iter().take(penalties).sum()
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
    /// Round-level diagnostics when this action resolved a round.
    pub round_diagnostics: Option<RoundDiagnostics>,
}

/// Aggregate board statistics produced when a round is resolved.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoundDiagnostics {
    /// Scoring-space penalty tiles for each player this round.
    pub penalties: [usize; PLAYER_COUNT],
    /// Bonus points earned by each player this round.
    pub bonus_points: [usize; PLAYER_COUNT],
    /// Newly completed wall rows for each player.
    pub rows_filled: [usize; PLAYER_COUNT],
    /// Newly completed wall columns for each player.
    pub columns_filled: [usize; PLAYER_COUNT],
    /// Newly collected five-of-a-kind bonuses for each player.
    pub tile_bonuses: [usize; PLAYER_COUNT],
}

#[derive(Clone, Copy)]
struct BoardRoundStats {
    rows_filled: usize,
    columns_filled: usize,
    bonuses: [bool; TILE_TYPES * 3],
}

fn board_round_stats(board: &Board) -> BoardRoundStats {
    let placed = board.get_placed();
    let columns_filled = (0..BOARD_SIZE)
        .filter(|&column| placed.iter().all(|row| row[column].is_some()))
        .count();
    let bonuses = board.get_bonuses();
    let mut bonus_flags = [false; TILE_TYPES * 3];
    bonus_flags[..TILE_TYPES].copy_from_slice(&bonuses.rows);
    bonus_flags[TILE_TYPES..2 * TILE_TYPES].copy_from_slice(&bonuses.columns);
    bonus_flags[2 * TILE_TYPES..].copy_from_slice(&bonuses.tile_types);
    BoardRoundStats {
        rows_filled: board.count_horizontal_lines(),
        columns_filled,
        bonuses: bonus_flags,
    }
}

fn round_diagnostics(
    before: &[BoardRoundStats],
    after: &[BoardRoundStats],
    penalties: &[usize],
) -> RoundDiagnostics {
    let mut diagnostics = RoundDiagnostics::default();
    for (player, (before, after)) in before.iter().zip(after).enumerate() {
        diagnostics.penalties[player] = penalties.get(player).copied().unwrap_or_default();
        diagnostics.rows_filled[player] = after.rows_filled.saturating_sub(before.rows_filled);
        diagnostics.columns_filled[player] =
            after.columns_filled.saturating_sub(before.columns_filled);
        for (index, (was_collected, is_collected)) in
            before.bonuses.iter().zip(after.bonuses).enumerate()
        {
            if !was_collected && is_collected {
                diagnostics.bonus_points[player] += match index / TILE_TYPES {
                    0 => 2,
                    1 => 7,
                    _ => 10,
                };
            }
        }
        diagnostics.tile_bonuses[player] += after.bonuses[2 * TILE_TYPES..]
            .iter()
            .zip(&before.bonuses[2 * TILE_TYPES..])
            .filter(|(was_collected, is_collected)| !**was_collected && **is_collected)
            .count();
    }
    diagnostics
}

impl AzulEnv {
    /// Creates a two-player environment with a seeded, playable first round.
    pub fn new(seed: u64, max_steps: usize) -> Self {
        let mut environment = AzulEnv {
            gamestate: GameState::new(PLAYER_COUNT, seed)
                .expect("two-player game state must be valid"),
            max_steps,
            steps: 0,
        };
        environment.gamestate.setup_next_round();
        environment
    }

    /// Resets the environment with an explicit seed and starts a playable round.
    pub fn seeded_reset(&mut self, seed: u64, max_steps: usize) -> Tensor {
        self.gamestate =
            GameState::new(PLAYER_COUNT, seed).expect("two-player game state must be valid");
        self.steps = 0;
        self.max_steps = max_steps;
        self.gamestate.setup_next_round();
        encode_state(&self.gamestate)
    }

    /// Resets the environment with a random seed and starts a playable round.
    pub fn reset(&mut self, max_steps: usize) -> Tensor {
        self.gamestate = GameState::new(PLAYER_COUNT, rand::random())
            .expect("two-player game state must be valid");
        self.steps = 0;
        self.max_steps = max_steps;
        self.gamestate.setup_next_round();
        encode_state(&self.gamestate)
    }

    /// Returns a fixed-size mask containing one for every currently legal action.
    pub fn action_mask(&self) -> Tensor {
        let mut mask = vec![0.0; ACTION_SPACE_SIZE];
        for action in self.legal_actions() {
            mask[action] = 1.0;
        }
        Tensor::from_slice(&mask).to_device(get_device())
    }

    /// Returns sorted, unique canonical IDs for the currently legal actions.
    pub fn legal_actions(&self) -> Vec<usize> {
        if self.gamestate.is_game_over() || self.steps >= self.max_steps {
            return Vec::new();
        }

        let factory_order = FactoryOrder::new(self.gamestate.get_factory_bowls());
        let mut actions: Vec<_> = self
            .gamestate
            .get_valid_moves()
            .iter()
            .filter_map(|choice| Self::action_for_move(choice, &factory_order))
            .collect();
        actions.sort_unstable();
        assert!(
            actions.windows(2).all(|pair| pair[0] < pair[1]),
            "legal move generation produced duplicate canonical action IDs"
        );
        actions
    }

    /// Returns each legal action with static and pre-action dynamic features.
    pub fn legal_action_features(&self) -> Vec<(usize, [f32; ACTION_FEATURE_SIZE])> {
        legal_move_features(&self.gamestate)
            .into_iter()
            .filter_map(|(choice, features)| {
                let order = FactoryOrder::new(self.gamestate.get_factory_bowls());
                Some((Self::action_for_move(&choice, &order)?, features))
            })
            .collect()
    }
}

fn action_features_for_gamestate(
    gamestate: &GameState,
    action: usize,
    factory_order: &FactoryOrder,
) -> Option<[f32; ACTION_FEATURE_SIZE]> {
    let choice = AzulEnv::map_action(action, factory_order)?;
    let mut features = encode_action_features(action)?;
    let active_board = gamestate.get_boards().get(gamestate.get_active_player())?;
    let (source, source_capacity, is_centre) = match choice.bowl {
        BowlChoice::Centre => (gamestate.get_centre_bowl(), TOTAL_TILE_COUNT as f32, true),
        BowlChoice::Factory(index) => (
            gamestate.get_factory_bowls().get(index)?,
            FACTORY_BOWL_CAPACITY,
            false,
        ),
    };
    let selected_tile_count = source
        .get_tiles()
        .iter()
        .filter(|&&tile_type| tile_type == choice.tile_type)
        .count();
    let remaining_source_tile_count = source.get_tiles().len() - selected_tile_count;

    let (destination_fullness, destination_remaining, overflow, completes_line) = match choice.row {
        Row::Floor => (0.0, 0.0, selected_tile_count, 0.0),
        Row::Wall(row_index) => {
            let row = active_board.get_holds().get(row_index)?;
            let capacity = row_index + 1;
            let occupied = row.iter().filter(|tile| tile.is_some()).count();
            let available = capacity.saturating_sub(occupied);
            (
                occupied as f32 / capacity as f32,
                available as f32 / capacity as f32,
                selected_tile_count.saturating_sub(available),
                if selected_tile_count >= available {
                    1.0
                } else {
                    0.0
                },
            )
        }
    };

    let token_penalty = if is_centre && gamestate.get_first_token_owner().is_none() {
        1
    } else {
        0
    };
    let added_penalties = overflow + token_penalty;
    let expected_penalty_points = penalty_points(active_board.get_penalties() + added_penalties)
        - penalty_points(active_board.get_penalties());

    let dynamic_offset = ACTION_SPEC_FEATURES;
    features[dynamic_offset] = remaining_source_tile_count as f32 / source_capacity;
    features[dynamic_offset + 1] = selected_tile_count as f32 / MAX_TILE_TYPE_COUNT;
    features[dynamic_offset + 2] = if is_centre { 1.0 } else { 0.0 };
    features[dynamic_offset + 3] = destination_fullness;
    features[dynamic_offset + 4] = destination_remaining;
    features[dynamic_offset + 5] = (overflow as f32 / MAX_PENALTY_TILES).min(1.0);
    features[dynamic_offset + 6] = expected_penalty_points as f32 / MAX_PENALTY_SCORE;
    features[dynamic_offset + 7] = completes_line;
    Some(features)
}

impl AzulEnv {
    /// Applies an action, rejecting out-of-range and illegal moves without panicking.
    pub fn step(&mut self, action: usize) -> Result<StepResult, IllegalMoveError> {
        if self.gamestate.is_game_over() || self.steps >= self.max_steps {
            return Err(IllegalMoveError);
        }

        let factory_order = FactoryOrder::new(self.gamestate.get_factory_bowls());
        let choice = Self::map_action(action, &factory_order).ok_or(IllegalMoveError)?;
        let acting_player = self.gamestate.get_active_player();
        let before_round_stats: Vec<_> = self
            .gamestate
            .get_boards()
            .iter()
            .map(board_round_stats)
            .collect();
        let before_scores: Vec<i64> = self
            .gamestate
            .get_boards()
            .iter()
            .map(|board| board.get_score() as i64)
            .collect();

        self.gamestate.make_move(&choice)?;
        self.steps += 1;

        let round_over = self.gamestate.round_over();
        let round_penalties = if round_over {
            let mut penalties = [0; PLAYER_COUNT];
            for (player, board) in self.gamestate.get_boards().iter().enumerate() {
                penalties[player] = board.get_penalties();
            }
            penalties
        } else {
            [0; PLAYER_COUNT]
        };
        if round_over {
            self.gamestate.setup_next_round();
        }

        let round_diagnostics = if round_over {
            let after_round_stats: Vec<_> = self
                .gamestate
                .get_boards()
                .iter()
                .map(board_round_stats)
                .collect();
            Some(round_diagnostics(
                &before_round_stats,
                &after_round_stats,
                &round_penalties,
            ))
        } else {
            None
        };

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
            next_state: encode_state(&self.gamestate),
            reward: calculate_reward(&score_deltas, acting_player),
            terminated,
            truncated,
            round_diagnostics,
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
        if bowl >= BOWL_SLOTS || choice.tile_type >= TILE_TYPES {
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

        assert!(encode_state(first_environment.get_gamestate()).allclose(
            &encode_state(second_environment.get_gamestate()),
            1e-6,
            1e-6,
            false,
        ));
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

    #[test]
    fn two_player_interface_dimensions_are_consistent() {
        let environment = AzulEnv::new(1, 100);

        assert_eq!(FACTORY_BOWLS, 5);
        assert_eq!(BOWL_SLOTS, 6);
        assert_eq!(ACTION_SPACE_SIZE, 180);
        assert_eq!(OBSERVATION_SIZE, 386);
        assert_eq!(environment.action_mask().numel(), ACTION_SPACE_SIZE);
        assert_eq!(
            encode_state(environment.get_gamestate()).numel(),
            OBSERVATION_SIZE
        );
    }

    #[test]
    fn action_features_encode_each_move_component_once() {
        let action = ((2 * TILE_TYPES + 4) * DESTINATIONS_PER_ACTION) + 5;
        let features = encode_action_features(action).expect("action should be in range");

        assert_eq!(features.iter().filter(|&&value| value == 1.0).count(), 3);
        assert_eq!(features[2], 1.0);
        assert_eq!(features[ACTION_SOURCE_FEATURES + 4], 1.0);
        assert_eq!(features[ACTION_SOURCE_FEATURES + TILE_TYPES + 5], 1.0);
        assert!(encode_action_features(ACTION_SPACE_SIZE).is_none());
    }

    #[test]
    fn legal_action_features_encode_pre_move_consequences() {
        let mut holds = [[None; BOARD_SIZE]; BOARD_SIZE];
        holds[1][0] = Some(0);
        let board = Board::builder().holds(holds).build();
        let gamestate = GameState::builder()
            .boards(vec![board, Board::default()])
            .centre_bowl(Bowl::default())
            .factory_bowls(vec![
                Bowl::from_tiles(vec![0, 0, 1]),
                Bowl::default(),
                Bowl::default(),
                Bowl::default(),
                Bowl::default(),
            ])
            .bag(Bag::<Tile>::default())
            .set_seed(1)
            .build()
            .expect("test game state should be valid");
        let environment = AzulEnv {
            gamestate,
            max_steps: 100,
            steps: 0,
        };
        let factory_order = FactoryOrder::new(environment.gamestate.get_factory_bowls());
        let choice = Move {
            bowl: BowlChoice::Factory(0),
            tile_type: 0,
            row: Row::Wall(1),
        };
        let action = AzulEnv::action_for_move(&choice, &factory_order)
            .expect("test move should have a valid action index");
        let features =
            action_features_for_gamestate(&environment.gamestate, action, &factory_order)
                .expect("test move should be legal");
        let dynamic_offset = ACTION_SPEC_FEATURES;

        assert_eq!(features[dynamic_offset], 1.0 / FACTORY_BOWL_CAPACITY);
        assert_eq!(features[dynamic_offset + 1], 2.0 / MAX_TILE_TYPE_COUNT);
        assert_eq!(features[dynamic_offset + 2], 0.0);
        assert_eq!(features[dynamic_offset + 3], 0.5);
        assert_eq!(features[dynamic_offset + 4], 0.5);
        assert_eq!(features[dynamic_offset + 5], 1.0 / MAX_PENALTY_TILES);
        assert_eq!(features[dynamic_offset + 6], 1.0 / MAX_PENALTY_SCORE);
        assert_eq!(features[dynamic_offset + 7], 1.0);
    }

    #[test]
    fn legal_action_ids_are_sorted_and_match_the_mask() {
        let environment = AzulEnv::new(1, 100);
        let legal_actions = environment.legal_actions();
        let action_mask = environment.action_mask();

        assert!(!legal_actions.is_empty());
        assert!(legal_actions.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            legal_actions
                .iter()
                .all(|&action| action_mask.double_value(&[action as i64]) == 1.0)
        );
        assert_eq!(
            legal_actions.len(),
            action_mask.eq(1.0).sum(tch::Kind::Int64).int64_value(&[]) as usize
        );
    }
}
