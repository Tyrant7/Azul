use std::{
    char::MAX,
    ops::{Add, Div},
};

use rand::seq::IndexedRandom;

use azul_movegen::{Bag, Board, Bowl, GameState, Move, board};
use tch::Tensor;

const BOARD_SIZE: usize = board::BOARD_DIMENSION;
const MAX_PLAYERS: usize = 4;
const TILE_TYPES: usize = 5;
const REWARD_SCALE: f32 = 1.0;

fn encode_gamestate(gamestate: &GameState) -> Tensor {
    let encoded_bowls = encode_bowls(gamestate.get_bowls());
    let encoded_boards = encode_boards(&gamestate.get_boards());
    let encoded_bag = encode_bag(gamestate.get_bag());
    let player_count_encoding = Tensor::zeros(MAX_PLAYERS, (tch::Kind::Float, get_device()));
    player_count_encoding
        .get(gamestate.get_player_count() as i64)
        .add(1.);
    Tensor::cat(
        &[
            encoded_bowls,
            encoded_boards,
            encoded_bag,
            player_count_encoding,
        ],
        0,
    )
}

fn encode_bowls(bowls: &Vec<Bowl>) -> Tensor {
    // Max bowls including floor, with 1 space normalized per tile-type per bowl
    let size = (2 * MAX_PLAYERS + 2) * TILE_TYPES;
    let encoded_bowls = Tensor::zeros(size, (tch::Kind::Float, get_device()));
    for (bowl_index, bowl) in bowls.iter().enumerate() {
        for tile_type in bowl.get_tiles().iter() {
            let index = bowl_index * TILE_TYPES + tile_type;
            encoded_bowls.get(index as i64).add(1.);
        }
    }

    // Normalize by tiles by bowl
    encoded_bowls.i(TILE_TYPES..).div(4.);

    // Centre can hold up to 20 tiles of one type
    encoded_bowls.i(0..TILE_TYPES).div(20.);

    encoded_bowls
}

fn encode_boards(boards: &[Board]) -> Tensor {
    // Board occupancy map -> 5 rows and 5 cols, with 5 tiletypes
    let occupancy_size = BOARD_SIZE * BOARD_SIZE * TILE_TYPES;
    // Holds -> one-hot on tile type plus a normalized fullness
    let holds_size = TILE_TYPES + 1;
    // Penalties -> single normalized value
    let penalties_size = 1;
    // Score -> single scaled relative-scoring value between this board and each other board
    let score_size = MAX_PLAYERS;
    // Bonus types -> 5 rows, 5 cols, and 5 tiletypes
    let bonus_types_size = BOARD_SIZE * BOARD_SIZE * TILE_TYPES;

    let encoded_boards = Vec::<Tensor>::with_capacity(boards.len());
    for board in boards {
        // Occupancy
        let encoded_occupancy = Tensor::zeros(occupancy_size, (tch::Kind::Float, get_device()));
        for (row_index, row) in board.get_placed().iter().enumerate() {
            for (col_index, tile) in row.iter().enumerate() {
                if let Some(tile) = tile {
                    let index = row_index * BOARD_SIZE * TILE_TYPES + col_index * TILE_TYPES + tile;
                    encoded_boards.get(index as i64).add(1.);
                }
            }
        }

        // Holds
        let encoded_holds = Tensor::zeros(holds_size, (tch::Kind::Float, get_device()));
        for (row_index, row) in board.get_holds().iter().enumerate() {
            for tile in row.iter().flatten() {
                let index = row_index * TILE_TYPES + tile;
                encoded_holds.get(index as i64).add(1.);
            }
        }

        // Penalties
        let encoded_penalties = Tensor::zeros(penalties_size, (tch::Kind::Float, get_device()));
        let penalty = board.get_penalty();
        encoded_penalties.get(0).add(penalty);

        // Scores
        // This might have a bit of an issue since the score is relative but unordered so different boards will have the zero relative score encoded in a different position based on their index in the list of boards
        let encoded_scores = Tensor::zeros(score_size, (tch::Kind::Float, get_device()));
        let score = board.get_score();
        for (i, other_board) in boards.iter().enumerate() {
            let other_score = other_board.get_score();
            let relative_score = score as f32 - other_score as f32;
            encoded_scores.get(i as i64).add(relative_score);
        }

        // Bonuses
        let bonuses = board.get_bonuses();
        let encoded_bonuses = Tensor::from_slice(
            &bonuses
                .rows
                .iter()
                .chain(bonuses.columns.iter())
                .chain(bonuses.tile_types.iter())
                .map(|&b| if b { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>(),
        );

        encoded_boards.push(Tensor::cat(
            &[
                encoded_occupancy,
                encoded_holds,
                encoded_penalties,
                encoded_scores,
                encoded_bonuses,
            ],
            0,
        ));
    }

    Tensor::from_slice(encoded_boards.collect::<Vec<Tensor>>()).to_device(get_device())
}

fn encode_bag(bag: &Bag) -> Tensor {
    let size = TILE_TYPES;
    let encoded_bag = Tensor::zeros(size, (tch::Kind::Float, get_device()));
    for tile_type in bag.items().iter() {
        let index = tile_type * TILE_TYPES;
        encoded_bag.get(index as i64).add(1.);
    }

    // Normalize by max tiles in bag
    encoded_bag.div(20.);

    encoded_bag
}

fn calculate_reward(score_deltas: Vec<usize>, active_player: usize) -> f32 {
    // Max score delta of opponents
    let max_opp_score = score_deltas
        .iter()
        .enumerate()
        .fold(0usize, |acc, (player, &delta)| {
            if player == active_player {
                return acc;
            }
            if delta > acc { delta } else { acc }
        });
    (score_deltas[active_player] as f32 - max_opp_score as f32) * REWARD_SCALE
}

fn get_device() -> tch::Device {
    tch::Device::cuda_if_available()
}

pub struct AzulEnv {
    gamestate: GameState,
    max_steps: usize,
    steps: usize,
}

pub struct StepResult {
    pub next_state: Tensor,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

impl AzulEnv {
    pub fn new(seed: u64, max_steps: usize) -> Self {
        let gamestate = GameState::new(2, seed).expect("two-player game state must be valid");
        AzulEnv {
            gamestate,
            max_steps,
            steps: 0,
        }
    }

    pub fn seeded_reset(&mut self, seed: u64, max_steps: usize) -> Tensor {
        self.gamestate = GameState::new(2, seed).expect("two-player game state must be valid");
        self.steps = 0;
        self.max_steps = max_steps;
        encode_gamestate(&self.gamestate)
    }

    pub fn reset(&mut self, max_steps: usize) -> Tensor {
        self.gamestate =
            GameState::new(2, rand::random()).expect("two-player game state must be valid");
        self.steps = 0;
        encode_gamestate(&self.gamestate)
    }

    pub fn step(&mut self, action: usize) -> StepResult {
        let choice = Self::map_action(action);
        self.gamestate
            .make_move(&choice)
            .expect("valid move should be applied");
        self.steps += 1;
        let terminated = self.gamestate.is_game_over();
        let truncated = !terminated && self.steps >= self.max_steps;

        let before_scores = self
            .gamestate
            .get_boards()
            .iter()
            .map(|b| b.get_score())
            .collect();

        let score_deltas = if self.gamestate.round_over() {
            self.gamestate.setup_next_round();

            let after_scores = self
                .gamestate
                .get_boards()
                .iter()
                .map(|b| b.get_score())
                .collect();

            after_scores
                .iter()
                .zip(before_scores.iter())
                .map(|(after, before)| after - before)
                .collect()
        } else {
            vec![0; MAX_PLAYERS]
        };

        StepResult {
            next_state: encode_gamestate(&self.gamestate),
            reward: calculate_reward(score_deltas, self.gamestate.get_active_player()),
            terminated,
            truncated,
        }
    }

    fn map_action(action: usize) -> Move {
        // bowl in [0, 9], in a 4 player game
        // tile_type in [0, 4]
        // row in [0, 5] (0 = floor, 1-5 = wall)
        // so action in [0, 9*5*6) = [0, 270)
        let bowl = action / (TILE_TYPES * 6);
        let tile_type = (action / 6) % TILE_TYPES;
        let row = match action % 6 {
            0 => azul_movegen::row::Row::Floor,
            n => azul_movegen::row::Row::Wall(n - 1),
        };
        Move {
            bowl,
            tile_type,
            row,
        }
    }

    pub fn get_gamestate(&self) -> &GameState {
        &self.gamestate
    }
}

pub struct Transition {
    pub state: Tensor,
    pub action: usize,
    pub reward: f32,
    pub next_state: Tensor,
    pub terminated: bool,
    pub truncated: bool,
}

impl Transition {
    pub fn new(
        state: &Tensor,
        action: usize,
        reward: f32,
        next_state: &Tensor,
        terminated: bool,
        truncated: bool,
    ) -> Self {
        Transition {
            state: state.shallow_clone(),
            action,
            reward,
            next_state: next_state.shallow_clone(),
            terminated,
            truncated,
        }
    }
}

pub struct ReplayBuffer {
    capacity: usize,
    insertions: usize,
    transitions: Vec<Transition>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            insertions: 0,
            transitions: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.insertions < self.capacity {
            self.transitions.push(transition);
        } else {
            // Replace the oldest transition when buffer is full
            self.transitions[self.insertions] = transition;
        }
        self.insertions += 1;
        self.insertions %= self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        self.transitions
            .sample(&mut rand::rng(), batch_size)
            .collect()
    }

    pub fn sample_tensors(&self, buffer: &mut SampleBuffer) {
        let batch_size = buffer.states.size()[0] as usize;
        let batch = self.sample(batch_size);

        // Stack is one bulk GPU op -> much faster than per-element copy
        let states: Vec<_> = batch.iter().map(|t| t.state.shallow_clone()).collect();
        let next_states: Vec<_> = batch.iter().map(|t| t.next_state.shallow_clone()).collect();

        buffer.states = Tensor::stack(&states, 0);
        buffer.next_states = Tensor::stack(&next_states, 0);

        // Scalar fields are cheap -> build on CPU then move
        let actions: Vec<i64> = batch.iter().map(|t| t.action as i64).collect();
        let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let terminated: Vec<f32> = batch
            .iter()
            .map(|t| if t.terminated { 1.0 } else { 0.0 })
            .collect();
        let truncated: Vec<f32> = batch
            .iter()
            .map(|t| if t.truncated { 1.0 } else { 0.0 })
            .collect();

        buffer.actions = Tensor::from_slice(&actions).to_device(get_device());
        buffer.rewards = Tensor::from_slice(&rewards).to_device(get_device());
        buffer.terminated = Tensor::from_slice(&terminated).to_device(get_device());
        buffer.truncated = Tensor::from_slice(&truncated).to_device(get_device());
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn clear(&mut self) {
        self.transitions.clear();
        self.insertions = 0;
    }
}

pub struct SampleBuffer {
    pub states: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub next_states: Tensor,
    pub terminated: Tensor,
    pub truncated: Tensor,
}

impl SampleBuffer {
    pub fn new(batch_size: i64, state_size: i64) -> Self {
        let device = get_device();
        SampleBuffer {
            states: Tensor::zeros(&[batch_size, state_size], (tch::Kind::Float, device)),
            actions: Tensor::zeros(&[batch_size], (tch::Kind::Int64, device)),
            rewards: Tensor::zeros(&[batch_size], (tch::Kind::Float, device)),
            next_states: Tensor::zeros(&[batch_size, state_size], (tch::Kind::Float, device)),
            terminated: Tensor::zeros(&[batch_size], (tch::Kind::Float, device)),
            truncated: Tensor::zeros(&[batch_size], (tch::Kind::Float, device)),
        }
    }
}
