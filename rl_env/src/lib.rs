use rand::seq::IndexedRandom;

use azul_movegen::{GameState, Move};
use tch::Tensor;

fn encode_gamestate(gamestate: &GameState) -> Tensor {
    let bowls = gamestate.get_bowls();
    let boards = gamestate.get_boards();
    let encoded_bowls = encode_bowls(&bowls);
    let encoded_boards = encode_boards(&boards);
    Tensor::cat(&[encoded_bowls, encoded_boards], 0)
}

fn encode_bowls(bowls: &[Vec<u8>]) -> Tensor {
    unimplemented!()
}

fn encode_boards(boards: &[Board]) -> Tensor {
    unimplemented!()
}

fn calculate_reward(gamestate: &GameState) -> f32 {
    unimplemented!()
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
        StepResult {
            next_state: encode_gamestate(&self.gamestate),
            reward: calculate_reward(&self.gamestate),
            terminated,
            truncated,
        }
    }

    fn map_action(action: usize) -> Move {
        // bowl in [0, 9], in a 4 player game
        // tile_type in [0, 4]
        // row in [0, 5] (0 = floor, 1-5 = wall)
        // so action in [0, 9*5*6) = [0, 270)
        let bowl = action / (5 * 6);
        let tile_type = (action / 6) % 5;
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
