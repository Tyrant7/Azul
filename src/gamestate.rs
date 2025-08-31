use crate::{Board, movegen::Bowl};

pub struct GameState {
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
}

fn get_bowl_count(players: usize) -> usize {
    players * 2 + 1
}

impl GameState {
    pub fn new(players: usize) -> Self {
        GameState {
            boards: vec![Board::new(); players],
            bowls: vec![Bowl::new(); get_bowl_count(players)],
        }
    }
}
