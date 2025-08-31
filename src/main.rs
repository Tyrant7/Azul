// Rules adapted from: https://cdn.1j1ju.com/medias/03/14/fd-azul-rulebook.pdf

mod bag;
mod board;
mod gamestate;
mod movegen;

use board::Board;

use crate::gamestate::GameState;

fn main() {
    let mut gamestate = GameState::new(2);
}
