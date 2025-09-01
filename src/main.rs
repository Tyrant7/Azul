// Rules adapted from: https://cdn.1j1ju.com/medias/03/14/fd-azul-rulebook.pdf

const BOWL_CAPACITY: usize = 4;

mod bag;
mod board;
mod gamestate;
mod movegen;

use board::Board;

use crate::gamestate::GameState;

fn main() {
    let mut gamestate = GameState::new(2);
    gamestate.setup();

    println!("{:#?}", gamestate);
}
