// Rules adapted from: https://cdn.1j1ju.com/medias/03/14/fd-azul-rulebook.pdf

const BOWL_CAPACITY: usize = 4;

use std::io;

mod bag;
mod board;
mod gamestate;
mod movegen;
mod utility;

use board::Board;

use crate::gamestate::GameState;

fn main() {
    let mut gamestate = GameState::new(2);
    gamestate.setup();

    println!("{:?}", gamestate);

    loop {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let input = input.trim();
        let choice = match utility::parse_move(input) {
            Ok(m) => m,
            Err(e) => {
                println!("Invalid move: {:?}", e);
                continue;
            }
        };
        println!("move: {:?}", choice);
        match gamestate.make_move(choice) {
            Err(_) => println!("Illegal move"),
            Ok(_) => println!("{:?}", gamestate),
        };
    }
}
