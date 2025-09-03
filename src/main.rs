// Rules adapted from: https://cdn.1j1ju.com/medias/03/14/fd-azul-rulebook.pdf

const BOWL_CAPACITY: usize = 4;

use std::io;

mod bag;
mod board;
mod bowl;
mod gamestate;
mod utility;

use board::Board;
use rand::seq::IndexedRandom;

use crate::{bowl::Move, gamestate::GameState};

fn main() {
    let mut gamestate = GameState::new(2);
    gamestate.setup();
    println!("{:?}", gamestate);

    random_playout(gamestate);
}

fn listen_for_input(mut gamestate: GameState) {
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

        match gamestate.make_move(&choice) {
            Err(_) => println!("Illegal move"),
            Ok(_) => println!("{:?}", gamestate),
        };
    }
}

fn random_playout(mut gamestate: GameState) {
    loop {
        io::stdin()
            .read_line(&mut String::new())
            .expect("Failed to read input");

        let moves = gamestate.get_valid_moves();
        println!("moves: {:?}", moves);

        let selection = moves.choose(&mut rand::rng()).unwrap_or(&Move {
            bowl: 0,
            tile_type: 0,
            row: 0,
        });
        println!("selection: {:?}", selection);

        match gamestate.make_move(selection) {
            Err(_) => println!("Illegal move"),
            Ok(_) => println!("{:?}", gamestate),
        };
    }
}
