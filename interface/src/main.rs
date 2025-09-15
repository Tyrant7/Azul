// #![allow(dead_code)]

pub mod format;
pub mod parsing;
pub mod protocol;

use std::io;

use azul_movegen::GameState;
use rand::seq::IndexedRandom;

use crate::{format::ProtocolFormat, protocol::Protocol};

fn main() {
    let protocol = Protocol::extract();

    let mut gamestate = GameState::new(2);
    gamestate.setup_next_round();
    println!("{}", gamestate.fmt_protocol(protocol));

    listen_for_input(gamestate, protocol);
}

fn listen_for_input(mut gamestate: GameState, protocol: Protocol) {
    loop {
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let input = input.trim();
        let choice = match protocol::parse_move(input) {
            Ok(m) => m,
            Err(e) => {
                println!("Invalid move: {:?}", e);
                continue;
            }
        };
        println!("move: {:?}", choice);

        match gamestate.make_move(&choice) {
            Err(_) => println!("Illegal move"),
            Ok(_) => println!("{}", gamestate.fmt_protocol(protocol)),
        };

        if gamestate.round_over() {
            gamestate.setup_next_round();
        }

        if gamestate.is_game_over() {
            break;
        }
    }
    println!("Game over");
    println!("Winner: player {}", gamestate.get_winner());
}

fn random_playout(mut gamestate: GameState, protocol: Protocol) {
    loop {
        io::stdin()
            .read_line(&mut String::new())
            .expect("Failed to read input");

        let moves = gamestate.get_valid_moves();
        let selection = moves.choose(&mut rand::rng()).cloned().unwrap_or_default();
        println!("selection: {:?}", selection);

        match gamestate.make_move(&selection) {
            Err(_) => println!("Illegal move"),
            Ok(_) => println!("{}", gamestate.fmt_protocol(protocol)),
        };

        if gamestate.round_over() {
            gamestate.setup_next_round();
        }

        if gamestate.is_game_over() {
            break;
        }
    }
    println!("Game over");
    println!("Winner: player {}", gamestate.get_winner());
}
