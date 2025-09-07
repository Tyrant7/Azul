// Rules adapted from: https://cdn.1j1ju.com/medias/03/14/fd-azul-rulebook.pdf

const BOWL_CAPACITY: usize = 4;

use std::io;

mod bag;
mod board;
mod bowl;
mod gamestate;
mod protocol;

use board::Board;
use rand::seq::IndexedRandom;

use crate::{
    bowl::Move,
    gamestate::GameState,
    protocol::{Protocol, ProtocolFormat},
};

fn main() {
    let mut old_gamestate = GameState::new(2);
    old_gamestate.setup();
    let new_gamestate = GameState::from_azul_fen(&old_gamestate.get_azul_fen());

    println!("{:?}", new_gamestate);

    return;

    let protocol = Protocol::extract();

    let mut gamestate = GameState::new(2);
    gamestate.setup();
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
        let selection = moves.choose(&mut rand::rng()).unwrap_or(&Move {
            bowl: 0,
            tile_type: 0,
            row: bowl::Row::Floor,
        });
        println!("selection: {:?}", selection);

        match gamestate.make_move(selection) {
            Err(_) => println!("Illegal move"),
            Ok(_) => println!("{}", gamestate.fmt_protocol(protocol)),
        };

        if gamestate.is_game_over() {
            break;
        }
    }
    println!("Game over");
    println!("Winner: player {}", gamestate.get_winner());
}
