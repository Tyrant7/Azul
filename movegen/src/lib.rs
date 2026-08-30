//! Core rules and state representation for Azul.
//!
//! The crate models the parts of an Azul game needed to generate and apply moves:
//! [`GameState`] owns the players, centre bowl, factory bowls, tile bag, turn
//! order, and first-player token; [`Board`] owns one player's wall, pattern-line holds,
//! penalties, score, and collected bonuses. [`Move`] identifies a tile type
//! taken from a bowl and the destination row on the active player's board.
//!
//! A typical game loop creates a [`GameState`], calls
//! [`GameState::setup_next_round`], repeatedly selects an item from
//! [`GameState::get_valid_moves`] and applies it with
//! [`GameState::make_move`], then starts another round when
//! [`GameState::round_over`] becomes true. The game ends when
//! [`GameState::is_game_over`] becomes true.
//!
//! The implementation supports two- through four-player games, uses five tile
//! types, and uses a five-by-five wall. Tile types are represented by the
//! [`Tile`] alias, and rows are represented by [`Row`].
//! The rules source material is the [Azul rulebook][rulebook].
//!
//! [rulebook]: https://cdn.1j1ju.com/medias/03/14/fd-azul-rulebook.pdf

/// Identifies a tile type. Tile instances have no state beyond their type, so a small integer is sufficient.
pub type Tile = usize;

/// Generates read-only reference getters for the listed fields.
macro_rules! ref_getters {
    ($($field:ident : $ty:ty), *$(,)?) => {
        paste::paste! {
            $(
                pub fn [<get_$field>](&self) -> &$ty {
                    &self.$field
                }
            )*
        }
    };
}

/// Generates read-only value getters for the listed fields.
macro_rules! value_getters {
    ($($field:ident : $ty:ty), *$(,)?) => {
        paste::paste! {
            $(
                pub fn [<get_$field>](&self) -> $ty {
                    self.$field
                }
            )*
        }
    };
}

pub mod board;
pub mod game_move;
pub mod gamestate;

mod bag;
mod bowl;
mod row;

pub use bag::Bag;
pub use board::Board;
pub use bowl::Bowl;
pub use game_move::{BowlChoice, Move};
pub use gamestate::{GameState, GameStateError, TOTAL_TILE_COUNT, Xoshiro256PlusPlus};
pub use row::Row;
