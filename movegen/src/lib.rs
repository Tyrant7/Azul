/// The alias type for tiles. Since held and placed tiles have no unique properties beyond needing
/// to be differentiable, `usize` was used for the underlying type for tiles.
pub type Tile = usize;

pub mod board;
pub mod game_move;
pub mod gamestate;

mod bag;
mod bowl;
mod row;

pub use bag::Bag;
pub use board::Board;
pub use bowl::Bowl;
pub use game_move::Move;
pub use gamestate::GameState;
pub use row::Row;
