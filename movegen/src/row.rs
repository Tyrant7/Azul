/// Identifies a destination for tiles selected during a move.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Row {
    /// The floor is always a valid destination, but its tiles incur penalties and are not scored.
    /// A player may choose it even when a wall row is available.
    #[default]
    Floor,
    /// A wall pattern line. The parameter is a zero-based row index from top to bottom.
    Wall(usize),
}
