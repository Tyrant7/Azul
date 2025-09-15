/// This enum represents a row where tiles can be placed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Row {
    /// The floor row is always a valid position to place tiles, however floor tiles incur penalties and are not scored.
    /// When no valid `Wall` rows remaing, tiles must be placed on the floor row.
    #[default]
    Floor,
    /// Tiles may only be placed on the wall in valid rows. The parameter `usize` represents the index from top to bottom.
    Wall(usize),
}
