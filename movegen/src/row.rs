/// This enum represents a row where tiles can be placed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Row {
    /// The floor row is always a valid position to place tiles, however floor tiles incur penalties and are not scored.
    /// When no valid `Wall` rows remaing, tiles must be placed on the floor row.
    Floor,
    /// Tiles may only be placed on the wall in valid rows. The parameter `usize` represents the index from top to bottom.
    Wall(usize),
}

impl std::fmt::Display for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                Row::Floor => "-".to_string(),
                Row::Wall(i) => i.to_string(),
            }
        )
    }
}
