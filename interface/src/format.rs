//! Protocol and human-readable formatting for movegen values.

use azul_movegen::{Bag, Board, Bowl, GameState, board::BOARD_DIMENSION};

use crate::{parsing::ToAzulFEN, protocol::Protocol};

/// Formats movegen values for human output or the machine-readable protocol.
pub trait ProtocolFormat {
    /// Formats the value for a human reader.
    fn fmt_human(&self) -> String;
    /// Formats the value using the machine-readable AzulFEN representation
    /// used by the UAI protocol.
    fn fmt_uci_like(&self) -> String;

    /// Selects a formatter based on the requested protocol mode.
    fn fmt_protocol(&self, protocol: Protocol) -> String {
        match protocol {
            Protocol::Human => self.fmt_human(),
            Protocol::UAI => self.fmt_uci_like(),
        }
    }
}
impl ProtocolFormat for GameState {
    fn fmt_human(&self) -> String {
        let mut output = String::new();

        // Format player boards and turn information.
        output.push_str(&"-".repeat(20));
        output.push('\n');
        for (i, board) in self.get_boards().iter().enumerate() {
            output.push_str(&format!(
                "player {}{}",
                i,
                if self.get_active_player() == i {
                    " (active)"
                } else {
                    ""
                }
            ));
            output.push('\n');
            output.push_str(&board.fmt_human());
        }
        output.push_str(&"-".repeat(20));
        output.push('\n');

        // Format factory bowls and the centre area.
        for (i, bowl) in self.get_bowls().iter().enumerate() {
            output.push_str(&format!("{}: {} | ", i, bowl.fmt_human()));
        }
        output
    }

    fn fmt_uci_like(&self) -> String {
        self.to_azul_fen()
    }
}

impl ProtocolFormat for Board {
    fn fmt_human(&self) -> String {
        let mut output = String::new();
        for ((h_idx, hold), row) in self.get_holds().iter().enumerate().zip(self.get_placed()) {
            output.push_str(&(h_idx + 1).to_string());
            output.push_str(&"  ".repeat(BOARD_DIMENSION - h_idx));
            for h in 0..h_idx + 1 {
                if let Some(h) = hold.get(h).and_then(|x| *x) {
                    output.push_str(&h.to_string());
                    output.push(' ');
                } else {
                    output.push_str(". ");
                }
            }
            output.push_str(" | ");
            for p in 0..BOARD_DIMENSION {
                if let Some(p) = row.get(p).and_then(|x| *x) {
                    output.push_str(&p.to_string());
                    output.push(' ');
                } else {
                    output.push_str(". ");
                }
            }
            output.push('\n');
        }
        output.push_str(&format!("score: {}\n", self.get_score()));
        output.push_str(&format!("penalties: {}", self.get_penalties()));
        output.push('\n');
        output.push('\n');
        output
    }

    fn fmt_uci_like(&self) -> String {
        // Format according to the AzulFEN board component specification.
        let mut output = String::new();

        // Encode placed wall tiles with run-length counts for empty spaces.
        let mut counter = 0;
        for row in self.get_placed() {
            for tile in row {
                if tile.is_some() {
                    if counter > 0 {
                        output.push_str(&counter.to_string());
                    }
                    output.push('-');
                    counter = 0;
                } else {
                    counter += 1;
                }
            }
            if counter > 0 {
                output.push_str(&counter.to_string());
            }
            counter = 0;
            output.push('/');
        }
        output.pop();

        // Encode pattern lines as tile-type/count pairs.
        output.push(' ');
        for row in self.get_holds() {
            let mut tiles = row.iter().flatten();
            if let Some(t) = tiles.next() {
                let count = 1 + tiles.count();
                output.push_str(&t.to_string());
                output.push_str(&count.to_string());
            } else {
                output.push_str("00");
            }
        }

        // Encode collected row, column, and tile-type bonuses.
        output.push(' ');
        for row in self.get_bonuses().rows {
            output.push_str(&if row { 1 } else { 0 }.to_string());
        }
        output.push(' ');
        for column in self.get_bonuses().columns {
            output.push_str(&if column { 1 } else { 0 }.to_string());
        }
        output.push(' ');
        for tile_type in self.get_bonuses().tile_types {
            output.push_str(&if tile_type { 1 } else { 0 }.to_string());
        }

        // Encode score, occupied penalty spaces, and physical penalty tiles.
        output.push(' ');
        output.push_str(&self.get_score().to_string());
        output.push(' ');
        output.push_str(&self.get_penalties().to_string());
        output.push(' ');
        output.push_str(&self.get_penalty_tiles().to_string());

        // Terminate the board component.
        output.push_str(" ;");
        output
    }
}

impl ProtocolFormat for Bowl {
    fn fmt_human(&self) -> String {
        if self.tiles().is_empty() {
            return String::from("-");
        }
        self.tiles().iter().map(|t| t.to_string()).collect()
    }

    fn fmt_uci_like(&self) -> String {
        self.fmt_human()
    }
}

impl<T> ProtocolFormat for Bag<T>
where
    T: ToString,
{
    fn fmt_human(&self) -> String {
        "".to_string()
    }

    fn fmt_uci_like(&self) -> String {
        self.items().iter().map(|t| t.to_string()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::ProtocolFormat;
    use crate::parsing::{FromAzulFEN, ToAzulFEN};
    use crate::protocol::Protocol;
    use azul_movegen::{Bag, Board, Bowl, GameState};

    #[test]
    fn bowl_and_bag_formatters_use_expected_machine_values() {
        let bowl = Bowl::from_tiles(vec![2, 0, 2]);
        assert_eq!(bowl.fmt_human(), "022");
        assert_eq!(bowl.fmt_uci_like(), "022");
        assert_eq!(bowl.fmt_protocol(Protocol::Human), "022");
        assert_eq!(bowl.fmt_protocol(Protocol::UAI), "022");

        let bag = Bag::from_items(vec![1, 2, 3]);
        assert_eq!(bag.fmt_human(), "");
        assert_eq!(bag.fmt_uci_like(), "123");
    }

    #[test]
    fn board_machine_format_round_trips() {
        let board = Board::builder()
            .score(12)
            .penalties(3)
            .penalty_tiles(2)
            .build();
        let encoded = board.fmt_uci_like();
        let component = encoded.trim_end_matches(" ;");
        let parsed = Board::from_azul_fen(component).unwrap();

        assert_eq!(parsed.get_score(), board.get_score());
        assert_eq!(parsed.get_penalties(), board.get_penalties());
        assert_eq!(parsed.get_penalty_tiles(), board.get_penalty_tiles());
        assert!(board.fmt_human().contains("score: 12"));
        assert!(board.fmt_human().contains("penalties: 3"));
    }

    #[test]
    fn game_state_protocol_format_matches_azulfen() {
        let mut game = GameState::new(2, 42).unwrap();
        game.setup_next_round();

        assert_eq!(game.fmt_uci_like(), game.to_azul_fen());
        assert_eq!(game.fmt_protocol(Protocol::UAI), game.to_azul_fen());

        let human = game.fmt_protocol(Protocol::Human);
        assert!(human.contains("player 0 (active)"));
        assert!(human.contains("player 1"));
        assert!(human.contains("0: "));
    }
}
