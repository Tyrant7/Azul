use azul_movegen::{Bag, Board, Bowl, GameState, board::BOARD_DIMENSION};

use crate::{parsing::ToAzulFEN, protocol::Protocol};

pub trait ProtocolFormat {
    fn fmt_human(&self) -> String;
    fn fmt_uci_like(&self) -> String;

    fn fmt_protocol(&self, protocol: Protocol) -> String {
        match protocol {
            Protocol::Human => self.fmt_human(),
            Protocol::UCILike => self.fmt_uci_like(),
        }
    }
}

impl ProtocolFormat for GameState {
    fn fmt_human(&self) -> String {
        let mut output = String::new();

        // Board printouts
        output.push_str(&"-".repeat(20));
        output.push('\n');
        for (i, board) in self.boards().iter().enumerate() {
            output.push_str(&format!(
                "player {}{}",
                i,
                if *self.active_player() == i {
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

        // Bowl printouts
        for (i, bowl) in self.bowls().iter().enumerate() {
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
        for ((h_idx, hold), row) in self.holds().iter().enumerate().zip(self.placed()) {
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
        output.push_str(&format!("score: {}\n", self.score()));
        output.push_str(&format!("penalties: {}", self.penalties()));
        output.push('\n');
        output.push('\n');
        output
    }

    fn fmt_uci_like(&self) -> String {
        // Format according to AzulFEN specifications
        let mut output = String::new();

        // Placed
        let mut counter = 0;
        for row in self.placed() {
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

        // Holds
        output.push(' ');
        for row in self.holds() {
            let mut tiles = row.iter().flatten();
            if let Some(t) = tiles.next() {
                let count = 1 + tiles.count();
                output.push_str(&t.to_string());
                output.push_str(&count.to_string());
            } else {
                output.push_str("00");
            }
        }

        // Bonuses
        output.push(' ');
        for row in self.bonuses().rows {
            output.push_str(&if row { 1 } else { 0 }.to_string());
        }
        output.push(' ');
        for column in self.bonuses().columns {
            output.push_str(&if column { 1 } else { 0 }.to_string());
        }
        output.push(' ');
        for tile_type in self.bonuses().tile_types {
            output.push_str(&if tile_type { 1 } else { 0 }.to_string());
        }

        // Score and penalties
        output.push(' ');
        output.push_str(&self.score().to_string());
        output.push(' ');
        output.push_str(&self.penalties().to_string());

        // End marker
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

/*
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
*/
