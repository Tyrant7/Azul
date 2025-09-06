use crate::{
    BOWL_CAPACITY, Board,
    bag::Bag,
    board::BOARD_DIMENSION,
    bowl::{Bowl, IllegalMoveError, Move, Tile},
    protocol::ProtocolFormat,
};

const TILES_PER_TYPE: usize = 20;

const CENTRE_BOWL_IDX: usize = 0;

#[derive(Debug)]
pub struct GameState {
    active_player: usize,
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
}

fn get_bowl_count(players: usize) -> usize {
    players * 2 + 2
}

fn get_default_tileset() -> Vec<Tile> {
    let mut tiles = Vec::new();
    // There should always be the same number of tiles as board width
    for t in 0..BOARD_DIMENSION {
        tiles.append(&mut vec![t as Tile; TILES_PER_TYPE]);
    }
    tiles
}

impl GameState {
    pub fn new(players: usize) -> Self {
        GameState {
            active_player: 0,
            boards: vec![Board::new(); players],
            bowls: vec![Bowl::new(); get_bowl_count(players)],
            bag: Bag::new(get_default_tileset()),
            first_token_owner: None,
        }
    }

    pub fn from_azul_fen(azul_fen: &str) -> Self {
        let mut sections = azul_fen.split("| ");

        let board_fens = sections
            .next()
            .unwrap_or_else(|| panic!("Invalid AzulFEN: {}", azul_fen));
        let board_fens: Vec<_> = board_fens.split(";").collect();
        let boards: Vec<_> = board_fens.into_iter().map(Board::from_board_fen).collect();

        let bowl_fens = sections
            .next()
            .unwrap_or_else(|| panic!("Invalid AzulFEN: {}", azul_fen));
        let bowls: Vec<_> = bowl_fens
            .trim()
            .split_ascii_whitespace()
            .map(Bowl::from_bowl_fen)
            .collect();

        let bag_fen = sections
            .next()
            .unwrap_or_else(|| panic!("Invalid AzulFEN: {}", azul_fen));

        todo!();

        let active_player_and_first_token = sections
            .next()
            .unwrap_or_else(|| panic!("Invalid AzulFEN: {}", azul_fen));

        todo!();

        todo!()
    }

    pub fn get_azul_fen(&self) -> String {
        /*
        AzulFEN works as follows:

        -
        Boards:

        Each board's placed tiles are broken down into their own FEN-style string where numbers represent N empty spaces,
        "/" denotes a new line, and a - represents a tile in that position
        e.x.  5/5/5/5/5 is an empty board
        while 5/5/2-2/5/5 would have a single tile in the centre

        Each row of a board's hold section can be encoded with two numbers. The first represents the tile type in that row,
        and the second representing the number of tiles. The encodings for each row are written sequentially
        e.x.  0042000000 corresponds to 2 tiles of type 4 in the second row

        For each board, the collected bonuses also need to be known. Each bonus type is encoded individually, in the order of
        [row, column, tile_type], and sequentially to one another, with a space in between where 0 is an uncollected bonus
        and 1 is a collected bonus.
        e.x.  00001 00000 00000 corresponds to having collected only the horizontal bonus for the final row

        The score and penalty tiles for each board and encoded and single numbers at the end of the FEN
        e.x.  10 3 corresponds to 10 score and 3 penalty tiles

        And finally, each board FEN is separated by a semi-colon

        Altogether a typical board FEN may look something like follows:
        2-1-/-4/--3/5/4- 0011000013 00000 00000 00000 7 1 ;

        -
        Bowls:

        The bowl's section is prefixed with a "|" character

        Each bowl is encoded as a sequence of numbers corresponding to tile types, each with a space in between
        An empty bowl is denoted with a "-"
        e.x.  000234 - 1132 would correspond to three unique bowls, one centre, one empty, and one full

        -
        Bag:

        The bag's section is prefixed with another "|" character

        The bag is simply listed as a sequence of numbers corresponding to tile types
        e.x.  03440140321203

        -
        Active player and first player token:

        Finally, the active player and first player token owner and encoded at the end in order as two numbers,
        once again, prefixed with a "|" character
        e.x.  0 2 corresponds to the active player being player 0, and the first player token owner being player 2
        If nobody owns the first player token, then "-" will be written in its place

        -
        In full, a complete AzulFEN may look something like the following

        2-1-/-4/--3/5/4- 0011000013 00000 00000 00000 7 1 ;
        1--1-/-4/1-3/4-/4- 0000220013 00000 00000 00000 10 0 ;
        | 0123003 - - - 0123 0001
        | 0133041230412404142
        | 0 -

        AzulFENs should be outputted on a single-line, with a newline as the final character
        */

        // Boards
        let mut azul_fen = String::new();
        for board in self.boards.iter() {
            azul_fen.push_str(&board.fmt_uci_like());
            azul_fen.push(' ');
        }

        // Bowls
        azul_fen.push_str("| ");
        for bowl in self.bowls.iter() {
            azul_fen.push_str(&bowl.fmt_uci_like());
            azul_fen.push(' ');
        }

        // Bag
        azul_fen.push_str("| ");
        azul_fen.push_str(&self.bag.fmt_uci_like());

        // Active player and first player token
        azul_fen.push_str("| ");
        azul_fen.push_str(&self.active_player.to_string());
        azul_fen.push(' ');
        azul_fen.push_str(&if let Some(t) = self.first_token_owner {
            t.to_string()
        } else {
            "-".to_string()
        });

        azul_fen.push('\n');
        azul_fen
    }

    pub fn setup(&mut self) {
        // Fill each bowl, skipping the centre
        let (bowls, bag) = (&mut self.bowls, &mut self.bag);
        for bowl in bowls.iter_mut().skip(1) {
            let mut next: Vec<Tile> = bag.take(BOWL_CAPACITY).collect();
            if next.len() < BOWL_CAPACITY {
                // Refill the bag with all tiles currently not in play
                let mut used_tiles = Vec::new();
                for board in &self.boards {
                    used_tiles.extend(board.get_active_tiles());
                }
                let mut unused_tiles = Vec::new();
                for t in 0..BOARD_DIMENSION {
                    unused_tiles.append(&mut vec![
                        t as Tile;
                        TILES_PER_TYPE
                            - used_tiles
                                .iter()
                                .filter(|&&x| x == t as Tile)
                                .count()
                    ]);
                }
                bag.restock(unused_tiles);
            }
            next.extend(bag.take(BOWL_CAPACITY - next.len()));
            bowl.fill(next.clone());
        }

        // At the end of setup, the player with the first player's token goes first
        self.active_player = self.first_token_owner.unwrap_or_default();
        self.first_token_owner = None;
    }

    pub fn get_valid_moves(&self) -> Vec<Move> {
        let board = self.boards.get(self.active_player).expect("Invalid player");
        let mut moves = Vec::new();
        for (bowl_idx, bowl) in self.bowls.iter().enumerate() {
            for tile in bowl.get_tile_types() {
                for row in board.get_valid_rows_for_tile_type(tile) {
                    moves.push(Move {
                        bowl: bowl_idx,
                        tile_type: tile,
                        row,
                    });
                }
            }
        }
        moves
    }

    pub fn make_move(&mut self, choice: &Move) -> Result<(), IllegalMoveError> {
        let valid_moves = self.get_valid_moves();
        if !valid_moves.contains(choice) {
            return Err(IllegalMoveError);
        }

        // Get the tiles and update the bowls
        let tiles = self
            .bowls
            .get_mut(choice.bowl)
            .ok_or(IllegalMoveError)?
            .take_tiles(choice.tile_type);

        // A penalty is given if we're the first player to pick from the centre
        let penalty = if choice.bowl == CENTRE_BOWL_IDX && self.first_token_owner.is_none() {
            self.first_token_owner = Some(self.active_player);
            1
        } else {
            0
        };

        // Put the tiles into the appropriate row
        let active_board = self
            .boards
            .get_mut(self.active_player)
            .expect("Invalid player");
        active_board.hold_tiles(choice.tile_type, tiles.0.len(), choice.row, penalty)?;

        // Move the remaining tiles to the centre
        self.bowls
            .get_mut(CENTRE_BOWL_IDX)
            .expect("Invalid bowl")
            .extend(&tiles.1);

        // Cycle to the next player's turn
        self.active_player += 1;
        if self.active_player >= self.boards.len() {
            self.active_player = 0;
        }

        // If neither player can play, or if we have no more tiles
        // let's setup for the next round
        if self.bowls.iter().all(|b| b.get_tile_types().is_empty()) {
            for board in self.boards.iter_mut() {
                board.place_holds();
            }
            self.setup();
        }
        Ok(())
    }

    pub fn is_game_over(&self) -> bool {
        self.boards.iter().any(|b| b.count_horizontal_lines() > 0)
    }

    pub fn get_winner(&self) -> usize {
        self.boards
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| (b.get_score(), b.count_horizontal_lines()))
            .unwrap()
            .0
    }
}

impl ProtocolFormat for GameState {
    fn fmt_human(&self) -> String {
        let mut output = String::new();

        // Board printouts
        output.push_str(&"-".repeat(20));
        output.push('\n');
        for (i, board) in self.boards.iter().enumerate() {
            output.push_str(&format!(
                "player {}{}",
                i,
                if self.active_player == i {
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
        for (i, bowl) in self.bowls.iter().enumerate() {
            output.push_str(&format!("{}: {} | ", i, bowl.fmt_human()));
        }
        output
    }

    fn fmt_uci_like(&self) -> String {
        self.get_azul_fen()
    }
}
