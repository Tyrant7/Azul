use crate::{
    Board,
    bag::Bag,
    board::BOARD_DIMENSION,
    bowl::{Bowl, IllegalMoveError, Move, Tile},
    protocol::{ParseGameStateError, ProtocolFormat},
};

/// The number of tiles of each type to be added to the bag at the beginning of the game, and to be
/// used for reference during round setup.
const TILES_PER_TYPE: usize = 20;

/// The number of tiles that each bowl is restocked to contain during the roubnd setup.
const BOWL_CAPACITY: usize = 4;

/// The index of the centre tile space. Is area is not technically a bowl in the original game, but for
/// simplicity of the code, this decision has been made here.
const CENTRE_BOWL_IDX: usize = 0;

/// Represents a complete gamestate for a given number of players.
/// Supports generation from and serialization to a custom AzulFEN [TODO: link].
#[derive(Debug)]
pub struct GameState {
    active_player: usize,
    boards: Vec<Board>,
    bowls: Vec<Bowl>,
    bag: Bag<Tile>,
    first_token_owner: Option<usize>,
}

/// Bowl formula is given by 2n + 1, with an additional bowl for the centre space.
fn get_bowl_count(players: usize) -> usize {
    players * 2 + 2
}

/// Generates a default tileset for a game setup.
/// By default, [TILES_PER_TYPE] of each tile type are given.
fn get_default_tileset() -> Vec<Tile> {
    let mut tiles = Vec::new();
    // There should always be the same number of tiles as board width
    for t in 0..BOARD_DIMENSION {
        tiles.append(&mut vec![t as Tile; TILES_PER_TYPE]);
    }
    tiles
}

impl GameState {
    /// Creates a new gamestate for the given number of players.
    pub fn new(players: usize) -> Self {
        GameState {
            active_player: 0,
            boards: vec![Board::default(); players],
            bowls: vec![Bowl::default(); get_bowl_count(players)],
            bag: Bag::new(get_default_tileset()),
            first_token_owner: None,
        }
    }

    /// Parses the given AzulFEN into a gamestate.
    /// Will error if the given AzulFEN is invalid.
    /// See the [AzulFEN protocol specification](crate::protocol) for details on the format.
    pub fn from_azul_fen(azul_fen: &str) -> Result<Self, ParseGameStateError> {
        let mut sections = azul_fen.split("| ");

        let board_fens = sections.next().ok_or(ParseGameStateError)?.trim();
        let mut board_fens: Vec<_> = board_fens.split(";").map(|f| f.trim()).collect();
        // Last FEN will always be empty since we split at ";" and each board ends with one
        board_fens.pop();
        let board_fens = board_fens;
        let boards = board_fens
            .into_iter()
            .map(Board::from_board_fen)
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;

        let bowl_fens = sections.next().ok_or(ParseGameStateError)?;
        let bowls = bowl_fens
            .trim()
            .split_ascii_whitespace()
            .map(Bowl::from_bowl_fen)
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;

        let bag_fen = sections.next().ok_or(ParseGameStateError)?;
        let items = bag_fen
            .chars()
            .map(|c| c.to_string().parse::<Tile>().or(Err(ParseGameStateError)))
            .collect::<Result<Vec<_>, ParseGameStateError>>()?;
        let bag = Bag::new(items);

        let active_player_and_first_token = sections.next().ok_or(ParseGameStateError)?;
        let (active_player, first_token_owner) = match active_player_and_first_token
            .split_whitespace()
            .collect::<Vec<_>>()
            .as_slice()
        {
            [active_player, first_token_owner] => (
                active_player
                    .parse::<usize>()
                    .or(Err(ParseGameStateError))?,
                first_token_owner.parse::<usize>().map(Some).unwrap_or(None),
            ),
            _ => return Err(ParseGameStateError),
        };
        Ok(GameState {
            active_player,
            boards,
            bowls,
            bag,
            first_token_owner,
        })
    }

    /// Returns the AzulFEN encoding for this game state.
    /// See the [AzulFEN protocol specification](crate::protocol) for details on the format.
    pub fn get_azul_fen(&self) -> String {
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
        azul_fen.push_str(" | ");
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

    /// Performs a variety of tasks to setup the beginning of a round, including
    /// - Placing held tiles
    /// - Applying previous round penalties
    /// - Refilling bowls
    /// - Restocking the bag, if necessary
    /// - Determining the first player
    /// - Resetting the first player token holder
    pub fn setup_next_round(&mut self) {
        // Place each board's held tiles and apply penalties
        for board in self.boards.iter_mut() {
            board.place_holds();
        }

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

    /// Returns a list of all valid moves in the current gamestate.
    /// This list includes penalizing moves, such as placing tiles to the floor position.
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

    /// Makes a move, modifying the current gamestate.
    /// Will error if the given move is illegal.
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
        Ok(())
    }

    /// Returns true if all bowls are empty, otherwise false.
    pub fn round_over(&self) -> bool {
        self.bowls.iter().all(|b| b.get_tile_types().is_empty())
    }

    /// Returns true if any player has completed a horizontal line on their board.
    pub fn is_game_over(&self) -> bool {
        self.boards.iter().any(|b| b.count_horizontal_lines() > 0)
    }

    /// Gets the index of the board with the highest score.
    /// In the case of a tie, the number of horizontal lines are used.
    /// If there is still a tie, the lower-indexed player will be returned.  
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
