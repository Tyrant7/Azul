use azul_movegen::board::BOARD_DIMENSION;
use azul_movegen::{
    Bag, Board, Bowl, GameState, GameStateError, Move, Row, TOTAL_TILE_COUNT, Tile,
    Xoshiro256PlusPlus,
};

/// Creates a vector of empty bowls with the requested length.
fn empty_bowls(count: usize) -> Vec<Bowl> {
    vec![Bowl::default(); count]
}

/// Creates a minimal state suitable for testing transitions.
fn custom_state(boards: Vec<Board>, bowls: Vec<Bowl>) -> GameState {
    GameState::builder()
        .boards(boards)
        .bowls(bowls)
        .bag(Bag::<Tile>::default())
        .set_seed(17)
        .build()
        .unwrap()
}

#[test]
fn new_creates_the_expected_components() {
    let state = GameState::new(2, 42).unwrap();

    assert_eq!(state.get_active_player(), 0);
    assert_eq!(state.get_boards().len(), 2);
    assert_eq!(state.get_bowls().len(), 6);
    assert_eq!(state.get_bag().items().len(), 100);
    assert_eq!(state.get_first_token_owner(), None);
    assert_eq!(state.get_discarded_tiles(), 0);
    assert_eq!(state.get_tile_count(), TOTAL_TILE_COUNT);
    assert!(state.round_over());
}

#[test]
fn new_supports_two_three_and_four_players() {
    for players in 2..=4 {
        let state = GameState::new(players, 42).unwrap();

        assert_eq!(state.get_boards().len(), players);
        assert_eq!(state.get_bowls().len(), players * 2 + 2);
    }
}

#[test]
fn new_rejects_unsupported_player_counts() {
    for players in [0, 1, 5, usize::MAX] {
        assert!(matches!(
            GameState::new(players, 42),
            Err(GameStateError::InvalidPlayerCount { players: actual }) if actual == players
        ));
    }
}

#[test]
fn builder_rejects_empty_and_structurally_invalid_states() {
    assert!(matches!(
        GameState::builder().build(),
        Err(GameStateError::InvalidPlayerCount { players: 0 })
    ));

    assert!(matches!(
        GameState::builder()
            .boards(vec![Board::default(); 2])
            .bowls(empty_bowls(5))
            .build(),
        Err(GameStateError::InvalidBowlCount {
            expected: 6,
            actual: 5
        })
    ));

    assert!(matches!(
        GameState::builder()
            .active_player(2)
            .boards(vec![Board::default(); 2])
            .bowls(empty_bowls(6))
            .build(),
        Err(GameStateError::InvalidActivePlayer { active_player: 2 })
    ));

    assert!(matches!(
        GameState::builder()
            .boards(vec![Board::default(); 2])
            .bowls(empty_bowls(6))
            .first_token_owner(Some(2))
            .build(),
        Err(GameStateError::InvalidFirstTokenOwner { player: 2 })
    ));

    assert!(matches!(
        GameState::builder()
            .boards(vec![Board::default(); 2])
            .bowls(empty_bowls(6))
            .set_rng_state(vec![0; 32])
            .build(),
        Err(GameStateError::InvalidRngState)
    ));
}

#[test]
fn setup_fills_factory_bowls_and_preserves_determinism() {
    let mut first = GameState::new(2, 42).unwrap();
    let mut second = GameState::new(2, 42).unwrap();

    first.setup_next_round();
    second.setup_next_round();

    assert_eq!(first.get_active_player(), 0);
    assert_eq!(first.get_bowls()[0].tiles(), &Vec::<Tile>::new());
    assert!(
        first
            .get_bowls()
            .iter()
            .skip(1)
            .all(|bowl| bowl.tiles().len() == 4)
    );
    assert_eq!(first.get_bag().items().len(), 80);
    assert_eq!(first.get_tile_count(), TOTAL_TILE_COUNT);
    assert!(!first.round_over());
    assert_eq!(first.get_bag().items(), second.get_bag().items());
    for (first_bowl, second_bowl) in first.get_bowls().iter().zip(second.get_bowls()) {
        assert_eq!(first_bowl.tiles(), second_bowl.tiles());
    }
}

#[test]
fn valid_moves_include_each_tile_type_and_all_destinations() {
    let mut state = GameState::new(2, 9).unwrap();
    state.setup_next_round();
    let moves = state.get_valid_moves();
    let expected_count: usize = state
        .get_bowls()
        .iter()
        .map(|bowl| bowl.get_tile_types().len() * (BOARD_DIMENSION + 1))
        .sum();

    assert_eq!(moves.len(), expected_count);
    for (bowl_index, bowl) in state.get_bowls().iter().enumerate() {
        for tile_type in bowl.get_tile_types() {
            for row in (0..BOARD_DIMENSION)
                .map(Row::Wall)
                .chain(std::iter::once(Row::Floor))
            {
                assert!(moves.contains(&Move {
                    bowl: bowl_index,
                    tile_type,
                    row,
                }));
            }
        }
    }
}

#[test]
fn make_move_updates_board_bowls_and_active_player() {
    let mut state = GameState::new(2, 12).unwrap();
    state.setup_next_round();
    let chosen_bowl = 1;
    let chosen_tile = state.get_bowls()[chosen_bowl].tiles()[0];
    let remaining_count = state.get_bowls()[chosen_bowl]
        .tiles()
        .iter()
        .filter(|&&tile| tile != chosen_tile)
        .count();

    state
        .make_move(&Move {
            bowl: chosen_bowl,
            tile_type: chosen_tile,
            row: Row::Wall(0),
        })
        .unwrap();

    assert_eq!(state.get_active_player(), 1);
    assert_eq!(state.get_boards()[0].get_holds()[0][0], Some(chosen_tile));
    assert!(
        !state.get_bowls()[chosen_bowl]
            .tiles()
            .contains(&chosen_tile)
    );
    assert_eq!(state.get_bowls()[0].tiles().len(), remaining_count);
}

#[test]
fn illegal_move_is_rejected_without_advancing_the_turn() {
    let mut state = GameState::new(2, 12).unwrap();
    state.setup_next_round();

    let result = state.make_move(&Move {
        bowl: 1,
        tile_type: BOARD_DIMENSION as Tile,
        row: Row::Floor,
    });

    assert!(result.is_err());
    assert_eq!(state.get_active_player(), 0);
}

#[test]
fn first_centre_pick_assigns_the_token_and_penalty() {
    let mut bowls = empty_bowls(6);
    bowls[0] = Bowl::from_tiles(vec![2]);
    let mut state = custom_state(vec![Board::default(); 2], bowls);

    state
        .make_move(&Move {
            bowl: 0,
            tile_type: 2,
            row: Row::Floor,
        })
        .unwrap();

    assert_eq!(state.get_first_token_owner(), Some(0));
    assert_eq!(state.get_boards()[0].get_penalties(), 2);
    assert_eq!(state.get_active_player(), 1);
}

#[test]
fn round_over_requires_the_centre_and_all_factory_bowls_to_be_empty() {
    let mut bowls = empty_bowls(6);
    bowls[0] = Bowl::from_tiles(vec![2]);
    let mut state = custom_state(vec![Board::default(); 2], bowls);

    assert!(!state.round_over());
    state
        .make_move(&Move {
            bowl: 0,
            tile_type: 2,
            row: Row::Floor,
        })
        .unwrap();
    assert!(state.round_over());
}

#[test]
fn game_over_is_detected_after_a_completed_pattern_line_is_placed() {
    let mut holds = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
    holds[0][0] = Some(0);
    let mut placed = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
    for col in 1..BOARD_DIMENSION {
        placed[0][col] = Some(Board::get_tile_type_at_pos(0, col));
    }
    let state = GameState::builder()
        .boards(vec![
            Board::builder().holds(holds).placed(placed).build(),
            Board::default(),
        ])
        .bowls(empty_bowls(6))
        .bag(Bag::default())
        .set_seed(29)
        .build()
        .unwrap();
    let mut state = state;

    assert!(!state.is_game_over());
    state.setup_next_round();
    assert!(state.is_game_over());
}

#[test]
fn setup_tracks_discarded_tiles_without_counting_the_token() {
    let mut board = Board::default();
    board.hold_tiles(2, 2, Row::Floor, 1).unwrap();
    let mut bowls = empty_bowls(6);
    bowls[0] = Bowl::from_tiles(vec![3]);
    let mut state = GameState::builder()
        .boards(vec![board, Board::default()])
        .bowls(bowls)
        .bag(Bag::from_items(vec![0; 97]))
        .first_token_owner(Some(0))
        .set_seed(23)
        .build()
        .unwrap();

    assert_eq!(state.get_tile_count(), TOTAL_TILE_COUNT);
    state.setup_next_round();

    assert_eq!(state.get_discarded_tiles(), 2);
    assert_eq!(state.get_tile_count(), TOTAL_TILE_COUNT);
}

#[test]
fn restocking_excludes_tiles_already_dealt_during_setup() {
    let mut placed = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
    placed[0][0] = Some(0);
    let state = GameState::builder()
        .boards(vec![
            Board::builder().placed(placed).build(),
            Board::default(),
        ])
        .bowls(empty_bowls(6))
        .bag(Bag::from_items(vec![0]))
        .discarded_tiles(98)
        .set_seed(31)
        .build()
        .unwrap();
    let mut state = state;

    assert_eq!(state.get_tile_count(), TOTAL_TILE_COUNT);
    state.setup_next_round();

    let mut counts = [0; BOARD_DIMENSION];
    for tile in state
        .get_bag()
        .items()
        .iter()
        .copied()
        .chain(
            state
                .get_bowls()
                .iter()
                .flat_map(|bowl| bowl.tiles().iter().copied()),
        )
        .chain(
            state
                .get_boards()
                .iter()
                .flat_map(|board| board.get_active_tiles()),
        )
    {
        counts[tile] += 1;
    }
    assert!(counts.iter().all(|&count| count <= 20));
    assert_eq!(state.get_tile_count(), TOTAL_TILE_COUNT);
}

#[test]
fn tile_count_is_conserved_through_random_gameplay() {
    let mut state = GameState::new(2, 103).unwrap();
    state.setup_next_round();

    for step in 0..300 {
        assert_eq!(state.get_tile_count(), TOTAL_TILE_COUNT, "step {step}");
        if state.is_game_over() {
            break;
        }
        if state.round_over() {
            state.setup_next_round();
            continue;
        }
        let choice = state.get_valid_moves().into_iter().next().unwrap();
        state.make_move(&choice).unwrap();
    }

    assert_eq!(state.get_tile_count(), TOTAL_TILE_COUNT, "after gameplay");
}

#[test]
fn setup_uses_first_token_owner_and_clears_the_token() {
    let state = GameState::builder()
        .boards(vec![Board::default(); 2])
        .bowls(empty_bowls(6))
        .bag(Bag::<Tile>::default())
        .first_token_owner(Some(1))
        .set_seed(21)
        .build()
        .unwrap();
    let mut state = state;

    state.setup_next_round();

    assert_eq!(state.get_active_player(), 1);
    assert_eq!(state.get_first_token_owner(), None);
    assert!(
        state
            .get_bowls()
            .iter()
            .skip(1)
            .all(|bowl| bowl.tiles().len() == 4)
    );
}

#[test]
fn builder_preserves_explicit_state() {
    let mut rng = Xoshiro256PlusPlus::from_seed_u64(3);
    let bag = Bag::new(vec![1, 2, 3], &mut rng);
    let expected_bag = bag.items().clone();
    let boards = vec![Board::default(); 2];
    let mut bowls = empty_bowls(6);
    bowls[0] = Bowl::from_tiles(vec![4]);
    let state = GameState::builder()
        .active_player(0)
        .boards(boards)
        .bowls(bowls)
        .bag(bag)
        .first_token_owner(Some(0))
        .set_seed(99)
        .build()
        .unwrap();

    assert_eq!(state.get_active_player(), 0);
    assert_eq!(state.get_boards().len(), 2);
    assert_eq!(state.get_bowls()[0].tiles(), &vec![4]);
    assert_eq!(state.get_bag().items(), &expected_bag);
    assert_eq!(state.get_first_token_owner(), Some(0));
}

#[test]
fn game_over_and_winner_use_board_completion_and_score() {
    let mut completed_row = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
    for col in 0..BOARD_DIMENSION {
        completed_row[0][col] = Some(Board::get_tile_type_at_pos(0, col));
    }
    let completed_board = Board::builder().placed(completed_row).build();
    let higher_score = Board::builder().score(8).build();
    let state = custom_state(vec![completed_board, higher_score], empty_bowls(6));

    assert!(state.is_game_over());
    assert_eq!(state.get_winner(), 1);
}

#[test]
fn winner_tie_breaks_by_horizontal_lines_then_later_index() {
    let mut one_line = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
    for col in 0..BOARD_DIMENSION {
        one_line[0][col] = Some(Board::get_tile_type_at_pos(0, col));
    }
    let state = custom_state(
        vec![Board::default(), Board::builder().placed(one_line).build()],
        empty_bowls(6),
    );
    assert_eq!(state.get_winner(), 1);

    let tied = custom_state(vec![Board::default(), Board::default()], empty_bowls(6));
    assert_eq!(tied.get_winner(), 1);
}
