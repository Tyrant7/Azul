use azul_movegen::board::BOARD_DIMENSION;
use azul_movegen::{Bag, Board, Bowl, GameState, Move, Row, Tile};
use rand::SeedableRng;

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
}

#[test]
fn new_creates_the_expected_components() {
    let state = GameState::new(2, 42);

    assert_eq!(*state.active_player(), 0);
    assert_eq!(state.boards().len(), 2);
    assert_eq!(state.bowls().len(), 6);
    assert_eq!(state.bag().items().len(), 100);
    assert_eq!(*state.first_token_owner(), None);
    assert!(state.round_over());
}

#[test]
fn setup_fills_factory_bowls_and_preserves_determinism() {
    let mut first = GameState::new(2, 42);
    let mut second = GameState::new(2, 42);

    first.setup_next_round();
    second.setup_next_round();

    assert_eq!(*first.active_player(), 0);
    assert_eq!(first.bowls()[0].tiles(), &Vec::<Tile>::new());
    assert!(
        first
            .bowls()
            .iter()
            .skip(1)
            .all(|bowl| bowl.tiles().len() == 4)
    );
    assert_eq!(first.bag().items().len(), 80);
    assert!(!first.round_over());
    assert_eq!(first.bag().items(), second.bag().items());
    for (first_bowl, second_bowl) in first.bowls().iter().zip(second.bowls()) {
        assert_eq!(first_bowl.tiles(), second_bowl.tiles());
    }
}

#[test]
fn valid_moves_include_each_tile_type_and_all_destinations() {
    let mut state = GameState::new(2, 9);
    state.setup_next_round();
    let moves = state.get_valid_moves();
    let expected_count: usize = state
        .bowls()
        .iter()
        .map(|bowl| bowl.get_tile_types().len() * (BOARD_DIMENSION + 1))
        .sum();

    assert_eq!(moves.len(), expected_count);
    for (bowl_index, bowl) in state.bowls().iter().enumerate() {
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
    let mut state = GameState::new(2, 12);
    state.setup_next_round();
    let chosen_bowl = 1;
    let chosen_tile = state.bowls()[chosen_bowl].tiles()[0];
    let remaining_count = state.bowls()[chosen_bowl]
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

    assert_eq!(*state.active_player(), 1);
    assert_eq!(state.boards()[0].holds()[0][0], Some(chosen_tile));
    assert!(!state.bowls()[chosen_bowl].tiles().contains(&chosen_tile));
    assert_eq!(state.bowls()[0].tiles().len(), remaining_count);
}

#[test]
fn illegal_move_is_rejected_without_advancing_the_turn() {
    let mut state = GameState::new(2, 12);
    state.setup_next_round();

    let result = state.make_move(&Move {
        bowl: 1,
        tile_type: BOARD_DIMENSION as Tile,
        row: Row::Floor,
    });

    assert!(result.is_err());
    assert_eq!(*state.active_player(), 0);
}

#[test]
fn first_centre_pick_assigns_the_token_and_penalty() {
    let mut bowls = empty_bowls(4);
    bowls[0] = Bowl::from_tiles(vec![2]);
    let mut state = custom_state(vec![Board::default(); 2], bowls);

    state
        .make_move(&Move {
            bowl: 0,
            tile_type: 2,
            row: Row::Floor,
        })
        .unwrap();

    assert_eq!(*state.first_token_owner(), Some(0));
    assert_eq!(*state.boards()[0].penalties(), 2);
    assert_eq!(*state.active_player(), 1);
}

#[test]
fn setup_uses_first_token_owner_and_clears_the_token() {
    let state = GameState::builder()
        .boards(vec![Board::default(); 2])
        .bowls(empty_bowls(6))
        .bag(Bag::<Tile>::default())
        .first_token_owner(Some(1))
        .set_seed(21)
        .build();
    let mut state = state;

    state.setup_next_round();

    assert_eq!(*state.active_player(), 1);
    assert_eq!(*state.first_token_owner(), None);
    assert!(
        state
            .bowls()
            .iter()
            .skip(1)
            .all(|bowl| bowl.tiles().len() == 4)
    );
}

#[test]
fn builder_preserves_explicit_state() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
    let bag = Bag::new(vec![1, 2, 3], &mut rng);
    let expected_bag = bag.items().clone();
    let boards = vec![Board::default()];
    let bowls = vec![Bowl::from_tiles(vec![4])];
    let state = GameState::builder()
        .active_player(0)
        .boards(boards)
        .bowls(bowls)
        .bag(bag)
        .first_token_owner(Some(0))
        .set_seed(99)
        .build();

    assert_eq!(*state.active_player(), 0);
    assert_eq!(state.boards().len(), 1);
    assert_eq!(state.bowls()[0].tiles(), &vec![4]);
    assert_eq!(state.bag().items(), &expected_bag);
    assert_eq!(*state.first_token_owner(), Some(0));
}

#[test]
fn game_over_and_winner_use_board_completion_and_score() {
    let mut completed_row = [[None; BOARD_DIMENSION]; BOARD_DIMENSION];
    for col in 0..BOARD_DIMENSION {
        completed_row[0][col] = Some(Board::get_tile_type_at_pos(0, col));
    }
    let completed_board = Board::builder().placed(completed_row).build();
    let higher_score = Board::builder().score(8).build();
    let state = custom_state(vec![completed_board, higher_score], empty_bowls(2));

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
        empty_bowls(2),
    );
    assert_eq!(state.get_winner(), 1);

    let tied = custom_state(vec![Board::default(), Board::default()], empty_bowls(2));
    assert_eq!(tied.get_winner(), 1);
}
