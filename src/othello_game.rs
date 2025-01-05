use crate::action::{Action, ValidMoves};
use crate::othello_board::OthelloBoard;
use crate::player::Player;

pub fn get_next_state(board: &mut OthelloBoard, player: Player, action: Action) {
    if action.is_pass() {
        return;
    }
    board.execute_move(action.to_move(), player);
}

pub fn get_valid_moves(board: &OthelloBoard, player: Player) -> ValidMoves {
    let mut valids = ValidMoves::new();
    let legal_moves = board.get_legal_moves(player);
    if legal_moves.len() == 0 {
        *valids.pass() = true;
        return valids;
    }
    for m in legal_moves {
        *valids.sq(m.x(), m.y()) = true;
    }
    return valids;
}

pub fn get_game_ended(board: &OthelloBoard, player: Player) -> i32 {
    if board.has_legal_moves(player) {
        return 0;
    }
    if board.has_legal_moves(player.other()) {
        return 0;
    }

    let diff = board.count_diff(player);

    if 0 < diff {
        return 1;
    } else if diff == 0 {
        //同点は先手の勝ちにする(オセロとして正しい処理はドロー)
        return if player == Player::PLAYER1 { 1 } else { -1 };
    } else {
        return -1;
    }
}
