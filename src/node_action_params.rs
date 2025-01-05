use crate::{action::{Pi, ValidMoves}, mcts::Turn};

#[derive(Debug)]
pub struct NodeActionInfo {
    pub win_rate: f32,
    pub count: usize,
}

impl NodeActionInfo {
    pub fn new(win_rate: f32, count: usize) -> Self {
        Self { win_rate, count }
    }
}

#[derive(Debug)]
pub struct NodeInfo {
    pub predicted_pi: Pi,
    pub count: usize,
    pub valid_moves: ValidMoves,
	pub _turn: Turn,
}

impl NodeInfo {
    pub fn new(predicted_pi: Pi, count: usize, valid_moves: ValidMoves, turn: Turn) -> Self {
        Self {
            predicted_pi,
            count,
            valid_moves,
			_turn: turn
        }
    }
}
