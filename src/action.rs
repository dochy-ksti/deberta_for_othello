use std::ops::{Index, IndexMut};

use crate::constant::{MOVE_LEN, N};

#[derive(Debug, Clone, Copy)]
pub struct Action(usize);

impl Action {
    pub fn is_pass(&self) -> bool {
        self.0 == N * N
    }

    pub fn _val(&self) -> usize {
        self.0
    }

    pub fn new(v: usize) -> Self {
        Self(v)
    }

    pub fn to_move(&self) -> Move {
        debug_assert!(self.is_pass() == false);
        let a = self.0;
        return Move::new(a / N, a % N);
    }
}

#[derive(Debug, Clone)]
pub struct Pi {
    pub action_probs: Box<[f32; MOVE_LEN]>,
}

impl Pi {
    pub fn probs(&self) -> &[f32] {
        self.action_probs.as_ref()
    }

    pub fn _probs_mut(&mut self) -> &mut [f32] {
        self.action_probs.as_mut()
    }

    pub fn new(action_probs: &[f32]) -> Self {
        Self {
            action_probs: Box::new(
                action_probs
                    .try_into()
                    .expect("The length of action_probs is not MOVE_LEN"),
            ),
        }
    }

    pub fn _to_string1(&self) -> String {
        let mut r = String::new();
        for i in 0..MOVE_LEN {
            if 0.0 < self.action_probs[i] {
                r.push_str(&Action::new(i).to_move()._to_string());
            }
        }
        r
    }

    pub fn _to_string(&self) -> String {
        let mut r = String::new();
        for i in 0..MOVE_LEN {
            r.push_str(&format!("{:.2} ", self[i]));
        }
        r
    }
}

impl Index<usize> for Pi {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.action_probs[index]
    }
}

impl IndexMut<usize> for Pi {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.action_probs[index]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(usize, usize);

impl Move {
    pub fn new(x: usize, y: usize) -> Self {
        Self(x, y)
    }

    pub fn _to_action(&self) -> Action {
        Action::new(self.x() * N + self.y())
    }

    pub fn x(&self) -> usize {
        self.0
    }

    pub fn y(&self) -> usize {
        self.1
    }

    pub fn _to_string(&self) -> String {
        format!("[x:{} y:{}]", self.x(), self.y())
    }
}

#[derive(Debug)]
pub struct ValidMoves {
    pub actions: [bool; MOVE_LEN],
}

impl ValidMoves {
    pub fn new() -> Self {
        Self {
            actions: [false; MOVE_LEN],
        }
    }

    pub const fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn pass(&mut self) -> &mut bool {
        let len = self.len();
        //一番最後は"パス"のアクション
        &mut self.actions[len - 1]
    }

    pub fn sq(&mut self, x: usize, y: usize) -> &mut bool {
        &mut self.actions[x * N + y]
    }

    pub fn apply(&self, moves: &mut Pi) {
        for i in 0..MOVE_LEN {
            if self.actions[i] == false {
                moves[i] = 0.0;
            }
        }
    }

    pub fn _to_string(&self) -> String {
        let mut r = String::new();
        for i in 0..MOVE_LEN {
            if self.actions[i] {
                let a = Action::new(i).to_move();
                r.push_str(&a._to_string());
            }
        }
        r
    }
}

impl Index<usize> for ValidMoves {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        &self.actions[index]
    }
}

impl IndexMut<usize> for ValidMoves {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.actions[index]
    }
}
