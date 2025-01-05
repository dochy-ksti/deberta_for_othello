use crate::{action::Move, constant::N, player::Player};
use std::{
    collections::HashSet,
    ops::{Index, IndexMut},
};

pub const DIRECTIONS: [(i32, i32); 8] = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
];

#[derive(Debug, Clone, PartialEq)]
pub struct OthelloBoard(pub [[i32; N]; N]);

impl Index<usize> for OthelloBoard {
    type Output = [i32; N];

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for OthelloBoard {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl OthelloBoard {
    pub fn new() -> Self {
        Self([[0; N]; N])
    }

    pub fn initial_board() -> Self {
        let mut b = Self::new();
        b[N / 2 - 1][N / 2] = 1;
        b[N / 2][N / 2 - 1] = 1;
        b[N / 2 - 1][N / 2 - 1] = -1;
        b[N / 2][N / 2] = -1;
        b
    }

    pub fn count_diff(&self, player: Player) -> i32 {
        let color = player.color();
        let mut count = 0;
        for x in 0..N {
            for y in 0..N {
                if self[x][y] == color {
                    count += 1;
                }
                if self[x][y] == -color {
                    count -= 1;
                }
            }
        }
        return count;
    }

    pub fn get_legal_moves(&self, player: Player) -> Vec<Move> {
        let mut moves: HashSet<Move> = HashSet::new();
        self.get_legal_moves_impl(player, &mut moves);
        moves.into_iter().collect()
    }

    pub fn has_legal_moves(&self, player: Player) -> bool {
        let mut b = false;
        self.get_legal_moves_impl(player, &mut b);
        return b;
    }

    fn get_legal_moves_impl(&self, player: Player, moves: &mut impl MoveSet) {
        for x in 0..N {
            for y in 0..N {
                if self[x][y] == 0 {
                    self.get_moves_for_square(Move::new(x, y), player, moves);
                }
            }
        }
    }

    fn get_moves_for_square(&self, m: Move, player: Player, moves: &mut impl MoveSet) {
        for dir in DIRECTIONS {
            let mut flips: i32 = 0;
            if self.discover_move(m, dir, player, &mut flips) {
                if moves.insert_move(m) == false {
                    return;
                }
            }
        }
    }

    fn discover_move(
        &self,
        m: Move,
        dir: (i32, i32),
        player: Player,
        flips: &mut impl AddFlip,
    ) -> bool {

        let n = N as i32;
        let mut x = m.x() as i32;
        let mut y = m.y() as i32;
        loop {
            x += dir.0;
            y += dir.1;
            if x < 0 || n <= x {
                return false;
            }
            if y < 0 || n <= y {
                return false;
            }
            let x = x as usize;
            let y = y as usize;

            if self[x][y] == 0 {
                return false;
            } else if self[x][y] == player.other().color() {
                flips.add(Move::new(x, y));
            } else if self[x][y] == player.color() {
                if flips.has_flip() {
                    return true;
                } else {
                    return false;
                }
            }
        }
    }

    pub fn execute_move(&mut self, m: Move, player: Player) {
        let mut successed = false;
        for dir in DIRECTIONS {
            let mut vec = vec![];
            if self.discover_move(m, dir, player, &mut vec) {
                self[m.x()][m.y()] = player.color();
                for m in vec {
                    self[m.x()][m.y()] = player.color();
                }
                successed = true;
            }
        }
        if successed == false {
            panic!("impossible execute_move");
        }
    }

    pub fn _to_string(&self) -> String {
        fn to_masu(c: &i32) -> &'static str {
            match c {
                -1 => "⚪️",
                0 => "🔴",
                1 => "⚫️",
                _ => panic!("Othello board must be 0,-1, or 1"),
            }
        }
        let b: Vec<String> = self
            .0
            .iter()
            .map(|a| a.iter().map(|c| to_masu(c)).collect())
            .collect();
        b.join("\n")
    }

    pub fn create_canonical_board(&self, player: Player) -> OthelloBoard {
        let mut b = self.clone();
        b.canonical_form(player);
        b
    }
}

trait AddFlip {
    fn add(&mut self, sq: Move);
    fn has_flip(&self) -> bool;
}

impl AddFlip for Vec<Move> {
    fn add(&mut self, sq: Move) {
        self.push(sq)
    }

    fn has_flip(&self) -> bool {
        !self.is_empty()
    }
}

impl AddFlip for i32 {
    fn add(&mut self, _: Move) {
        *self += 1;
    }

    fn has_flip(&self) -> bool {
        *self != 0
    }
}

trait MoveSet {
    /// サーチを継続するかを返す
    fn insert_move(&mut self, sq: Move) -> bool;
}

impl MoveSet for HashSet<Move> {
    fn insert_move(&mut self, sq: Move) -> bool {
        self.insert(sq);
        return true;
    }
}

impl MoveSet for bool {
    fn insert_move(&mut self, _: Move) -> bool {
        *self = true;
        return false;
    }
}

impl OthelloBoard {
    /// プレイヤーの視点を変える。自分がどこに着手するかをAIに学習させるので、
    /// その場合の色は統一する必要があるから、Player2の場合白黒反転する
    pub fn canonical_form(&mut self, player: Player) {
        if player == Player::PLAYER2 {
            for x in 0..N {
                for y in 0..N {
                    self[x][y] = self[x][y] * -1;
                }
            }
        }
    }

    /// 8*8*(-1~1)なので128bitで表せる
    pub fn string_representation(&self) -> u128 {
        fn to_2bit(v: i32) -> u128 {
            if v < 0 {
                2
            } else {
                v as u128
            }
        }
        let mut r: u128 = 0;
        for x in 0..N {
            for y in 0..N {
                r = r | (to_2bit(self[x][y]) << ((x * N + y) * 2) as u128);
            }
        }
        r
    }
}
