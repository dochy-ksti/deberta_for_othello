#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Player(i32);

impl Player {
    pub const PLAYER1: Player = Player(1);
    pub const PLAYER2: Player = Player(-1);

    pub fn color(&self) -> i32 {
        self.0
    }

    pub fn other(&self) -> Player {
        Player(-self.0)
    }
}

