use std::collections::HashMap;
use std::hash::Hash;
use std::sync::mpsc;

use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::Rng;

use crate::action::{Action, Pi};
use crate::constant::{EPS, MOVE_LEN};
use crate::mcts_args::MctsArgs;
use crate::node_action_params::{NodeActionInfo, NodeInfo};
use crate::othello_game::{get_game_ended, get_next_state, get_valid_moves};
use crate::predict_result::PredictResult;
use crate::thread_id::ThreadID;
use crate::{othello_board::OthelloBoard, player::Player};

use std::fmt::Write;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct BoardState {
    pub board: u128,
    pub action: usize,
}

impl BoardState {
    pub fn new(board: u128, action: usize) -> Self {
        Self { board, action }
    }
}

#[derive(Debug)]
pub struct TrainExample {
    pub pi: Pi,
    pub canonical_board: OthelloBoard,
    pub player: Player,
    pub result: i32,
    pub _turn: Turn,
}

pub struct MctsContext {
    pub player_mode: PlayerMode,
    pub p1_mcts_info: MctsInfo,
    pub p2_mcts_info: MctsInfo,
    pub send_to_main: mpsc::Sender<ThreadToMain>,
    pub receive_from_main: mpsc::Receiver<MainToThread>,
    pub thread_id: ThreadID,
    pub args: MctsArgs,
}

pub struct Mcts<'a> {
    pub node_act: &'a mut HashMap<BoardState, NodeActionInfo>,
    pub node: &'a mut HashMap<u128, NodeInfo>,
    pub is_game_end: &'a mut HashMap<u128, i32>,
    pub send_to_main: &'a mut mpsc::Sender<ThreadToMain>,
    pub receive_from_main: &'a mut mpsc::Receiver<MainToThread>,
    pub thread_id: &'a mut ThreadID,
    pub args: &'a mut MctsArgs,
}

///Player1とPlayer2で思考担当が違う場合があり、その場合別々のデータが必要になる
pub struct MctsInfo {
    pub node_act: HashMap<BoardState, NodeActionInfo>,
    pub node: HashMap<u128, NodeInfo>,
    pub is_game_end: HashMap<u128, i32>,
}

impl MctsInfo {
    pub fn new() -> Self {
        Self {
            node_act: HashMap::new(),
            node: HashMap::new(),
            is_game_end: HashMap::new(),
        }
    }
}

pub enum ThreadToMain {
    Board(OthelloBoard, ThreadID, Player, Turn),
    TrainExamples(Vec<TrainExample>, ThreadID),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Turn(pub usize);

impl Turn {
    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

pub enum MainToThread {
    Prediction(PredictResult),
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PlayerMode {
    ///別のAIで思考する
    _2Player,
    ///一つのAIで思考する
    _1Player,
}

impl MctsContext {
    pub fn new(
        player_mode: PlayerMode,
        send_to_main: mpsc::Sender<ThreadToMain>,
        receive_from_main: mpsc::Receiver<MainToThread>,
        thread_id: ThreadID,
        args: MctsArgs,
    ) -> Self {
        Self {
            player_mode,
            p1_mcts_info: MctsInfo::new(),
            p2_mcts_info: MctsInfo::new(),
            send_to_main,
            receive_from_main,
            thread_id,
            args,
        }
    }

    pub fn execute_episode(&mut self) -> Vec<TrainExample> {
        let mut unorthodox_board = OthelloBoard::initial_board();
        let mut cur_player = Player::PLAYER1;
        let mut episode_step: usize = 0;
        let mut train_examples: Vec<(Pi, Player, OthelloBoard, Turn)> = vec![];
        loop {
            episode_step += 1;
            let turn = Turn(episode_step);
            let temp = ((episode_step as i32) < self.args.temp_threshold) as u32 as f32;

            let mut mcts =
                if self.player_mode == PlayerMode::_1Player || cur_player == Player::PLAYER1 {
                    Mcts::new(
                        &mut self.p1_mcts_info.node_act,
                        &mut self.p1_mcts_info.node,
                        &mut self.p1_mcts_info.is_game_end,
                        &mut self.send_to_main,
                        &mut self.receive_from_main,
                        &mut self.thread_id,
                        &mut self.args,
                    )
                } else {
                    Mcts::new(
                        &mut self.p2_mcts_info.node_act,
                        &mut self.p2_mcts_info.node,
                        &mut self.p2_mcts_info.is_game_end,
                        &mut self.send_to_main,
                        &mut self.receive_from_main,
                        &mut self.thread_id,
                        &mut self.args,
                    )
                };

            let pi = mcts.get_action_prob(&unorthodox_board, cur_player, turn, temp);

            train_examples.push((
                pi.clone(),
                cur_player.clone(),
                unorthodox_board.create_canonical_board(cur_player),
                turn,
            ));

            let mut rng = rand::thread_rng();

            let dist = WeightedIndex::new(pi.probs()).unwrap();
            let action = dist.sample(&mut rng);
            get_next_state(&mut unorthodox_board, cur_player, Action::new(action));

            cur_player = cur_player.other();

            let r = get_game_ended(&unorthodox_board, cur_player);

            if r != 0 {
                let result: Vec<TrainExample> = train_examples
                    .into_iter()
                    .map(|(pi, player, canonical_board, _turn)| {
                        let result = r * (-1i32).pow((player != cur_player) as u32);
                        TrainExample {
                            pi,
                            player,
                            canonical_board,
                            result,
                            _turn,
                        }
                    })
                    .collect();

                fn _get_data_to_print(
                    result: &Vec<TrainExample>,
                ) -> Result<String, std::fmt::Error> {
                    let mut print = String::new();
                    let p = &mut print;

                    writeln!(p, "----------TrainExample-----------")?;
                    for item in result {
                        let normal_board = item.canonical_board.create_canonical_board(item.player);
                        writeln!(p, "{}", normal_board._to_string())?;
                        writeln!(p, "{}", item.pi._to_string())?;
                        let diff = normal_board.count_diff(Player::PLAYER1);
                        writeln!(
                            p,
                            "Turn {} Player {} result {} Black {}",
                            item._turn.0,
                            item.player.color(),
                            item.result,
                            diff
                        )?
                    }
                    Ok(print)
                }

                //println!("{}", get_data_to_print(&result).unwrap());

                return result;
            }
        }
    }
}

impl<'a> Mcts<'a> {
    pub fn new(
        node_act: &'a mut HashMap<BoardState, NodeActionInfo>,
        node: &'a mut HashMap<u128, NodeInfo>,
        is_game_end: &'a mut HashMap<u128, i32>,
        send_to_main: &'a mut mpsc::Sender<ThreadToMain>,
        receive_from_main: &'a mut mpsc::Receiver<MainToThread>,
        thread_id: &'a mut ThreadID,
        args: &'a mut MctsArgs,
    ) -> Self {
        Self {
            node_act,
            node,
            is_game_end,
            send_to_main,
            receive_from_main,
            thread_id,
            args,
        }
    }

    pub fn get_action_prob(
        &mut self,
        unorthodox_board: &OthelloBoard,
        player: Player,
        turn: Turn,
        temp: f32,
    ) -> Pi {
        for _ in 0..self.args.num_mcts_sims {
            self.search(unorthodox_board, player, player, turn);
        }

        let canonical_board = unorthodox_board.create_canonical_board(player);
        let s = canonical_board.string_representation();

        let counts: Vec<usize> = (0..MOVE_LEN)
            .map(|a| {
                if let Some(info) = self.node_act.get(&BoardState::new(s, a)) {
                    info.count
                } else {
                    0
                }
            })
            .collect();

        if temp == 0.0 {
            let count_max = *counts.iter().max().unwrap();
            let best_as: Vec<usize> = counts
                .iter()
                .enumerate()
                .filter(|&(_index, &v)| v == count_max)
                .map(|(index, _)| index)
                .collect();

            let mut rng = rand::thread_rng();
            let index = rng.gen_range(0..best_as.len());
            let best_a = best_as[index];
            let mut probs = vec![0.0; counts.len()];
            probs[best_a] = 1.0;
            return Pi::new(&probs);
        } else {
            let counts: Vec<f32> = counts
                .iter()
                .map(|&c| (c as f32).powf(1.0 / temp as f32))
                .collect();
            let counts_sum: f32 = counts.iter().sum();
            let probs: Vec<f32> = counts.iter().map(|&a| a / counts_sum).collect();
            return Pi::new(&probs);
        }
    }

    /// returnするのはPLAYER1から見た勝敗の逆
    pub fn search(
        &mut self,
        unorthodox_board: &OthelloBoard,
        current_player: Player,
        thinking_player: Player,
        turn: Turn,
    ) -> f32 {
        let canonical_board = unorthodox_board.create_canonical_board(current_player);

        let s = canonical_board.string_representation();

        let game_end = if let Some(&game_end) = self.is_game_end.get(&s) {
            game_end
        } else {
            //Canonical BoardのPlayer1から見た勝敗
            let game_end = get_game_ended(&canonical_board, Player::PLAYER1);
            self.is_game_end.insert(s, game_end);
            game_end
        };
        if game_end != 0 {
            //Canonical BoardのPlayer1から見た勝敗はunorthodox boardでplayer2から見た勝敗と一致する
            //なので実際はcurrent_playerの情報はsearch関数では必要ない。元ソースにはないが、分かりやすくしたいのでいれている。
            return -game_end as f32;
        }

        let Some(node_info) = self.node.get_mut(&s) else {
            self.send_to_main
                .send(ThreadToMain::Board(
                    canonical_board.clone(),
                    self.thread_id.clone(),
                    thinking_player,
                    turn,
                ))
                .unwrap();
            let MainToThread::Prediction(r) = self.receive_from_main.recv().unwrap();

            let mut pi = r.action_probs;

            let valid_moves = get_valid_moves(&canonical_board, Player::PLAYER1);
            valid_moves.apply(&mut pi);
            let sum_ps_s: f32 = pi.probs().iter().sum();
            if 0.0 < sum_ps_s {
                for i in 0..MOVE_LEN {
                    pi[i] /= sum_ps_s;
                }
            } else {
                // 可能な手の確率がすべて0。こんなことあるんだろうか？
                // とりあえず可能な手を均等に選ぶ
                for i in 0..MOVE_LEN {
                    pi[i] = if valid_moves.actions[i] { 1.0 } else { 0.0 };
                }
                let sum: f32 = pi.probs().iter().sum();
                for i in 0..MOVE_LEN {
                    pi[i] /= sum;
                }
            }
            //普通は最初のcountは1であろうが、元ソースでは0で動くようになっているので踏襲。
            self.node.insert(s, NodeInfo::new(pi, 0, valid_moves, turn));
            return -r.win_rate;
        };

        let valids = &node_info.valid_moves;
        let mut cur_best = f32::NEG_INFINITY;
        let mut best_act = 0;

        for a in 0..MOVE_LEN {
            if valids[a] {
                let u = if let Some(node_act) = self.node_act.get(&BoardState::new(s, a)) {
                    let q = node_act.win_rate;
                    q + self.args.cpuct
                        * node_info.predicted_pi[a]
                        * (node_info.count as f32).sqrt()
                        / (1.0 + node_act.count as f32)
                } else {
                    self.args.cpuct
                        * node_info.predicted_pi[a]
                        * (node_info.count as f32 + EPS).sqrt()
                };

                if cur_best < u {
                    cur_best = u;
                    best_act = a;
                }
            }
        }
        node_info.count += 1;

        let a = best_act;
        //boardはもう使わないので実際のところcloneしなくてもよいが、論理的にはcloneすべきだと思うのでcloneする
        let mut next_s = unorthodox_board.clone();
        get_next_state(&mut next_s, current_player, Action::new(a));
        let v = self.search(
            &next_s,
            current_player.other(),
            thinking_player,
            turn.next(),
        );

        let bs = BoardState::new(s, a);
        if let Some(node_act) = self.node_act.get_mut(&bs) {
            let n = &mut node_act.count;
            let q = &mut node_act.win_rate;

            *q = ((*q) * (*n as f32) + v) / (*n as f32 + 1.0);
            *n += 1;
        } else {
            self.node_act.insert(bs.clone(), NodeActionInfo::new(v, 1));
        }

        return -v;
    }
}

fn _predict_dummy() -> PredictResult {
    let vec: Vec<_> = (0..MOVE_LEN).map(|i| 1.0 - 0.0001 * i as f32).collect();
    PredictResult {
        action_probs: Pi::new(&vec),
        win_rate: 0.1,
    }
}
