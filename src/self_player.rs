use std::sync::mpsc::{self, Receiver, Sender};

use threadpool::ThreadPool;

use crate::{
    c_array::CArray,
    constant::{BATCH_SIZE, MOVE_LEN, N},
    mcts::{MainToThread, MctsContext, PlayerMode, ThreadToMain, TrainExample},
    mcts_args::MctsArgs,
    othello_board::OthelloBoard,
    player::Player,
    predict_result::PredictResult,
    py_communicator::PyCommunicator,
    thread_id::ThreadID,
};

pub struct ThreadInfo {
    pub send_to_thread: Sender<MainToThread>,
    pub receive_from_thread: Receiver<ThreadToMain>,
    pub data: Option<ThreadToMain>,
}

pub struct SelfPlayer {
    thread_infos: Vec<ThreadInfo>,
    train_examples: Vec<Vec<TrainExample>>,
    examples_count: Option<usize>,
}

impl SelfPlayer {
    pub fn new(player_mode: PlayerMode, pool: &ThreadPool, mcts_args: &MctsArgs) -> Self {
        let mut thread_infos = vec![];
        for index in 0..BATCH_SIZE {
            let thread_id = ThreadID::new(index);
            let (send_to_main, receive_from_thread) = mpsc::channel::<ThreadToMain>();
            let (send_to_thread, receiver_for_thread) = mpsc::channel::<MainToThread>();
            //thread_idとvecのindexが同値になるようにしている
            thread_infos.push(ThreadInfo {
                send_to_thread,
                receive_from_thread,
                data: None,
            });
            let mcts_args = mcts_args.clone();
            pool.execute(move || {
                let mut mcts = MctsContext::new(
                    player_mode,
                    send_to_main.clone(),
                    receiver_for_thread,
                    thread_id.clone(),
                    mcts_args,
                );
                let r = mcts.execute_episode();
                send_to_main
                    .send(ThreadToMain::TrainExamples(r, thread_id))
                    .unwrap();
            });
        }
        Self {
            thread_infos,
            train_examples: vec![],
            examples_count: None,
        }
    }

    fn examples_flatten(&self) -> (impl Iterator<Item = &TrainExample>, usize) {
        //くっそ汚い
        (
            self.train_examples.iter().flat_map(|a| a.iter()),
            self.examples_count.unwrap(),
        )
    }

    /// NスレッドでN個の試合を同時にシミュレーションしている。一手進めて盤面を返す
    ///
    /// 最初の指し手は必ずplayer1とする。
    ///
    /// player: isize
    /// 0ならplayerを問わない
    /// 1ならplayer1の盤面を準備する
    /// -1ならplayer2の盤面を準備する
    ///
    /// 戻り値:
    /// 0: playerの取得できる盤面がない
    /// 1: 盤面の準備が出来た
    /// 2: すべての試合が既に終わっていて、トレーニング用のデータの準備が出来た
    pub fn prepare_next(&mut self, player: isize, mixed_thinker: bool) -> usize {
        if self.train_examples.is_empty() == false {
            panic!("Train examples have been prepared. No need to do prepare_next()");
        }

        for info in &mut self.thread_infos {
            if info.data.is_none() {
                info.data = Some(info.receive_from_thread.recv().unwrap())
            }
        }

        let mut all_training = true;

        for info in &self.thread_infos {
            match &info.data {
                Some(ThreadToMain::Board(_board, id, p, _t)) => {
                    if is_player(p, player, mixed_thinker, id) {
                        return 1;
                    } else {
                        all_training = false;
                    }
                }
                Some(ThreadToMain::TrainExamples(_, _)) => {}
                None => {
                    return 0;
                }
            }
        }

        if all_training {
            let mut vec = vec![];
            for info in &mut self.thread_infos {
                if let ThreadToMain::TrainExamples(a, _) = info.data.take().unwrap() {
                    vec.push(a);
                } else {
                    unreachable!()
                }
            }
            self.train_examples = vec;
            self.examples_count = Some(self.train_examples.iter().map(|a| a.len()).sum());
            return 2;
        } else {
            return 0;
        }
    }

    pub fn get_boards_for_prediction(&self, player: isize, mixed_thinker: bool) -> CArray<f32> {
        let mut r = CArray::<f32>::new3(BATCH_SIZE, N, N);

        for info in &self.thread_infos {
            match &info.data {
                Some(ThreadToMain::Board(b, id, thinking_player, _t)) => {
                    if is_player(thinking_player, player, mixed_thinker, id) {
                        copy_board(r.ref_mut3_1(id.id()), b);
                    }
                }
                _ => {}
            }
        }

        r
    }

    pub fn receive_prediction(
        &mut self,
        pis: &CArray<f32>,
        win_rates: &CArray<f32>,
        int_player: isize,
		mixed_thinker: bool,
    ) {
        let predicts = PredictResult::convert_from_carrays(pis, win_rates);
        for (predict, info) in predicts.into_iter().zip(self.thread_infos.iter_mut()) {
            let b = if let Some(ThreadToMain::Board(_b, id, p, _t)) = &info.data {
                if is_player(p, int_player, mixed_thinker, id) {
                    info.send_to_thread
                        .send(MainToThread::Prediction(predict))
                        .unwrap();
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if b {
                info.data = None
            }
        }
    }

    pub fn get_pis_for_training(&self) -> CArray<f32> {
        if self.train_examples.is_empty() {
            panic!("train_examples is not prepared");
        }
        let (examples, len) = self.examples_flatten();
        let mut array = CArray::<f32>::new2(len, MOVE_LEN);

        for (idx, example) in examples.enumerate() {
            array.ref_mut2(idx).copy_from_slice(example.pi.probs())
        }

        array
    }

    pub fn get_boards_for_training(&self) -> CArray<f32> {
        if self.train_examples.is_empty() {
            panic!("train_examples is not prepared");
        }
        let (examples, len) = self.examples_flatten();
        let mut array = CArray::<f32>::new3(len, N, N);

        for (idx, example) in examples.enumerate() {
            copy_board(array.ref_mut3_1(idx), &example.canonical_board);
        }
        array
    }

    pub fn get_players_for_training(&mut self) -> CArray<f32> {
        if self.train_examples.is_empty() {
            panic!("train_examples is not prepared");
        }
        let (examples, len) = self.examples_flatten();
        let mut array = CArray::<f32>::new1(len);

        for (idx, example) in examples.enumerate() {
            array.as_mut()[idx] = example.player.color() as f32;
        }
        array
    }

    pub fn get_results_for_training(&mut self) -> CArray<f32> {
        if self.train_examples.is_empty() {
            panic!("train_examples is not prepared");
        }
        let (examples, len) = self.examples_flatten();

        let mut array = CArray::<f32>::new1(len);

        for (idx, example) in examples.enumerate() {
            array.as_mut()[idx] = example.result as f32;
        }
        array
    }

    pub fn get_results_for_counting(&mut self) -> CArray<f32> {
        if self.train_examples.is_empty() {
            panic!("train_examples is not prepared");
        }
        let example_len = self.train_examples.len();
        let mut array = CArray::<f32>::new1(example_len);

        for (idx, example) in self.train_examples.iter().enumerate() {
            array.as_mut()[idx] = example[0].result as f32;
        }
        array
    }
}

fn is_player(player: &Player, int_player: isize, mixed_thinker: bool, id: &ThreadID) -> bool {
    let is_player = int_player == 0 || int_player == player.color() as isize;
    if mixed_thinker {
		//半分より上のThreadIDでは、Thinkerを逆にする
		//XORの逆のNXOR演算になっているらしい。
        if id.id() < BATCH_SIZE / 2 {
            is_player
        } else {
            !is_player
        }
    } else {
        is_player
    }
}

fn copy_board(slice: &mut [f32], board: &OthelloBoard) {
    let b = board.0.as_flattened();
    for i in 0..b.len() {
        slice[i] = b[i] as f32;
    }
}

/// player_modeは1か2。それ以外の場合NULL POINTER(0)が返る
#[no_mangle]
pub extern "C" fn create_self_player(
    p: *mut PyCommunicator,
    player_mode: usize,
) -> *mut SelfPlayer {
    let player_mode = if player_mode == 1 {
        PlayerMode::_1Player
    } else if player_mode == 2 {
        PlayerMode::_2Player
    } else {
        return 0 as *mut SelfPlayer;
    };
    unsafe {
        let b = Box::new(SelfPlayer::new(player_mode, &(*p).pool, &(*p).mcts_args));
        Box::into_raw(b)
    }
}

#[no_mangle]
pub extern "C" fn destroy_self_player(p: *mut SelfPlayer) {
    unsafe {
        let _ = Box::from_raw(p);
    }
}

/// NスレッドでN個の試合を同時にシミュレーションしている。一手進めて盤面を返す
///
/// 最初の指し手は必ずplayer1とする。
///
/// player: usize
/// 0ならplayerを問わない
/// 1ならplayer1の盤面を準備する
/// -1ならplayer2の盤面を準備する
///
/// 戻り値:
/// 0: そのプレイヤーが全部passであったりして、返すべき盤面がない
/// 1: 盤面の準備が出来た
/// 2: すべての試合が既に終わっていて、トレーニング用のデータの準備が出来た
#[no_mangle]
pub extern "C" fn self_player_prepare_next(p: *mut SelfPlayer, player: isize, mixed_thinker: isize) -> usize {
    unsafe { (*p).prepare_next(player, mixed_thinker != 0) }
}

#[no_mangle]
pub extern "C" fn self_player_get_boards_for_prediction(
    p: *mut SelfPlayer,
    player: isize,
	is_mixed_thinker: isize,
) -> *mut CArray<f32> {
    unsafe {
        let b = Box::new((*p).get_boards_for_prediction(player, is_mixed_thinker != 0));
        Box::into_raw(b)
    }
}

#[no_mangle]
pub extern "C" fn self_player_get_pis_for_training(p: *mut SelfPlayer) -> *mut CArray<f32> {
    unsafe {
        let b = Box::new((*p).get_pis_for_training());
        Box::into_raw(b)
    }
}

#[no_mangle]
pub extern "C" fn self_player_get_boards_for_training(p: *mut SelfPlayer) -> *mut CArray<f32> {
    unsafe {
        let b = Box::new((*p).get_boards_for_training());
        Box::into_raw(b)
    }
}

#[no_mangle]
pub extern "C" fn self_player_get_players_for_training(p: *mut SelfPlayer) -> *mut CArray<f32> {
    unsafe {
        let b = Box::new((*p).get_players_for_training());
        Box::into_raw(b)
    }
}

#[no_mangle]
pub extern "C" fn self_player_get_results_for_training(p: *mut SelfPlayer) -> *mut CArray<f32> {
    unsafe {
        let b = Box::new((*p).get_results_for_training());
        Box::into_raw(b)
    }
}

#[no_mangle]
pub extern "C" fn self_player_get_results_for_counting(p: *mut SelfPlayer) -> *mut CArray<f32> {
    unsafe {
        let b = Box::new((*p).get_results_for_counting());
        Box::into_raw(b)
    }
}

#[no_mangle]
pub extern "C" fn self_player_receive_prediction(
    p: *mut SelfPlayer,
    pis: *mut CArray<f32>,
    win_rates: *mut CArray<f32>,
    player: isize,
	is_mixed_thinker: usize,
) {
    unsafe {
        (*p).receive_prediction(&*pis, &*win_rates, player, is_mixed_thinker != 0);
    }
}
