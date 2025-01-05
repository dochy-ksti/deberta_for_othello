#![allow(unused_imports)]
#![allow(dead_code)]
use std::{sync::mpsc, thread};

use rand::{distributions::WeightedIndex, prelude::Distribution, random, Rng};

use crate::{
    action::Pi,
    c_array::CArray,
    constant::{BATCH_SIZE, MOVE_LEN},
    mcts::{MainToThread, Mcts, MctsContext, PlayerMode, ThreadToMain},
    mcts_args::MctsArgs,
    othello_board::OthelloBoard,
    player::Player,
    predict_result::PredictResult,
    py_communicator::PyCommunicator,
    self_player::SelfPlayer,
    thread_id::ThreadID,
};

//#[test]
pub fn do_test_mcts() {
    small_test();
    //thread_test();
    //commu_test()
}

pub fn small_test() {
    let mut hoge = CArray::<f32>::new3(3, 5, 5);

    hoge.set3(2, 2, 2, 10.0);
    println!("size {:?}", hoge.size());

    println!("{}", hoge._to_string());
}
pub fn thread_test() {
    let (send_to_main, receive_from_thread) = mpsc::channel();
    let (send_to_thread, receive_from_main) = mpsc::channel();
    let thread_id = ThreadID::new(0);
    thread::spawn(move || {
        
        let mut mcts = MctsContext::new(
			PlayerMode::_1Player,
            send_to_main.clone(),
            receive_from_main,
            thread_id.clone(),
            MctsArgs::default(),
        );
        let examples = mcts.execute_episode();
        send_to_main.send(ThreadToMain::TrainExamples(examples, thread_id))
    });
    let mut vec = vec![];

    let examples = loop {
        match receive_from_thread.recv().unwrap() {
            ThreadToMain::Board(_board, _thread_id, _player, _turn) => {
                vec.push((_board._to_string(), _player));
                //println!("{}", board.to_string());
                send_to_thread.send(dummy_data()).unwrap()
            }
            ThreadToMain::TrainExamples(_examples, _thread_id) => {
                //println!("done");
                //println!("{:?}", examples);
                break _examples;
            }
        }
    };
    for (_board_s, _p) in vec {
        //println!("{board_s}\n{}", p.color());
    }
    // for ex in examples {
    //     println!(
    //         "{}\nresult {} player {}",
    //         ex.canonical_board.create_canonical_board(ex.player)._to_string(),
    //         ex.result,
    //         ex.player.color()
    //     );
    // }
    for _ex in examples {
        // println!(
        //     "{}\nreal_result {} player {} move_candidates {}",
        //     ex.canonical_board
        //         .create_canonical_board(ex.player)
        //         ._to_string(),
        //     ex.result * ex.player.color(),
        //     ex.player.color(),
        // 	ex.pi._to_string(),
        // );
    }
}

pub fn commu_test() {
    let py = PyCommunicator::new();
    let mut sp = SelfPlayer::new(PlayerMode::_1Player, &py.pool, &MctsArgs::default());

    loop {
        match sp.prepare_next(0, false) {
            0 => {}
            1 => {
                let _boards = sp.get_boards_for_prediction(0, false);
                let (pis, win_rates) = dummy_carrays();
                sp.receive_prediction(&pis, &win_rates, 0, false);
            }
            2 => {
                let hoge = sp.get_pis_for_training();
                println!("{}", hoge._to_string());
                break;
            }
            _ => unreachable!(),
        }
    }
}

fn dummy_data() -> MainToThread {
    MainToThread::Prediction(dummy_data_b())
}

fn dummy_data_b() -> PredictResult {
    let vec = (0..MOVE_LEN)
        .into_iter()
        .map(|_| random::<f32>())
        .collect::<Vec<f32>>();
    PredictResult {
        win_rate: (0.1 * rand::thread_rng().gen_range(1..=19) as f32) - 1.0,
        action_probs: Pi {
            action_probs: Box::new(vec.try_into().unwrap()),
        },
    }
}

fn dummy_carrays() -> (CArray<f32>, CArray<f32>) {
    let vec: Vec<PredictResult> = (0..BATCH_SIZE)
        .into_iter()
        .map(|_| dummy_data_b())
        .collect();

    let (vec1, vec2): (Vec<_>, Vec<_>) = vec
        .into_iter()
        .map(|a| (a.action_probs, a.win_rate))
        .collect();
    let mut pis = CArray::<f32>::new2(BATCH_SIZE, MOVE_LEN);
    for (idx, item) in vec1.into_iter().enumerate() {
        pis.ref_mut2(idx).copy_from_slice(item.probs());
    }
    let mut win_rates = CArray::<f32>::new2(BATCH_SIZE, 1);
    win_rates.as_mut().copy_from_slice(&vec2);
    (pis, win_rates)
}
