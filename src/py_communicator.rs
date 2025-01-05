
use crate::{
    constant::{BATCH_SIZE, BOARD_SIZE, MOVE_LEN, N},
    mcts_args::MctsArgs,
};

use threadpool::ThreadPool;

pub struct PyCommunicator {
    pub pool: ThreadPool,
    pub mcts_args: MctsArgs,
}

impl PyCommunicator {
    pub fn new() -> Self {
        let mcts_args = MctsArgs::default();
        Self {
            pool: ThreadPool::new(BATCH_SIZE),
            mcts_args,
        }
    }
}

#[no_mangle]
pub extern "C" fn create_py_communicator() -> *mut PyCommunicator {
    let b = Box::new(PyCommunicator::new());
    Box::into_raw(b)
}

#[no_mangle]
pub extern "C" fn destroy_py_communicator(p: *mut PyCommunicator) {
    unsafe {
        let _ = Box::from_raw(p);
    }
}

#[no_mangle]
pub extern "C" fn batch_size() -> usize {
    BATCH_SIZE
}

#[no_mangle]
pub extern "C" fn size_y() -> usize {
    N
}

#[no_mangle]
pub extern "C" fn size_x() -> usize {
    N
}

#[no_mangle]
pub extern "C" fn move_len() -> usize {
    MOVE_LEN
}

#[no_mangle]
pub extern "C" fn board_size() -> usize {
    BOARD_SIZE
}