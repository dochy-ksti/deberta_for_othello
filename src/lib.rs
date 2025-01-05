mod action;
mod c_array;
mod constant;
mod mcts;
mod mcts_args;
mod othello_board;
mod othello_game;
mod player;
mod predict_result;
mod py_communicator;
mod self_player;
mod test_mcts;
mod thread_id;
mod node_action_params;

fn _main(){
	test_mcts::do_test_mcts();
}

#[test]
fn main_test(){
	test_mcts::do_test_mcts();
}