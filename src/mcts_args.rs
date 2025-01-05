#[derive(Debug, Clone)]
pub struct MctsArgs {
    pub temp_threshold: i32,
    pub num_mcts_sims: i32,
    pub cpuct: f32,
}

impl Default for MctsArgs {
    fn default() -> Self {
        Self {
            temp_threshold: 15,
            num_mcts_sims: 25,
            //num_mcts_sims: 50,
            cpuct: 1.0,
        }
    }
}
