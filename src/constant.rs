pub const N: usize = 6;
pub const MOVE_LEN: usize = N * N + 1;
pub const BOARD_SIZE: usize = N * N;
//GPUメモリは足りてるが512を裁けていない。CPUは余裕があるようだが・・・
//pub const BATCH_SIZE: usize = 512;
pub const BATCH_SIZE: usize = 128;
pub const EPS: f32 = 1e-8;
