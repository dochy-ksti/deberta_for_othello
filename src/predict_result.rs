use crate::{action::Pi, c_array::CArray};

#[derive(Debug)]
pub struct PredictResult {
    pub win_rate: f32,
    pub action_probs: Pi,
}

impl PredictResult{
	pub fn convert_from_carrays(pis : &CArray<f32>, winrates: &CArray<f32>) -> Vec<PredictResult>{
		if winrates.size0() != pis.size0(){
			panic!("winrates and pis sizes must be the same");
		}

		let len = winrates.size0();

		let mut vec = Vec::with_capacity(len);

		for i in 0..winrates.size0(){
			vec.push(PredictResult{
				action_probs: Pi::new(pis.ref2(i)),
				win_rate: winrates.get2(i, 0)
			});
		}

		vec
	}
}