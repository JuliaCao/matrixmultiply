extern crate rawpointer;

mod dgemm;
pub mod flop_measurement;
mod ref_dgemm;

pub use dgemm::dgemm;
pub use ref_dgemm::ref_dgemm;
