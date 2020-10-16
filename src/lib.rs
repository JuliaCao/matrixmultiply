extern crate rawpointer;

mod dgemm;
mod ref_dgemm;

pub use dgemm::dgemm;
pub use ref_dgemm::ref_dgemm;
