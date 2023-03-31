//! Autograd implementation on `Tensor`s.
pub mod nn;
mod tensor;

// Re-export
pub use tensor::{Tensor, TensorData};
