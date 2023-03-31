//! Autograd implementation on single `Value`s.
pub mod nn;
mod value;

// Re-export
pub use value::{Value, ValueData};
