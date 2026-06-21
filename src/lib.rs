//! # ferrograd
//!
//! A small deep-learning framework written in **native Rust with zero
//! dependencies**. It provides a reverse-mode automatic differentiation
//! engine over 2-D tensors and a set of neural-network primitives built on
//! top of it.
//!
//! The pieces fit together the same way they do in PyTorch:
//!
//! 1. Build a model out of [`nn::Module`]s (e.g. [`nn::Sequential`]).
//! 2. Run a forward pass to get predictions, then compute a scalar loss
//!    (e.g. [`engine::Tensor::softmax_cross_entropy`]).
//! 3. Call [`engine::Tensor::backward`] to fill in gradients.
//! 4. Step an [`optim::Optimizer`] and zero the gradients.
//!
//! ## Quick start
//!
//! ```
//! use ferrograd::prelude::*;
//!
//! let mut rng = Rng::new(42);
//! // A 2 -> 16 -> 2 classifier with a ReLU hidden layer.
//! let model = Sequential::mlp(&[2, 16, 2], Activation::ReLU, Activation::None, &mut rng);
//!
//! let data = data::xor();
//! let x = data.x_tensor();
//! let mut opt = Adam::new(model.parameters(), 0.05);
//!
//! for _ in 0..200 {
//!     let logits = model.forward(&x);
//!     let loss = logits.softmax_cross_entropy(&data.y);
//!     opt.zero_grad();
//!     loss.backward();
//!     opt.step();
//! }
//!
//! let acc = data.accuracy(&model.forward(&x));
//! assert!(acc >= 0.75, "model should learn XOR, got accuracy {acc}");
//! ```

// The tensor ops (`add`, `sub`, `mul`) intentionally mirror PyTorch's method
// names rather than implementing `std::ops` traits, since they build graph
// nodes instead of returning plain values.
#![allow(clippy::should_implement_trait)]
// Index-based loops read more naturally than iterators for the explicit
// row/column arithmetic in the tensor kernels.
#![allow(clippy::needless_range_loop)]

pub mod data;
pub mod engine;
pub mod nn;
pub mod optim;
pub mod rng;

/// Commonly used types, re-exported for `use ferrograd::prelude::*;`.
pub mod prelude {
    pub use crate::data;
    pub use crate::engine::{Shape, Tensor};
    pub use crate::nn::{Activation, Dense, Linear, Module, Sequential};
    pub use crate::optim::{Adam, Optimizer, SGD};
    pub use crate::rng::Rng;
}
