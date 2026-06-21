//! Neural-network building blocks: parameterised layers and a sequential
//! container, built on top of the autograd [`Tensor`](crate::engine::Tensor).

use crate::engine::Tensor;
use crate::rng::Rng;

/// A non-linearity that can be applied between layers.
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    /// Identity (no activation).
    None,
}

impl Activation {
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => x.relu(),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::None => x.clone(),
        }
    }
}

/// Anything that transforms an input tensor and exposes learnable parameters.
pub trait Module {
    /// Forward pass.
    fn forward(&self, x: &Tensor) -> Tensor;
    /// The learnable parameters, for the optimizer to update.
    fn parameters(&self) -> Vec<Tensor>;
}

/// A fully connected (dense / affine) layer: `y = x @ W + b`.
pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
}

impl Linear {
    /// Create a layer with He (Kaiming) initialization, which keeps the
    /// variance of activations stable for ReLU-style networks.
    pub fn new(in_features: usize, out_features: usize, rng: &mut Rng) -> Linear {
        let std = (2.0 / in_features as f32).sqrt();
        let w_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.normal() * std)
            .collect();
        let w = Tensor::new(w_data, (in_features, out_features), true);
        let b = Tensor::zeros((1, out_features), true);
        Linear { w, b }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        // (batch, in) @ (in, out) -> (batch, out), then broadcast-add bias.
        x.matmul(&self.w).add(&self.b)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.w.clone(), self.b.clone()]
    }
}

/// One `Linear` layer followed by an activation.
pub struct Dense {
    pub linear: Linear,
    pub activation: Activation,
}

impl Dense {
    pub fn new(
        in_features: usize,
        out_features: usize,
        activation: Activation,
        rng: &mut Rng,
    ) -> Dense {
        Dense {
            linear: Linear::new(in_features, out_features, rng),
            activation,
        }
    }
}

impl Module for Dense {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.activation.apply(&self.linear.forward(x))
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.linear.parameters()
    }
}

/// A stack of modules applied in order — a classic multi-layer perceptron.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential { layers: Vec::new() }
    }

    /// Add a layer, returning `self` for fluent construction.
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Sequential {
        self.layers.push(Box::new(layer));
        self
    }

    /// Build a plain MLP from a list of layer sizes and a hidden activation.
    ///
    /// `sizes = [in, h1, h2, ..., out]`. The hidden layers use `hidden`; the
    /// output layer uses `output`.
    pub fn mlp(
        sizes: &[usize],
        hidden: Activation,
        output: Activation,
        rng: &mut Rng,
    ) -> Sequential {
        assert!(sizes.len() >= 2, "need at least an input and output size");
        let mut model = Sequential::new();
        for i in 0..sizes.len() - 1 {
            let act = if i == sizes.len() - 2 { output } else { hidden };
            model = model.add(Dense::new(sizes[i], sizes[i + 1], act, rng));
        }
        model
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Sequential::new()
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
