//! Toy dataset generators and helpers for the examples and tests.

use crate::engine::Tensor;
use crate::rng::Rng;

/// A batch of features `x` with shape `(n, features)` and integer class
/// labels `y` of length `n`.
pub struct Dataset {
    pub x: Vec<f32>,
    pub y: Vec<usize>,
    pub n: usize,
    pub features: usize,
}

impl Dataset {
    /// The features as a `(n, features)` tensor (no gradient).
    pub fn x_tensor(&self) -> Tensor {
        Tensor::new(self.x.clone(), (self.n, self.features), false)
    }

    /// Classification accuracy of a `(n, classes)` logits tensor.
    pub fn accuracy(&self, logits: &Tensor) -> f32 {
        let data = logits.data();
        let (rows, cols) = logits.shape();
        assert_eq!(rows, self.n);
        let mut correct = 0;
        for i in 0..rows {
            let row = &data[i * cols..(i + 1) * cols];
            let mut best = 0;
            for j in 1..cols {
                if row[j] > row[best] {
                    best = j;
                }
            }
            if best == self.y[i] {
                correct += 1;
            }
        }
        correct as f32 / rows as f32
    }
}

/// The classic two-spirals dataset: `classes` interleaved spiral arms. A
/// linear model cannot separate it, so it is a good smoke test for a network
/// with a non-linear hidden layer.
pub fn spirals(points_per_class: usize, classes: usize, noise: f32, rng: &mut Rng) -> Dataset {
    let n = points_per_class * classes;
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for c in 0..classes {
        for i in 0..points_per_class {
            let r = i as f32 / points_per_class as f32; // radius 0..1
            let t = (c as f32 * 4.0) + 4.0 * r + rng.normal() * noise; // angle
            x.push(r * (t * 2.5).sin());
            x.push(r * (t * 2.5).cos());
            y.push(c);
        }
    }
    Dataset {
        x,
        y,
        n,
        features: 2,
    }
}

/// The XOR problem as a 4-row dataset.
pub fn xor() -> Dataset {
    Dataset {
        x: vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        y: vec![0, 1, 1, 0],
        n: 4,
        features: 2,
    }
}
