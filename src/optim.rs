//! Gradient-descent optimizers that update parameters in place.

use crate::engine::Tensor;

/// Common interface for optimizers.
pub trait Optimizer {
    /// Apply one update step using the gradients currently stored on the
    /// parameters.
    fn step(&mut self);
    /// Reset all parameter gradients to zero before the next backward pass.
    fn zero_grad(&self);
}

/// Stochastic gradient descent with optional momentum and weight decay.
pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> SGD {
        let velocity = params.iter().map(|p| vec![0.0; p.data().len()]).collect();
        SGD {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity,
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> SGD {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> SGD {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (p, vel) in self.params.iter().zip(self.velocity.iter_mut()) {
            let mut t = p.borrow_mut();
            for i in 0..t.data.len() {
                let mut g = t.grad[i];
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * t.data[i];
                }
                if self.momentum != 0.0 {
                    vel[i] = self.momentum * vel[i] + g;
                    g = vel[i];
                }
                t.data[i] -= self.lr * g;
            }
        }
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

/// The Adam optimizer (adaptive moment estimation).
pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: u64,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Adam {
        let m = params.iter().map(|p| vec![0.0; p.data().len()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.data().len()]).collect();
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m,
            v,
            t: 0,
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Adam {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Adam {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (idx, p) in self.params.iter().enumerate() {
            let mut t = p.borrow_mut();
            for i in 0..t.data.len() {
                let mut g = t.grad[i];
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * t.data[i];
                }
                self.m[idx][i] = self.beta1 * self.m[idx][i] + (1.0 - self.beta1) * g;
                self.v[idx][i] = self.beta2 * self.v[idx][i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m[idx][i] / bc1;
                let v_hat = self.v[idx][i] / bc2;
                t.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}
