//! Correctness tests for the autograd engine.
//!
//! The key test verifies analytic gradients against numerical (finite
//! difference) gradients, which catches mistakes in any backward rule.

use ferrograd::prelude::*;

/// Build a scalar output from a single input tensor, used by the gradient
/// checker below.
fn run_graph(input: &Tensor, build: &dyn Fn(&Tensor) -> Tensor) -> f32 {
    build(input).item()
}

/// Compare analytic gradients (from `backward`) against central finite
/// differences for an arbitrary scalar-valued function of one tensor.
fn check_grad(input_data: Vec<f32>, shape: Shape, build: impl Fn(&Tensor) -> Tensor) {
    let input = Tensor::new(input_data.clone(), shape, true);
    let out = build(&input);
    assert_eq!(out.shape(), (1, 1), "function must return a scalar");
    out.backward();
    let analytic = input.grad();

    let eps = 1e-3;
    for i in 0..input_data.len() {
        let mut plus = input_data.clone();
        plus[i] += eps;
        let mut minus = input_data.clone();
        minus[i] -= eps;

        let fp = run_graph(&Tensor::new(plus, shape, false), &build);
        let fm = run_graph(&Tensor::new(minus, shape, false), &build);
        let numeric = (fp - fm) / (2.0 * eps);

        let diff = (numeric - analytic[i]).abs();
        let tol = 1e-2 * (1.0 + numeric.abs());
        assert!(
            diff < tol,
            "gradient mismatch at index {i}: analytic={}, numeric={}, diff={}",
            analytic[i],
            numeric,
            diff
        );
    }
}

#[test]
fn grad_add_mul_chain() {
    // f(x) = mean((x * x) + x)
    check_grad(vec![0.5, -1.2, 2.0, 0.1], (2, 2), |x| {
        x.mul(x).add(x).mean()
    });
}

#[test]
fn grad_sub_and_scalar() {
    // f(x) = sum((x - 0.3) * 2.0)
    check_grad(vec![1.0, 2.0, 3.0], (1, 3), |x| {
        x.scalar_add(-0.3).scalar_mul(2.0).sum()
    });
}

#[test]
fn grad_matmul() {
    // f(x) = sum(x @ W) where W is a constant.
    let w = Tensor::new(vec![1.0, -2.0, 0.5, 3.0, 0.0, -1.0], (2, 3), false);
    check_grad(vec![0.4, -0.7], (1, 2), move |x| x.matmul(&w).sum());
}

#[test]
fn grad_activations() {
    check_grad(vec![0.5, -0.5, 1.5, -1.5], (2, 2), |x| x.relu().mean());
    check_grad(vec![0.5, -0.5, 1.5, -1.5], (2, 2), |x| x.tanh().sum());
    check_grad(vec![0.5, -0.5, 1.5, -1.5], (2, 2), |x| x.sigmoid().sum());
    check_grad(vec![0.2, -0.4, 0.7, 0.1], (2, 2), |x| x.exp().mean());
}

#[test]
fn grad_softmax_cross_entropy() {
    // f(logits) = softmax_cross_entropy(logits, targets)
    let targets = vec![0usize, 2usize];
    check_grad(vec![1.0, 2.0, 0.5, -1.0, 0.3, 2.2], (2, 3), move |x| {
        x.softmax_cross_entropy(&targets)
    });
}

#[test]
fn grad_mse_loss() {
    let target = Tensor::new(vec![1.0, 0.0, -1.0, 0.5], (2, 2), false);
    check_grad(vec![0.9, 0.2, -0.8, 0.0], (2, 2), move |x| {
        x.mse_loss(&target)
    });
}

#[test]
fn grad_accumulates_for_reused_node() {
    // x used twice: f(x) = sum(x * x). df/dx = 2x.
    let x = Tensor::new(vec![3.0, -2.0], (1, 2), true);
    let out = x.mul(&x).sum();
    out.backward();
    let g = x.grad();
    assert!((g[0] - 6.0).abs() < 1e-4, "got {}", g[0]);
    assert!((g[1] - (-4.0)).abs() < 1e-4, "got {}", g[1]);
}

#[test]
fn broadcast_bias_gradient() {
    // y = x + b where b is (1, 2) broadcast over 3 rows. dL/db sums the rows.
    let x = Tensor::new(vec![1.0; 6], (3, 2), false);
    let b = Tensor::new(vec![0.0, 0.0], (1, 2), true);
    let out = x.add(&b).sum();
    out.backward();
    let g = b.grad();
    assert!((g[0] - 3.0).abs() < 1e-5, "got {}", g[0]);
    assert!((g[1] - 3.0).abs() < 1e-5, "got {}", g[1]);
}
