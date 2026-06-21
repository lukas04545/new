# ferrograd

A small **deep-learning framework written in native Rust** — built from
scratch on the standard library with **zero external dependencies**. No
`ndarray`, no `rand`, no BLAS, no PyTorch bindings. Just `std` and some
careful math.

It gives you the pieces you need to define, train, and evaluate neural
networks, organised the same way a mainstream framework is:

- A **reverse-mode automatic differentiation** engine over 2-D tensors.
- **Neural-network layers** (`Linear`, `Dense`, `Sequential`/MLP) and
  activations (ReLU, Tanh, Sigmoid).
- **Loss functions**: mean-squared error and a numerically stable, fused
  softmax cross-entropy.
- **Optimizers**: SGD (with momentum + weight decay) and Adam.
- A dependency-free PRNG and a couple of toy datasets to play with.

Everything is gradient-checked against finite differences in the test suite.

## Why?

Modern ML frameworks are huge and hide the interesting parts behind C++ and
CUDA. `ferrograd` is the opposite: every line — the computation graph, the
backward pass, the optimizers — is plain, readable Rust you can step through.
It is meant for **learning how autograd actually works** and as a compact,
hackable base for experiments.

## Quick start

```rust
use ferrograd::prelude::*;

fn main() {
    let mut rng = Rng::new(42);

    // A 2 -> 16 -> 2 classifier with a ReLU hidden layer.
    let model = Sequential::mlp(&[2, 16, 2], Activation::ReLU, Activation::None, &mut rng);

    let data = data::xor();
    let x = data.x_tensor();
    let mut opt = Adam::new(model.parameters(), 0.05);

    for _ in 0..200 {
        let logits = model.forward(&x);                  // forward pass
        let loss = logits.softmax_cross_entropy(&data.y); // scalar loss
        opt.zero_grad();
        loss.backward();                                  // autograd
        opt.step();                                       // update weights
    }

    let acc = data.accuracy(&model.forward(&x));
    println!("accuracy: {:.0}%", acc * 100.0);
}
```

## Run the examples

```bash
# Solve XOR with a tiny MLP (reaches 100% accuracy).
cargo run --release --example xor

# Classify the two-spirals dataset, impossible for a linear model
# (reaches ~98% accuracy).
cargo run --release --example spirals
```

## Run the tests

```bash
cargo test
```

The suite includes **gradient checks** (analytic gradients vs. central finite
differences for every operation) and **end-to-end training tests** that assert
the framework can actually learn XOR and the spirals.

## How it works

Each `Tensor` is an `Rc<RefCell<…>>` node in a computation graph. Running an
operation (`a.matmul(&b)`, `x.relu()`, …) produces a new tensor that records
the operation and a handle to its inputs. Because each node only points *back*
at its inputs, the graph is a leak-free DAG.

Calling `.backward()` on a scalar:

1. builds a topological ordering of the graph with a depth-first search,
2. seeds the output gradient with `1.0`, and
3. walks the nodes in reverse, letting each operation push gradients into its
   parents via the chain rule.

The backward rule for every operation is a plain `match` arm in
[`src/engine.rs`](src/engine.rs) — including broadcasting (so a `(1, n)` bias
accumulates gradients summed over the batch) and a fused softmax
cross-entropy whose gradient is the tidy `softmax - onehot`.

## Project layout

| File | Contents |
|------|----------|
| `src/engine.rs` | The `Tensor` type and the autograd engine (ops + backward). |
| `src/nn.rs`     | `Module` trait, `Linear`, `Dense`, `Sequential`/MLP, activations. |
| `src/optim.rs`  | `SGD` (momentum, weight decay) and `Adam`. |
| `src/rng.rs`    | `xoshiro256**` PRNG for weight init and data generation. |
| `src/data.rs`   | Toy datasets (XOR, two-spirals) and accuracy helper. |
| `examples/`     | Runnable training demos. |
| `tests/`        | Gradient checks and training tests. |

## Limitations

This is a teaching-grade framework, not a production one. Tensors are 2-D
only, kernels are straightforward triple-loops (no SIMD/threads/GPU), and
there is no model serialization. The trade-off is clarity: you can read the
whole thing in an afternoon.

## License

MIT
