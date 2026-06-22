//! Benchmark: time training the spirals MLP for a fixed number of epochs.
//! Prints wall-clock time so it can be compared against the Python baselines
//! in `bench/`.
//!
//! Run with: `cargo run --release --example bench`

use ferrograd::prelude::*;
use std::time::Instant;

fn main() {
    let epochs = 1000;
    let mut rng = Rng::new(7);
    let classes = 3;
    let train = data::spirals(100, classes, 0.15, &mut rng);
    let x = train.x_tensor();

    let model = Sequential::mlp(
        &[2, 64, 64, classes],
        Activation::ReLU,
        Activation::None,
        &mut rng,
    );
    let mut opt = Adam::new(model.parameters(), 0.02).with_weight_decay(1e-4);

    // Warm up one step (caches, allocator) before timing.
    {
        let loss = model.forward(&x).softmax_cross_entropy(&train.y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    let start = Instant::now();
    let mut last_loss = 0.0;
    for _ in 0..epochs {
        let logits = model.forward(&x);
        let loss = logits.softmax_cross_entropy(&train.y);
        opt.zero_grad();
        loss.backward();
        opt.step();
        last_loss = loss.item();
    }
    let elapsed = start.elapsed();

    let acc = train.accuracy(&model.forward(&x));
    println!("ferrograd (native Rust)");
    println!("  epochs:        {epochs}");
    println!("  total time:    {:.3} s", elapsed.as_secs_f64());
    println!(
        "  per epoch:     {:.3} ms",
        elapsed.as_secs_f64() * 1000.0 / epochs as f64
    );
    println!("  final loss:    {last_loss:.4}");
    println!("  final acc:     {:.1}%", acc * 100.0);
}
