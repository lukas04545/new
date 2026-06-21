//! Train a small MLP to solve XOR — the "hello world" of neural networks.
//!
//! Run with: `cargo run --release --example xor`

use ferrograd::prelude::*;

fn main() {
    let mut rng = Rng::new(1);
    let data = data::xor();
    let x = data.x_tensor();

    // 2 inputs -> 8 hidden (tanh) -> 2 class logits.
    let model = Sequential::mlp(&[2, 8, 2], Activation::Tanh, Activation::None, &mut rng);
    let mut opt = Adam::new(model.parameters(), 0.05);

    println!("Training a 2-8-2 MLP on XOR...\n");
    for epoch in 0..=400 {
        let logits = model.forward(&x);
        let loss = logits.softmax_cross_entropy(&data.y);

        opt.zero_grad();
        loss.backward();
        opt.step();

        if epoch % 50 == 0 {
            let acc = data.accuracy(&model.forward(&x));
            println!(
                "epoch {epoch:>4}  loss {:.4}  accuracy {:.0}%",
                loss.item(),
                acc * 100.0
            );
        }
    }

    println!("\nFinal predictions:");
    let logits = model.forward(&x);
    let out = logits.data();
    for i in 0..data.n {
        let row = &out[i * 2..i * 2 + 2];
        let pred = if row[1] > row[0] { 1 } else { 0 };
        println!(
            "  [{}, {}] -> {pred}  (target {})",
            data.x[i * 2],
            data.x[i * 2 + 1],
            data.y[i]
        );
    }
}
