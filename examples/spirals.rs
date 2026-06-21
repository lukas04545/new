//! Train a deeper MLP to classify the two-spirals dataset — a problem that is
//! impossible for a linear model but easy for a network with hidden layers.
//!
//! Run with: `cargo run --release --example spirals`

use ferrograd::prelude::*;

fn main() {
    let mut rng = Rng::new(7);

    // 3 interleaved spiral arms, 100 points each.
    let classes = 3;
    let train = data::spirals(100, classes, 0.15, &mut rng);
    let x = train.x_tensor();

    // 2 -> 64 -> 64 -> 3, ReLU hidden layers.
    let model = Sequential::mlp(
        &[2, 64, 64, classes],
        Activation::ReLU,
        Activation::None,
        &mut rng,
    );
    let mut opt = Adam::new(model.parameters(), 0.02).with_weight_decay(1e-4);

    println!(
        "Training a 2-64-64-{classes} MLP on {} spiral points...\n",
        train.n
    );
    for epoch in 0..=1000 {
        let logits = model.forward(&x);
        let loss = logits.softmax_cross_entropy(&train.y);

        opt.zero_grad();
        loss.backward();
        opt.step();

        if epoch % 100 == 0 {
            let acc = train.accuracy(&model.forward(&x));
            println!(
                "epoch {epoch:>4}  loss {:.4}  train accuracy {:.1}%",
                loss.item(),
                acc * 100.0
            );
        }
    }

    let acc = train.accuracy(&model.forward(&x));
    println!("\nFinal training accuracy: {:.1}%", acc * 100.0);
}
