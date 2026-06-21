//! End-to-end tests: the framework should actually be able to learn.

use ferrograd::prelude::*;

#[test]
fn learns_xor() {
    let mut rng = Rng::new(0);
    let data = data::xor();
    let x = data.x_tensor();

    let model = Sequential::mlp(&[2, 8, 2], Activation::Tanh, Activation::None, &mut rng);
    let mut opt = Adam::new(model.parameters(), 0.05);

    for _ in 0..500 {
        let logits = model.forward(&x);
        let loss = logits.softmax_cross_entropy(&data.y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    let acc = data.accuracy(&model.forward(&x));
    assert!(acc >= 0.99, "XOR should be solved exactly, got {acc}");
}

#[test]
fn learns_spirals_above_chance() {
    let mut rng = Rng::new(3);
    let classes = 3;
    let train = data::spirals(80, classes, 0.1, &mut rng);
    let x = train.x_tensor();

    let model = Sequential::mlp(
        &[2, 32, 32, classes],
        Activation::ReLU,
        Activation::None,
        &mut rng,
    );
    let mut opt = Adam::new(model.parameters(), 0.02);

    for _ in 0..600 {
        let logits = model.forward(&x);
        let loss = logits.softmax_cross_entropy(&train.y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    let acc = train.accuracy(&model.forward(&x));
    // Chance is ~33%; a working network should comfortably exceed it.
    assert!(acc >= 0.85, "spirals accuracy too low: {acc}");
}

#[test]
fn sgd_reduces_loss() {
    let mut rng = Rng::new(5);
    let data = data::xor();
    let x = data.x_tensor();
    let model = Sequential::mlp(&[2, 8, 2], Activation::Tanh, Activation::None, &mut rng);

    let first = model.forward(&x).softmax_cross_entropy(&data.y).item();

    let mut opt = SGD::new(model.parameters(), 0.5).with_momentum(0.9);
    for _ in 0..300 {
        let loss = model.forward(&x).softmax_cross_entropy(&data.y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    let last = model.forward(&x).softmax_cross_entropy(&data.y).item();
    assert!(
        last < first * 0.5,
        "SGD failed to reduce loss: {first} -> {last}"
    );
}
