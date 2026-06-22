"""Hand-rolled NumPy baseline matching examples/bench.rs: the same 2-64-64-3
MLP trained on two-spirals with a from-scratch backward pass and Adam.

This is the fairest comparison to ferrograd, since both implement the math
themselves rather than dispatching to a precompiled tensor library.
"""

import math
import time

import numpy as np

EPOCHS = 1000
CLASSES = 3
POINTS_PER_CLASS = 100

rng = np.random.default_rng(7)


def make_spirals():
    n = POINTS_PER_CLASS * CLASSES
    x = np.zeros((n, 2), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    k = 0
    for c in range(CLASSES):
        for i in range(POINTS_PER_CLASS):
            r = i / POINTS_PER_CLASS
            t = c * 4.0 + 4.0 * r + rng.standard_normal() * 0.15
            x[k] = [r * math.sin(t * 2.5), r * math.cos(t * 2.5)]
            y[k] = c
            k += 1
    return x, y


def he(shape):
    return (rng.standard_normal(shape) * math.sqrt(2.0 / shape[0])).astype(np.float32)


def main():
    x, y = make_spirals()
    n = x.shape[0]

    sizes = [(2, 64), (64, 64), (64, CLASSES)]
    W = [he(s) for s in sizes]
    b = [np.zeros((1, s[1]), dtype=np.float32) for s in sizes]

    lr, beta1, beta2, eps, wd = 0.02, 0.9, 0.999, 1e-8, 1e-4
    mW = [np.zeros_like(w) for w in W]
    vW = [np.zeros_like(w) for w in W]
    mb = [np.zeros_like(bb) for bb in b]
    vb = [np.zeros_like(bb) for bb in b]

    def forward(x):
        a0 = x @ W[0] + b[0]
        h0 = np.maximum(a0, 0)
        a1 = h0 @ W[1] + b[1]
        h1 = np.maximum(a1, 0)
        logits = h1 @ W[2] + b[2]
        return a0, h0, a1, h1, logits

    def step(t):
        a0, h0, a1, h1, logits = forward(x)
        shifted = logits - logits.max(1, keepdims=True)
        exp = np.exp(shifted)
        sm = exp / exp.sum(1, keepdims=True)
        loss = -np.log(sm[np.arange(n), y] + 1e-12).mean()

        dlogits = sm.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        grads_w = [None] * 3
        grads_b = [None] * 3
        grads_w[2] = h1.T @ dlogits
        grads_b[2] = dlogits.sum(0, keepdims=True)
        dh1 = dlogits @ W[2].T
        da1 = dh1 * (a1 > 0)
        grads_w[1] = h0.T @ da1
        grads_b[1] = da1.sum(0, keepdims=True)
        dh0 = da1 @ W[1].T
        da0 = dh0 * (a0 > 0)
        grads_w[0] = x.T @ da0
        grads_b[0] = da0.sum(0, keepdims=True)

        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        for i in range(3):
            for (param, grad, m, v) in (
                (W[i], grads_w[i], mW[i], vW[i]),
                (b[i], grads_b[i], mb[i], vb[i]),
            ):
                grad = grad + wd * param
                m[...] = beta1 * m + (1 - beta1) * grad
                v[...] = beta2 * v + (1 - beta2) * grad * grad
                param -= lr * (m / bc1) / (np.sqrt(v / bc2) + eps)
        return loss

    # Warm up.
    step(1)

    start = time.perf_counter()
    last_loss = 0.0
    for t in range(2, EPOCHS + 2):
        last_loss = step(t)
    elapsed = time.perf_counter() - start

    *_, logits = forward(x)
    acc = (logits.argmax(1) == y).mean()

    print("NumPy hand-rolled (Python)")
    print(f"  epochs:        {EPOCHS}")
    print(f"  total time:    {elapsed:.3f} s")
    print(f"  per epoch:     {elapsed * 1000.0 / EPOCHS:.3f} ms")
    print(f"  final loss:    {last_loss:.4f}")
    print(f"  final acc:     {acc * 100.0:.1f}%")


if __name__ == "__main__":
    main()
