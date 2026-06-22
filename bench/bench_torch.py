"""PyTorch baseline matching examples/bench.rs: train a 2-64-64-3 MLP on the
two-spirals dataset for a fixed number of epochs (CPU), and report timing.

This is the "AI framework in Python" comparison point.
"""

import math
import time

import torch
import torch.nn as nn

EPOCHS = 1000
CLASSES = 3
POINTS_PER_CLASS = 100

torch.manual_seed(7)
torch.set_num_threads(torch.get_num_threads())  # use default thread count


def make_spirals():
    xs, ys = [], []
    for c in range(CLASSES):
        for i in range(POINTS_PER_CLASS):
            r = i / POINTS_PER_CLASS
            t = c * 4.0 + 4.0 * r + torch.randn(1).item() * 0.15
            xs.append([r * math.sin(t * 2.5), r * math.cos(t * 2.5)])
            ys.append(c)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def main():
    x, y = make_spirals()
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, CLASSES),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Warm up.
    opt.zero_grad()
    loss_fn(model(x), y).backward()
    opt.step()

    start = time.perf_counter()
    last_loss = 0.0
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        last_loss = loss.item()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()

    print("PyTorch (Python)")
    print(f"  epochs:        {EPOCHS}")
    print(f"  total time:    {elapsed:.3f} s")
    print(f"  per epoch:     {elapsed * 1000.0 / EPOCHS:.3f} ms")
    print(f"  final loss:    {last_loss:.4f}")
    print(f"  final acc:     {acc * 100.0:.1f}%")


if __name__ == "__main__":
    main()
