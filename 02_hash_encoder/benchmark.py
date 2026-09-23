import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from sdf_model import HashMLP_SDF, FourierMLP_SDF
from data import sample_batch


def loss_fn(model, points, targets):
    preds = model(points)
    return mx.mean((preds - targets) ** 2)


def run(model, name, num_steps=500, batch_size=1024, lr=1e-3, log_every=100):
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    t0 = time.perf_counter()
    for step in range(num_steps):
        points, targets = sample_batch(batch_size)
        loss, grads = loss_and_grad(model, points, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(float(loss.item()))
        if step % log_every == 0 or step == num_steps - 1:
            print(f"[{name}] step {step:4d}  loss {loss.item():.6f}")
    total_time = time.perf_counter() - t0

    # held-out accuracy check
    test_points, test_targets = sample_batch(4096)
    preds = model(test_points)
    mx.eval(preds)
    test_mse = float(mx.mean((preds - test_targets) ** 2).item())

    print(f"[{name}] total train time: {total_time:.2f}s  final test MSE: {test_mse:.6f}\n")
    return {"name": name, "losses": losses, "train_time": total_time, "test_mse": test_mse}


if __name__ == "__main__":
    mx.random.seed(0)
    hash_model = HashMLP_SDF()
    hash_result = run(hash_model, "hash_encoder")

    mx.random.seed(0)
    fourier_model = FourierMLP_SDF()
    fourier_result = run(fourier_model, "fourier_baseline")

    print("=== Summary ===")
    for r in (hash_result, fourier_result):
        print(f"{r['name']:>16}: train_time={r['train_time']:.2f}s  test_mse={r['test_mse']:.6f}")
