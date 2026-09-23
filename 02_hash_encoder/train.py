import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from sdf_model import HashMLP_SDF
from data import sample_batch


def loss_fn(model, points, targets):
    preds = model(points)
    return mx.mean((preds - targets) ** 2)


def train(model, num_steps=500, batch_size=1024, lr=1e-3, log_every=50):
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    history = []
    t0 = time.perf_counter()
    for step in range(num_steps):
        points, targets = sample_batch(batch_size)
        loss, grads = loss_and_grad(model, points, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % log_every == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - t0
            print(f"step {step:4d}  loss {loss.item():.6f}  elapsed {elapsed:.2f}s")
        history.append(float(loss.item()))

    return history


if __name__ == "__main__":
    model = HashMLP_SDF()
    train(model)
