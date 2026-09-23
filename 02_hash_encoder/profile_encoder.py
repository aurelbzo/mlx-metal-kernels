import time
import mlx.core as mx
import mlx.nn as nn

from sdf_model import HashMLP_SDF
from data import sample_batch

model = HashMLP_SDF()
points, targets = sample_batch(1024)

def loss_fn(model, points, targets):
    return mx.mean((model(points) - targets) ** 2)

# forward only (no backward)
mx.eval(model(points))  # warm-up
t0 = time.perf_counter()
for _ in range(50):
    out = model(points)
    mx.eval(out)
fwd_time = (time.perf_counter() - t0) / 50

# forward + backward
loss_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad(model, points, targets)
mx.eval(loss, grads)  # warm-up
t0 = time.perf_counter()
for _ in range(50):
    loss, grads = loss_and_grad(model, points, targets)
    mx.eval(loss, grads)
fwd_bwd_time = (time.perf_counter() - t0) / 50

print(f"forward only:      {fwd_time*1000:.3f} ms")
print(f"forward + backward: {fwd_bwd_time*1000:.3f} ms")
print(f"backward overhead:  {(fwd_bwd_time - fwd_time)*1000:.3f} ms  ({(fwd_bwd_time-fwd_time)/fwd_bwd_time*100:.1f}% of total)")