import time
import mlx.core as mx
from baseline import naive_projection, sphere_sdf 

def make_data(N, M=2):
    points = mx.random.uniform(low=-2, high=2, shape=(N, 3))
    centers = [mx.array([0.0, 0.0, 0.0]), mx.array([1.5, 0.0, 0.0])]
    radii = [1.0, 0.8]
    dists, grads = [], []
    for c, r in zip(centers, radii):
        d, g = sphere_sdf(points, c, r)
        dists.append(d)
        grads.append(g)
    return points, mx.stack(dists, axis=1), mx.stack(grads, axis=1)

def time_fn(fn, *args, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        out = fn(*args)
        mx.eval(out)
    start = time.perf_counter()
    for _ in range(n_runs):
        out = fn(*args)
        mx.eval(out)
    elapsed = time.perf_counter() - start
    return elapsed / n_runs

for N in [1_000, 10_000, 100_000, 1_000_000]:
    points, sdf_values, sdf_grads = make_data(N)
    avg_time = time_fn(naive_projection, points, sdf_values, sdf_grads)
    print(f"N={N:>9}  naive: {avg_time*1000:.4f} ms")