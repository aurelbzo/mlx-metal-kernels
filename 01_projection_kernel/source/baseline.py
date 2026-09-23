import mlx.core as mx

def naive_projection(points, sdf_values, sdf_grads):
    # points: (N, 3)
    # sdf_values: (N, M)      -- distance from each point to each of M surfaces
    # sdf_grads: (N, M, 3)    -- gradient of each SDF at each point
    abs_vals = mx.abs(sdf_values)
    idx = mx.argmin(abs_vals, axis=1)
    chosen_dist = mx.take_along_axis(sdf_values, idx[:, None], axis=1)
    chosen_grad = mx.take_along_axis(sdf_grads, idx[:, None, None], axis=1).squeeze(1)
    projected = points - chosen_dist * chosen_grad
    return projected

# --- synthetic test data: 2 spheres as toy SDFs ---
def sphere_sdf(points, center, radius):
    diff = points - center
    dist = mx.sqrt(mx.sum(diff**2, axis=-1)) - radius
    grad = diff / mx.sqrt(mx.sum(diff**2, axis=-1, keepdims=True))
    return dist, grad

N = 1000
points = mx.random.uniform(low=-2, high=2, shape=(N, 3))

centers = [mx.array([0.0, 0.0, 0.0]), mx.array([1.5, 0.0, 0.0])]
radii = [1.0, 0.8]

dists, grads = [], []
for c, r in zip(centers, radii):
    d, g = sphere_sdf(points, c, r)
    dists.append(d)
    grads.append(g)

sdf_values = mx.stack(dists, axis=1)      # (N, M)
sdf_grads = mx.stack(grads, axis=1)       # (N, M, 3)

projected = naive_projection(points, sdf_values, sdf_grads)
print(projected[:5])