import mlx.core as mx


# Two spheres, positioned so the domain [0,1]^3 fully contains them with margin.
_CENTERS = [mx.array([0.35, 0.5, 0.5]), mx.array([0.65, 0.5, 0.5])]
_RADII = [0.22, 0.18]


def sphere_sdf(points, center, radius):
    diff = points - center
    return mx.sqrt(mx.sum(diff**2, axis=-1)) - radius


def ground_truth_sdf(points):
    """points: (N, 3) in [0,1]^3. Returns (N,) signed distance to the
    union of two spheres (hard min — fine as a regression target)."""
    d0 = sphere_sdf(points, _CENTERS[0], _RADII[0])
    d1 = sphere_sdf(points, _CENTERS[1], _RADII[1])
    return mx.minimum(d0, d1)


def sample_batch(batch_size):
    points = mx.random.uniform(low=0.0, high=1.0, shape=(batch_size, 3))
    targets = ground_truth_sdf(points)
    return points, targets
