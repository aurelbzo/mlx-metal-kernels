"""Small, differentiable MLX reference for one hash-grid level.

This implementation is intentionally expressed with regular MLX operations,
so MLX supplies the VJP. It is for correctness comparisons, not performance.
"""

import mlx.core as mx


_PRIMES = (1, 2654435761, 805459861)
_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in range(2)
    for dy in range(2)
    for dz in range(2)
)


def hash_encode_reference(points, table, resolution):
    """Encode ``(N, 3)`` points using a ``(T, F)`` feature table."""
    offsets = mx.array(_OFFSETS, dtype=mx.uint32)
    scaled = points * resolution
    base_float = mx.floor(scaled)
    frac = scaled - base_float
    base = base_float.astype(mx.uint32)
    corners = base[:, None, :] + offsets[None, :, :]

    x, y, z = corners[:, :, 0], corners[:, :, 1], corners[:, :, 2]
    h = mx.bitwise_xor(
        mx.bitwise_xor(x * _PRIMES[0], y * _PRIMES[1]),
        z * _PRIMES[2],
    )
    # Grid-cell selection and hashing are discrete. Gradients flow through
    # interpolation weights and table values, never through the indices.
    indices = mx.stop_gradient((h % table.shape[0]).astype(mx.int32))

    weights = mx.ones((points.shape[0], len(_OFFSETS)), dtype=points.dtype)
    for axis in range(3):
        f = frac[:, axis : axis + 1]
        is_upper = offsets[None, :, axis] == 1
        axis_weights = mx.where(is_upper, f, 1.0 - f)
        weights = weights * axis_weights

    corner_features = table[indices]
    return mx.sum(weights[:, :, None] * corner_features, axis=1)
