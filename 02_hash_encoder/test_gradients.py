"""Metal-device correctness checks for the hash-grid encoder."""

import numpy as np
import pytest

try:
    import mlx.core as mx
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"MLX GPU runtime unavailable: {exc}", allow_module_level=True)

from hash_encoder import HashEncoder
from hash_encoder_reference import hash_encode_reference


try:
    mx.default_device()
except Exception as exc:  # Headless/macOS sessions may have no Metal device.
    pytest.skip(f"MLX device unavailable: {exc}", allow_module_level=True)


RESOLUTION = 5
TABLE_SIZE = 97
FEATURE_DIM = 3


def _inputs():
    # Avoid cell boundaries so the central finite difference stays in one
    # piecewise-smooth interpolation cell.
    points = mx.array(
        [
            [0.173, 0.284, 0.391],
            [0.173, 0.284, 0.391],  # Deliberate table-gradient collision.
            [0.617, 0.739, 0.463],
            [0.831, 0.347, 0.683],
        ],
        dtype=mx.float32,
    )
    table = mx.random.normal(shape=(TABLE_SIZE, FEATURE_DIM)) * 0.1
    cotangent = mx.random.normal(shape=(points.shape[0], FEATURE_DIM))
    return points, table, cotangent


def _objective(encode, points, table, cotangent):
    return mx.sum(encode(points, table) * cotangent)


@pytest.mark.parametrize("backend", ["mlx_scatter", "metal"])
def test_forward_and_vjp_match_differentiable_reference(backend):
    points, table, cotangent = _inputs()
    encoder = HashEncoder(
        num_levels=1,
        base_resolution=RESOLUTION,
        growth_factor=1.0,
        table_size=TABLE_SIZE,
        feature_dim=FEATURE_DIM,
        backward_backend=backend,
    )
    encode = encoder._encode_fns[0]
    reference = lambda p, t: hash_encode_reference(p, t, RESOLUTION)

    actual = encode(points, table)
    expected = reference(points, table)
    grad_points, grad_table = mx.grad(
        lambda p, t: _objective(encode, p, t, cotangent), argnums=(0, 1)
    )(points, table)
    ref_grad_points, ref_grad_table = mx.grad(
        lambda p, t: _objective(reference, p, t, cotangent), argnums=(0, 1)
    )(points, table)
    mx.eval(actual, expected, grad_points, ref_grad_points, grad_table, ref_grad_table)

    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        np.array(grad_points), np.array(ref_grad_points), rtol=3e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        np.array(grad_table), np.array(ref_grad_table), rtol=3e-4, atol=2e-5
    )


def test_coordinate_gradient_matches_finite_difference():
    points, table, cotangent = _inputs()
    encoder = HashEncoder(
        num_levels=1,
        base_resolution=RESOLUTION,
        growth_factor=1.0,
        table_size=TABLE_SIZE,
        feature_dim=FEATURE_DIM,
        backward_backend="metal",
    )
    encode = encoder._encode_fns[0]
    grad_points = mx.grad(
        lambda p: _objective(encode, p, table, cotangent)
    )(points)
    mx.eval(grad_points)

    eps = 1e-3
    numerical = np.zeros(points.shape, dtype=np.float32)
    points_np = np.array(points)
    for row in range(points.shape[0]):
        for axis in range(3):
            plus, minus = points_np.copy(), points_np.copy()
            plus[row, axis] += eps
            minus[row, axis] -= eps
            loss_plus = _objective(encode, mx.array(plus), table, cotangent)
            loss_minus = _objective(encode, mx.array(minus), table, cotangent)
            mx.eval(loss_plus, loss_minus)
            numerical[row, axis] = (loss_plus.item() - loss_minus.item()) / (2 * eps)

    np.testing.assert_allclose(np.array(grad_points), numerical, rtol=2e-2, atol=2e-3)
