import mlx.core as mx
import numpy as np
import pytest

from mlx_cpp_projection import nearest_surface_projection


def reference_projection(points, sdf_values, sdf_grads):
    indices = mx.argmin(mx.abs(sdf_values), axis=1)
    rows = mx.arange(points.shape[0])
    chosen_distance = sdf_values[rows, indices]
    chosen_gradient = sdf_grads[rows, indices]
    return points - chosen_distance[:, None] * chosen_gradient


def test_projection_matches_mlx_reference():
    points = mx.array(
        [[0.2, -0.1, 0.7], [1.0, 2.0, 3.0], [-0.5, 0.4, 0.1]],
        dtype=mx.float32,
    )
    sdf_values = mx.array(
        [[0.2, -0.05], [-0.4, 0.1], [0.08, -0.3]], dtype=mx.float32
    )
    sdf_grads = mx.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=mx.float32,
    )

    actual = nearest_surface_projection(points, sdf_values, sdf_grads)
    expected = reference_projection(points, sdf_values, sdf_grads)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-6, atol=1e-6)


def test_rejects_mismatched_shapes():
    points = mx.zeros((2, 3))
    sdf_values = mx.zeros((2, 2))
    sdf_grads = mx.zeros((2, 1, 3))
    with pytest.raises(ValueError, match="sdf_grads"):
        nearest_surface_projection(points, sdf_values, sdf_grads)
