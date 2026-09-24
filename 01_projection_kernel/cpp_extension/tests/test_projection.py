import mlx.core as mx
import numpy as np
import pytest

from mlx_cpp_projection import nearest_surface_projection


def reference_projection(points, sdf_values, sdf_grads):
    # Surface selection is piecewise constant; the custom VJP differentiates
    # through the selected distance and normal, not through the argmin index.
    indices = mx.stop_gradient(mx.argmin(mx.abs(sdf_values), axis=1))
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


def test_custom_vjp_matches_mlx_reference():
    points = mx.array([[0.2, -0.1, 0.7], [-0.5, 0.4, 0.1]], dtype=mx.float32)
    sdf_values = mx.array(
        [[0.2, -0.05, 0.7], [-0.4, 0.1, 0.3]], dtype=mx.float32
    )
    sdf_grads = mx.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=mx.float32,
    )
    upstream = mx.array([[0.3, -0.8, 0.2], [-0.6, 0.5, 0.9]], dtype=mx.float32)

    def weighted_output_sum(fn, p, values, grads):
        return mx.sum(fn(p, values, grads) * upstream)

    _, expected = mx.value_and_grad(
        lambda p, values, grads: weighted_output_sum(
            reference_projection, p, values, grads
        ),
        argnums=(0, 1, 2),
    )(points, sdf_values, sdf_grads)
    _, actual = mx.value_and_grad(
        lambda p, values, grads: weighted_output_sum(
            nearest_surface_projection, p, values, grads
        ),
        argnums=(0, 1, 2),
    )(points, sdf_values, sdf_grads)
    mx.eval(*expected, *actual)

    for actual_grad, expected_grad, name in zip(
        actual, expected, ("points", "sdf_values", "sdf_grads")
    ):
        np.testing.assert_allclose(
            np.array(actual_grad),
            np.array(expected_grad),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"custom VJP mismatch for {name}",
        )
