"""MLX C++ extension for a fused nearest-SDF projection operator."""

# Load MLX's shared library before importing the extension module.
import mlx.core as _mx  # noqa: F401

from ._ext import (
    _nearest_surface_projection,
    _nearest_surface_projection_vjp,
)


@_mx.custom_function
def nearest_surface_projection(points, sdf_values, sdf_grads):
    """Project each point to its nearest SDF, with a custom first-order VJP."""
    return _nearest_surface_projection(points, sdf_values, sdf_grads)


@nearest_surface_projection.vjp
def _nearest_surface_projection_custom_vjp(primals, cotangent, output):
    points, sdf_values, sdf_grads = primals
    return tuple(
        _nearest_surface_projection_vjp(points, sdf_values, sdf_grads, cotangent)
    )

__all__ = ["nearest_surface_projection"]
