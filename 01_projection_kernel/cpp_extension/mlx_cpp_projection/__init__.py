"""MLX C++ extension for a fused nearest-SDF projection operator."""

# Load MLX's shared library before importing the extension module.
import mlx.core as _mx  # noqa: F401

from ._ext import nearest_surface_projection

__all__ = ["nearest_surface_projection"]
