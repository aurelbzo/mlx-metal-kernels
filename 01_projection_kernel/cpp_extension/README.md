# C++/MLX nearest-SDF projection

A small C++ extension exposing the fused nearest-surface projection as a
Python-callable MLX operation. The C++ binding calls `mx::fast::metal_kernel`
with an inline Metal kernel; Nanobind provides the Python module binding and
CMake builds the extension.

Given points `p` and multiple signed distances and gradients at each point,
the kernel chooses the surface with smallest absolute signed distance and
applies `p - d * grad`. The selected gradient is assumed to be a unit SDF
normal. This is a forward-only operator; it does not define an MLX VJP.

## Build

Use the same Python environment that has MLX installed. The build uses the
MLX CMake package, CMake, Nanobind, and AppleClang from Xcode Command Line
Tools. Build dependencies are declared in `pyproject.toml`.

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e 01_projection_kernel/cpp_extension
```

## Use

```python
from mlx_cpp_projection import nearest_surface_projection

projected = nearest_surface_projection(points, sdf_values, sdf_gradients)
```

Inputs must be `float32` MLX arrays with shapes `(N, 3)`, `(N, M)`, and
`(N, M, 3)`, where `N > 0` and `M > 0`. Inputs are made row-contiguous by
MLX's custom-kernel wrapper when needed.

## Check

```bash
python -m pytest -q 01_projection_kernel/cpp_extension/tests
```

The tests compare the extension result with an ordinary MLX reference and
check invalid-shape handling. The Metal kernel compiles on first evaluation,
so run the tests on a Metal-enabled Apple Silicon Mac.
