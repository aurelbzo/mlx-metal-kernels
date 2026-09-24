# C++/MLX nearest-SDF projection

A small C++ extension exposing the fused nearest-surface projection as a
Python-callable MLX operation. The C++ binding calls `mx::fast::metal_kernel`
with an inline Metal kernel; Nanobind provides the Python module binding and
CMake builds the extension.

Given points `p` and multiple signed distances and gradients at each point,
the operator chooses the surface with smallest absolute signed distance and
applies `p - d * grad`. The selected gradient is assumed to be a unit SDF
normal. The forward and custom first-order VJP are implemented in Metal
kernels dispatched from the C++ extension and registered as an MLX custom
function in Python.

This primitive is motivated by multi-SDF geometry: it moves a query point to
the nearest represented surface and can serve as a small building block in a
separation-constraint pipeline. It does not by itself enforce pairwise object
separation or reproduce S2MDF.

The selected surface is treated as constant during the VJP (the derivative of
the `argmin(abs(d))` decision is zero). At exact ties, the first surface wins,
matching `argmin`. For upstream gradient `u` and selected surface `(d, g)`, the
VJP is `grad_p = u`, `grad_d = -dot(u, g)`, and `grad_g = -d * u`; unselected
surfaces receive zero gradients. This is a piecewise first-order derivative;
the custom VJP does not implement higher-order derivatives.

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

The tests compare the forward result and the first-order VJP with ordinary MLX
references, and check invalid-shape handling. The Metal kernels compile on
first evaluation, so run the tests on a Metal-enabled Apple Silicon Mac.
The user-reported Apple M1 Pro run passed the VJP reference comparison.
Finite-difference coverage remains a useful follow-up, especially near cases
where two surfaces have similar absolute distances.

## Benchmark

Compare the C++/Metal operator with the equivalent MLX array expression over
different point counts and numbers of surfaces:

```bash
mkdir -p 01_projection_kernel/cpp_extension/benchmarks
python 01_projection_kernel/cpp_extension/benchmark_projection.py \
  --batch-sizes 256 4096 65536 --surface-counts 2 8 32 \
  --warmup 5 --repeats 50 \
  --metadata-output 01_projection_kernel/cpp_extension/benchmarks/projection-metadata.json \
  > 01_projection_kernel/cpp_extension/benchmarks/projection.csv
```

The CSV reports median latency, median absolute deviation (MAD), throughput,
maximum output error, and speedup. The JSON records Python/MLX/macOS versions,
MLX device, GPU model when macOS reports one, and benchmark settings. It checks
numerical agreement before timing, excludes compilation/warm-up, synchronizes
GPU work for every sample, and rotates which implementation runs first.
Timings include Python and MLX dispatch overhead and are end-to-end operator
measurements, not isolated Metal-kernel timings.

### Recorded results

Measured on an Apple M1 Pro with macOS 26.2, Python 3.11.3, MLX 0.32.2,
50 timed repetitions per method, and 5 warm-ups. The C++/Metal path was faster
in all nine cases; its speedup ranged from 1.24× to 4.11×. The largest gain
was at 65,536 points and 2 surfaces. Maximum absolute error was
`1.19e-7` in every case.

| Points | Surfaces | MLX median | C++/Metal median | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 2 | 0.338 ms | 0.243 ms | 1.39× |
| 256 | 8 | 0.335 ms | 0.246 ms | 1.36× |
| 256 | 32 | 0.312 ms | 0.252 ms | 1.24× |
| 4,096 | 2 | 0.325 ms | 0.202 ms | 1.61× |
| 4,096 | 8 | 0.369 ms | 0.239 ms | 1.54× |
| 4,096 | 32 | 0.390 ms | 0.288 ms | 1.35× |
| 65,536 | 2 | 1.061 ms | 0.258 ms | 4.11× |
| 65,536 | 8 | 1.108 ms | 0.312 ms | 3.55× |
| 65,536 | 32 | 1.195 ms | 0.554 ms | 2.16× |

Full per-runner medians, MADs, throughput, and numerical error are in
[`benchmarks/projection.csv`](benchmarks/projection.csv); environment details
are in [`benchmarks/projection-metadata.json`](benchmarks/projection-metadata.json).
These are measurements of this Mac and configuration, not portable performance
claims or isolated kernel timings.

The recorded timings cover the forward projection only. The VJP passes its
MLX-reference correctness check on Metal, but has not yet been benchmarked.
Extend the benchmark to measure forward-plus-backward before making a
training-performance claim.
