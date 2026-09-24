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
