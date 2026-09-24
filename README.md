# MLX GPU Programming: Hash Grids and Neural SDFs

Small, research-motivated experiments in writing and differentiating custom GPU
kernels with [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. The
project builds on my co-authored work on S2MDF, which studies neural implicit
SDFs and constraints that prevent object surfaces from overlapping. That
research code uses CUDA; this repository is a focused MLX learning project,
not a translation of the CUDA system.

## What is here

### 01 — Projection kernel

[`01_projection_kernel/`](01_projection_kernel/) is an introductory custom
Metal kernel exercise. It fuses a multi-surface nearest-point projection into
one GPU dispatch. The kernel is forward-only.

An optional [C++/MLX extension](01_projection_kernel/cpp_extension/) exposes
the same operator through a Nanobind module and MLX's C++ custom-kernel API.
It is a compact example of packaging a Metal operator behind a C++ interface;
it remains forward-only and accepts float32 inputs.

### 02 — Hash-grid encoder and SDF regression

[`02_hash_encoder/`](02_hash_encoder/) contains a small multiresolution
hash-grid encoder, a coordinate MLP, a Fourier-feature baseline, and a
synthetic two-sphere SDF task.

The forward encoder launches one `mx.fast.metal_kernel` per resolution level.
Each thread handles one 3D point, hashes its eight grid corners, and
interpolates learned features. The default custom VJP uses a fused Metal
kernel to compute coordinate gradients and atomically accumulate feature-table
gradients. `backward_backend="mlx_scatter"` retains the previous MLX indexed-add
backward for comparison. A differentiable MLX reference is provided for small
correctness checks.

This is deliberately a learning-scale implementation, not a reproduction of
Instant-NGP or S2MDF. It uses a simple hash at every level. First-order
forward/backward checks pass on the recorded Apple GPU; higher-order gradients
needed for an Eikonal loss are not implemented. The current demo optimizes SDF
values only.

## Requirements

- Apple Silicon Mac with a working Metal-capable macOS installation
- Python 3.11 or newer
- Dependencies listed in [`requirements.txt`](requirements.txt)
- A real Metal device for the encoder and GPU correctness suite

MLX uses unified memory and supports custom Metal kernels on Apple GPUs. The
project does not have a CUDA or CPU fallback for its custom kernel path.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that MLX can see the GPU and compile the small starter kernel:

```bash
python sanity_check.py
```

## Run

Commands in this section assume the environment above is active.

Train the hash-grid model on the synthetic two-sphere target:

```bash
python 02_hash_encoder/train.py
```

Train and compare the hash-grid and Fourier models:

```bash
python 02_hash_encoder/benchmark.py
```

Profile encoder forward and forward-plus-backward time:

```bash
python 02_hash_encoder/profile_encoder.py
```

Train a model and render a center slice comparison:

```bash
python 02_hash_encoder/visualize.py
```

The visualization script writes `sdf_slice_comparison.png` in the repository
root.

Run the hash-grid forward and gradient correctness suite:

```bash
python -m pytest -q 02_hash_encoder/test_gradients.py
```

The suite compares both custom backward paths against a differentiable MLX
reference, checks repeated table-index accumulation, and checks coordinate
gradients against central finite differences away from grid boundaries. The
recorded Apple GPU run passed all three tests in 1.65 seconds.

Benchmark the reference, previous MLX-scatter backward, and fused Metal
backward. The script checks numerical agreement before timing, synchronizes
each sample, rotates measurement order, and reports medians and median
absolute deviations. Hardware and MLX metadata go to stderr; CSV results go
to stdout.

```bash
python 02_hash_encoder/benchmark_v2.py --batch-sizes 1 128 1024 4096 \
  --warmup 5 --repeats 30 \
  > 02_hash_encoder/benchmarks/macos-26.2-mlx-0.32.2.csv
```

### Recorded GPU results

Measured on macOS 26.2 with Python 3.11.3 and MLX 0.32.2 on an Apple GPU.
The machine model was not recorded. Values are medians from one run; timings
include Python and MLX dispatch/synchronization overhead, so they are not
GPU-only kernel timings.

| Batch | Metal forward | Forward speedup vs reference | Metal forward + backward | Speedup vs reference | Speedup vs MLX scatter |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.490 ms | 2.48× | 0.937 ms | 3.06× | 7.21× |
| 128 | 0.462 ms | 2.66× | 0.870 ms | 3.21× | 7.43× |
| 1,024 | 0.416 ms | 2.78× | 0.937 ms | 2.81× | 6.90× |
| 4,096 | 0.429 ms | 3.14× | 1.242 ms | 2.37× | 7.75× |

The full per-method timing data is in
[`02_hash_encoder/benchmarks/macos-26.2-mlx-0.32.2.csv`](02_hash_encoder/benchmarks/macos-26.2-mlx-0.32.2.csv).
The fused backward is faster than the old MLX-scatter path across these
batches; the exact speedup should be re-measured on other machines and MLX
versions.

## C++ extension exercise

Build the optional C++ projection extension in the same environment as MLX:

```bash
python -m pip install -e 01_projection_kernel/cpp_extension
python -m pytest -q 01_projection_kernel/cpp_extension/tests
```

The extension's interface, shape/dtype contract, and limitations are
documented in its [README](01_projection_kernel/cpp_extension/README.md).
Its build pins MLX 0.32.2 to match the recorded project environment; update
that pin and rerun its tests when upgrading MLX. The recorded Apple GPU run
passed both extension tests in 0.97 seconds.

Training and benchmark timings depend on the Mac, MLX version, thermal state,
and run configuration; treat them as local measurements rather than portable
performance claims.

## Known limitations and next steps

- Synthetic target only: two analytic spheres in the unit cube.
- Hashing is used even at resolutions where a dense grid could avoid
  collisions.
- The default backward uses float atomics for table-gradient collisions, so
  accumulation order can cause small floating-point differences.
- Coordinate gradients are first-order only. The forward and first-order
  gradient suite passes on the recorded Metal device; verify higher-order
  differentiation separately before adding Eikonal training.
- No checkpointing, dataset loader, mesh extraction, or multi-object
  separation constraints are implemented here.
- Results should be re-measured on the target machine before making speed or
  accuracy claims.

Suggested progression: add per-level profiling and inspect GPU dispatches in
Xcode's Metal debugger; then explore second-order gradients and an Eikonal
objective. The C++ extension is a starting point for deeper work with MLX's
custom primitive API and C++ VJP/JVP implementations.
