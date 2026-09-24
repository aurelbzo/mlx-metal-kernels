"""Compare the C++/Metal projection with an idiomatic MLX implementation.

Timings are end-to-end Python call plus MLX evaluation time. The first call is
excluded from measurement so MLX can compile/cache its Metal kernels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_cpp_projection import nearest_surface_projection


def mlx_projection(points, sdf_values, sdf_grads):
    """Reference using ordinary MLX array operations."""
    indices = mx.argmin(mx.abs(sdf_values), axis=1)
    rows = mx.arange(points.shape[0])
    distances = sdf_values[rows, indices]
    gradients = sdf_grads[rows, indices]
    return points - distances[:, None] * gradients


def gpu_model() -> str | None:
    """Read the GPU chipset name from macOS system_profiler, when available."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        displays = json.loads(result.stdout).get("SPDisplaysDataType", [])
        models = [display.get("sppci_model") for display in displays]
        models = [model for model in models if model]
        return "; ".join(models) if models else None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def make_inputs(batch_size: int, surface_count: int):
    points = mx.random.uniform(low=-1.0, high=1.0, shape=(batch_size, 3))
    sdf_values = mx.random.uniform(
        low=-1.0, high=1.0, shape=(batch_size, surface_count)
    )
    sdf_grads = mx.random.uniform(
        low=-1.0, high=1.0, shape=(batch_size, surface_count, 3)
    )
    return points, sdf_values, sdf_grads


def measure(function, args, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        result = function(*args)
        mx.eval(result)

    timings_ms = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function(*args)
        mx.eval(result)
        timings_ms.append((time.perf_counter_ns() - start) / 1e6)
    return timings_ms


def median_and_mad(values: list[float]) -> tuple[float, float]:
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return median, mad


def eval_sync(value) -> None:
    mx.eval(value)
    mx.synchronize()


def verify(reference, candidate) -> float:
    mx.eval(reference, candidate)
    mx.synchronize()
    difference = mx.max(mx.abs(reference - candidate)).item()
    if not np.isfinite(difference) or difference > 2e-6:
        raise AssertionError(f"C++ result differs from MLX reference: max_abs={difference}")
    return float(difference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[256, 4096, 65536])
    parser.add_argument("--surface-counts", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="write environment and benchmark settings as JSON",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats must be >= 1")
    if any(size < 1 for size in args.batch_sizes + args.surface_counts):
        parser.error("batch sizes and surface counts must be positive")

    metadata = {
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "platform": platform.platform(),
        "device": str(mx.default_device()),
        "gpu_model": gpu_model(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "timing_method": "interleaved order; perf_counter_ns around call + mx.eval + mx.synchronize",
        "statistics": "median and median absolute deviation",
        "methods": ["mlx_reference", "cpp_metal"],
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True), file=sys.stderr)
    mx.random.seed(args.seed)

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "batch_size",
            "surface_count",
            "method",
            "median_ms",
            "mad_ms",
            "points_per_second",
            "max_abs_error_vs_mlx",
            "speedup_vs_mlx_reference",
        ],
    )
    writer.writeheader()
    methods = {
        "mlx_reference": mlx_projection,
        "cpp_metal": nearest_surface_projection,
    }

    for batch_size in args.batch_sizes:
        for surface_count in args.surface_counts:
            inputs = make_inputs(batch_size, surface_count)
            reference = mlx_projection(*inputs)
            custom = nearest_surface_projection(*inputs)
            max_error = verify(reference, custom)

            samples = {name: [] for name in methods}
            # Rotate which implementation goes first to reduce ordering bias.
            order = list(methods.items())
            for iteration in range(args.repeats):
                offset = iteration % len(order)
                for name, function in order[offset:] + order[:offset]:
                    if iteration == 0:
                        for _ in range(args.warmup):
                            eval_sync(function(*inputs))
                    start = time.perf_counter_ns()
                    output = function(*inputs)
                    eval_sync(output)
                    samples[name].append((time.perf_counter_ns() - start) / 1e6)

            reference_median = median_and_mad(samples["mlx_reference"])[0]
            for name, timings in samples.items():
                median, mad = median_and_mad(timings)
                writer.writerow(
                    {
                        "batch_size": batch_size,
                        "surface_count": surface_count,
                        "method": name,
                        "median_ms": f"{median:.6f}",
                        "mad_ms": f"{mad:.6f}",
                        "points_per_second": f"{batch_size / (median / 1000):.1f}",
                        "max_abs_error_vs_mlx": f"{max_error:.9g}",
                        "speedup_vs_mlx_reference": f"{reference_median / median:.3f}",
                    }
                )
                sys.stdout.flush()


if __name__ == "__main__":
    main()
