"""Correctness-checked, synchronized benchmark for hash-grid VJPs.

Run from the repository root and redirect stdout to a CSV file. Metadata is
written to stderr so the CSV remains directly usable by pandas/Numbers.
"""

import argparse
import csv
import importlib.metadata
import platform
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from hash_encoder import HashEncoder
from hash_encoder_reference import hash_encode_reference


LEVELS = 4
TABLE_SIZE = 2**12
FEATURE_DIM = 2
BASE_RESOLUTION = 4


def _build_encoders():
    encoders = {
        backend: HashEncoder(
            num_levels=LEVELS,
            base_resolution=BASE_RESOLUTION,
            growth_factor=2.0,
            table_size=TABLE_SIZE,
            feature_dim=FEATURE_DIM,
            backward_backend=backend,
        )
        for backend in ("mlx_scatter", "metal")
    }

    def reference(points, tables):
        return mx.concatenate(
            [
                hash_encode_reference(points, tables[level], resolution)
                for level, resolution in enumerate(
                    int(round(BASE_RESOLUTION * (2.0**level)))
                    for level in range(LEVELS)
                )
            ],
            axis=-1,
        )

    def from_encoder(encoder):
        def encode(points, tables):
            return mx.concatenate(
                [
                    encoder._encode_fns[level](points, tables[level])
                    for level in range(LEVELS)
                ],
                axis=-1,
            )

        return encode

    return {
        "reference": reference,
        "mlx_scatter": from_encoder(encoders["mlx_scatter"]),
        "metal": from_encoder(encoders["metal"]),
    }


def _median_and_mad(values):
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return median, mad


def _eval_sync(value):
    arrays = []

    def collect(item):
        if isinstance(item, (tuple, list)):
            for child in item:
                collect(child)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        else:
            arrays.append(item)

    collect(value)
    mx.eval(*arrays)
    mx.synchronize()


def _verify(encoders, points, tables):
    reference = encoders["reference"]
    ref_output = reference(points, tables)
    ref_loss = lambda p, t: mx.mean(mx.square(reference(p, t)))
    ref_value, ref_grads = mx.value_and_grad(ref_loss, argnums=(0, 1))(points, tables)
    _eval_sync((ref_output, ref_value, ref_grads))

    for name, encode in encoders.items():
        if name == "reference":
            continue
        output = encode(points, tables)
        loss_fn = lambda p, t: mx.mean(mx.square(encode(p, t)))
        value, grads = mx.value_and_grad(loss_fn, argnums=(0, 1))(points, tables)
        _eval_sync((output, value, grads))
        np.testing.assert_allclose(
            np.array(output), np.array(ref_output), rtol=3e-5, atol=3e-6,
            err_msg=f"{name}: forward mismatch",
        )
        np.testing.assert_allclose(
            np.array(value), np.array(ref_value), rtol=3e-5, atol=3e-6,
            err_msg=f"{name}: loss mismatch",
        )
        for grad, ref_grad, label in zip(grads, ref_grads, ("point", "table")):
            np.testing.assert_allclose(
                np.array(grad), np.array(ref_grad), rtol=5e-4, atol=3e-5,
                err_msg=f"{name}: {label} gradient mismatch",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 128, 1024, 4096])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or any(n < 1 for n in args.batch_sizes):
        parser.error("batch sizes and repeats must be positive; warmup cannot be negative")

    print(f"python={platform.python_version()} mlx={importlib.metadata.version('mlx')}", file=sys.stderr)
    print(f"platform={platform.platform()} device={mx.default_device()}", file=sys.stderr)
    print(
        f"levels={LEVELS} table_size={TABLE_SIZE} feature_dim={FEATURE_DIM} "
        f"warmup={args.warmup} repeats={args.repeats}",
        file=sys.stderr,
    )

    mx.random.seed(7)
    encoders = _build_encoders()
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "batch_size", "method", "forward_median_ms", "forward_mad_ms",
            "forward_backward_median_ms", "forward_backward_mad_ms",
            "forward_speedup_vs_reference", "forward_backward_speedup_vs_reference",
        ],
    )
    writer.writeheader()

    for batch_size in args.batch_sizes:
        points = mx.random.uniform(low=0.01, high=0.99, shape=(batch_size, 3))
        tables = mx.random.normal(shape=(LEVELS, TABLE_SIZE, FEATURE_DIM)) * 1e-4
        _eval_sync((points, tables))
        _verify(encoders, points, tables)

        forward_results = {}
        backward_results = {}
        names = ["reference", "mlx_scatter", "metal"]
        for repeat in range(args.repeats):
            # Rotate measurement order to reduce systematic thermal/order bias.
            rotation = repeat % len(names)
            order = names[rotation:] + names[:rotation]
            for name in order:
                encode = encoders[name]
                forward_fn = lambda e=encode: e(points, tables)
                loss_fn = lambda p, t, e=encode: mx.mean(mx.square(e(p, t)))
                value_and_grad = mx.value_and_grad(loss_fn, argnums=(0, 1))
                backward_fn = lambda vg=value_and_grad: vg(points, tables)

                if repeat == 0:
                    # Compile/warm each path outside timed samples.
                    for _ in range(args.warmup):
                        _eval_sync(forward_fn())
                        _eval_sync(backward_fn())
                start = time.perf_counter()
                _eval_sync(forward_fn())
                forward_results.setdefault(name, []).append(
                    (time.perf_counter() - start) * 1e3
                )
                start = time.perf_counter()
                _eval_sync(backward_fn())
                backward_results.setdefault(name, []).append(
                    (time.perf_counter() - start) * 1e3
                )

        ref_forward = statistics.median(forward_results["reference"])
        ref_backward = statistics.median(backward_results["reference"])
        for name in names:
            f_median, f_mad = _median_and_mad(forward_results[name])
            b_median, b_mad = _median_and_mad(backward_results[name])
            writer.writerow(
                {
                    "batch_size": batch_size,
                    "method": name,
                    "forward_median_ms": f"{f_median:.6f}",
                    "forward_mad_ms": f"{f_mad:.6f}",
                    "forward_backward_median_ms": f"{b_median:.6f}",
                    "forward_backward_mad_ms": f"{b_mad:.6f}",
                    "forward_speedup_vs_reference": f"{ref_forward / f_median:.3f}",
                    "forward_backward_speedup_vs_reference": f"{ref_backward / b_median:.3f}",
                }
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
