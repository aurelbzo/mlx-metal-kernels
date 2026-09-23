"""
Simplified multiresolution hash-grid encoder (Instant-NGP style),
mirroring the role of a CUDA hash encoder (e.g. hashencoder.cu) but
implemented for Apple Silicon via MLX custom Metal kernels.

Design:
  - Forward: one fused Metal kernel per resolution level. Each GPU thread
    handles one point: computes its 8 surrounding grid-cell corners,
    hashes each corner to a row in a learned feature table, and
    trilinearly interpolates.
  - Backward: NOT done via Metal atomics (buffer-lifetime/atomic-cast
    issues are hard to debug without a Metal device on hand). Instead,
    the corner indices/weights are recomputed cheaply in NumPy (pure
    index math, no gradient needed there) and the gradient is scattered
    into the table with MLX's native `array.at[idx].add(...)`, which is
    autodiff-safe and lets the surrounding nn.Module machinery work
    normally.

KNOWN SIMPLIFICATIONS (documented on purpose, not oversights):
  - No gradient w.r.t. input coordinates (d_points is returned as zeros).
    This is fine for plain SDF-value regression, but would need to be
    implemented properly to support an Eikonal (|grad|=1) loss term.
  - Hashing is always used, even for coarse levels where a dense (collision
    -free) index would be possible and slightly more accurate. This
    matches the *spirit* of Instant-NGP, not the exact reference algorithm.

KNOWN RISK (untested — no Metal device available in the environment this
was written in): the exact call signature of `mx.custom_vjp` may differ
slightly from what's used below depending on your installed MLX version.
If you hit a signature/argument-order error, run
`python3 -c "import mlx.core as mx; help(mx.custom_vjp)"` and adjust
`_make_encode_fn` accordingly — the forward Metal kernel itself is the
part most likely to compile and run correctly as-is; the custom_vjp
wiring is the part most likely to need a small fix.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn

PRIMES = (np.uint32(1), np.uint32(2654435761), np.uint32(805459861))


def _hash_corners_numpy(points_np, resolution, table_size):
    """Pure index/weight math, no gradient needed here.

    points_np: (N, 3) float32 in [0, 1]
    returns:
      idx: (N, 8) int64 — hashed table row per corner
      w:   (N, 8) float32 — trilinear weight per corner
    Corner ordering matches the Metal kernel: c = dx*4 + dy*2 + dz.
    """
    scaled = points_np.astype(np.float32) * resolution
    base = np.floor(scaled).astype(np.int64)
    frac = scaled - base

    offsets = np.array(
        [[dx, dy, dz] for dx in range(2) for dy in range(2) for dz in range(2)],
        dtype=np.int64,
    )  # (8, 3), ordering matches kernel's c = dx*4+dy*2+dz

    corners = base[:, None, :] + offsets[None, :, :]  # (N, 8, 3)

    w = np.ones((points_np.shape[0], 8), dtype=np.float32)
    for k in range(3):
        f = frac[:, k][:, None]
        o = offsets[None, :, k]
        w = w * np.where(o == 1, f, 1.0 - f)

    x = corners[..., 0].astype(np.uint32)
    y = corners[..., 1].astype(np.uint32)
    z = corners[..., 2].astype(np.uint32)
    h = (x * PRIMES[0]) ^ (y * PRIMES[1]) ^ (z * PRIMES[2])
    idx = (h % np.uint32(table_size)).astype(np.int64)

    return idx, w


def _make_forward_kernel(resolution, table_size, feature_dim):
    source = f"""
        uint i = thread_position_in_grid.x;
        uint N = points_shape[0];
        if (i >= N) {{ return; }}

        const uint RES = {int(resolution)}u;
        const uint T = {int(table_size)}u;
        const uint F = {int(feature_dim)}u;
        const uint P1 = 1u;
        const uint P2 = 2654435761u;
        const uint P3 = 805459861u;

        float px = points[i*3+0] * (float)RES;
        float py = points[i*3+1] * (float)RES;
        float pz = points[i*3+2] * (float)RES;

        int x0 = (int)floor(px);
        int y0 = (int)floor(py);
        int z0 = (int)floor(pz);
        float fx = px - (float)x0;
        float fy = py - (float)y0;
        float fz = pz - (float)z0;

        for (uint f = 0; f < F; f++) {{
            out[i*F+f] = 0.0;
        }}

        for (uint c = 0; c < 8; c++) {{
            uint dx = (c >> 2) & 1u;
            uint dy = (c >> 1) & 1u;
            uint dz = c & 1u;

            uint cx = (uint)(x0 + (int)dx);
            uint cy = (uint)(y0 + (int)dy);
            uint cz = (uint)(z0 + (int)dz);

            uint h = (cx * P1) ^ (cy * P2) ^ (cz * P3);
            uint idx = h % T;

            float wx = (dx == 1u) ? fx : (1.0 - fx);
            float wy = (dy == 1u) ? fy : (1.0 - fy);
            float wz = (dz == 1u) ? fz : (1.0 - fz);
            float w = wx * wy * wz;

            for (uint f = 0; f < F; f++) {{
                out[i*F+f] += w * table[idx*F+f];
            }}
        }}
    """
    return mx.fast.metal_kernel(
        name=f"hash_encode_res{int(resolution)}",
        input_names=["points", "table"],
        output_names=["out"],
        source=source,
    )


class HashEncoder(nn.Module):
    """Multiresolution hash-grid encoder. Input points must be in [0, 1]^3."""

    def __init__(
        self,
        num_levels=4,
        base_resolution=4,
        growth_factor=2.0,
        table_size=2**12,
        feature_dim=2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.table_size = table_size
        self.feature_dim = feature_dim
        self.resolutions = [
            int(round(base_resolution * (growth_factor**l))) for l in range(num_levels)
        ]

        # All levels' tables stacked into one trainable parameter.
        self.tables = mx.random.uniform(
            low=-1e-4, high=1e-4, shape=(num_levels, table_size, feature_dim)
        )

        self._kernels = [
            _make_forward_kernel(res, table_size, feature_dim) for res in self.resolutions
        ]
        self._encode_fns = [
            self._make_encode_fn(l, self.resolutions[l]) for l in range(num_levels)
        ]

    def _forward_raw(self, level, points, table):
        N = points.shape[0]
        outputs = self._kernels[level](
            inputs=[points, table],
            grid=(N, 1, 1),
            threadgroup=(min(N, 256), 1, 1),
            output_shapes=[(N, self.feature_dim)],
            output_dtypes=[mx.float32],
        )
        return outputs[0]
    def _make_encode_fn(self, level, resolution):
        table_size = self.table_size
        feature_dim = self.feature_dim
        forward_raw = self._forward_raw

        @mx.custom_function
        def encode(points, table):
            return forward_raw(level, points, table)

        @encode.vjp
        def encode_vjp(primals, cotangent, output):
            # NOTE: with a single-output forward, MLX passes cotangent/output
            # as single arrays, not lists — and expects a plain tuple back.
            points, table = primals
            d_out = cotangent

            points_np = np.array(points)
            idx_np, w_np = _hash_corners_numpy(points_np, resolution, table_size)

            idx_mx = mx.array(idx_np)  # (N, 8)
            w_mx = mx.array(w_np)  # (N, 8)

            d_table = mx.zeros((table_size, feature_dim))
            for c in range(8):
                contrib = w_mx[:, c : c + 1] * d_out  # (N, F)
                d_table = d_table.at[idx_mx[:, c]].add(contrib)

            d_points = mx.zeros_like(points)  # see module docstring: no coord grad
            return d_points, d_table

        return encode
    
    def __call__(self, points):
        feats = []
        for l in range(self.num_levels):
            table_l = self.tables[l]
            feats.append(self._encode_fns[l](points, table_l))
        return mx.concatenate(feats, axis=-1)  # (N, num_levels * feature_dim)
