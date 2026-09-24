"""
Simplified multiresolution hash-grid encoder (Instant-NGP style),
mirroring the role of a CUDA hash encoder (e.g. hashencoder.cu) but
implemented for Apple Silicon via MLX custom Metal kernels.

Design:
  - Forward: one fused Metal kernel per resolution level. Each GPU thread
    handles one point: computes its 8 surrounding grid-cell corners,
    hashes each corner to a row in a learned feature table, and
    trilinearly interpolates.
  - Backward: a custom Metal kernel computes coordinate gradients and
    atomically accumulates table gradients. `backward_backend="mlx_scatter"`
    retains the earlier MLX indexed-add implementation for comparison.

KNOWN SIMPLIFICATIONS (documented on purpose, not oversights):
  - The custom VJP computes first-order gradients w.r.t. input coordinates
    analytically from the trilinear interpolation weights. This is needed
    for SDF normals and is a step toward Eikonal training; higher-order
    differentiation through this custom VJP still needs validation.
  - Hashing is always used, even for coarse levels where a dense (collision
    -free) index would be possible and slightly more accurate. This
    matches the *spirit* of Instant-NGP, not the exact reference algorithm.

The GPU backward path requires validation on a real Metal device. The MLX
scatter backend is retained as a comparison, not as a CPU fallback for the
Metal forward pass.
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


def _make_backward_kernel(resolution, table_size, feature_dim):
    """Build a fused VJP kernel for one hash-grid level.

    Each thread handles one point. Coordinate gradients are private to that
    thread; colliding table updates are accumulated atomically.
    """
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
        float3 point_grad = float3(0.0);

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
            float sx = (dx == 1u) ? 1.0 : -1.0;
            float sy = (dy == 1u) ? 1.0 : -1.0;
            float sz = (dz == 1u) ? 1.0 : -1.0;
            float w = wx * wy * wz;

            for (uint f = 0; f < F; f++) {{
                float g = d_out[i*F+f];
                float value = table[idx*F+f];
                atomic_fetch_add_explicit(
                    &d_table[idx*F+f], w * g, memory_order_relaxed);
                point_grad.x += value * g * sx * (float)RES * wy * wz;
                point_grad.y += value * g * sy * (float)RES * wx * wz;
                point_grad.z += value * g * sz * (float)RES * wx * wy;
            }}
        }}
        atomic_store_explicit(&d_points[i*3+0], point_grad.x, memory_order_relaxed);
        atomic_store_explicit(&d_points[i*3+1], point_grad.y, memory_order_relaxed);
        atomic_store_explicit(&d_points[i*3+2], point_grad.z, memory_order_relaxed);
    """
    return mx.fast.metal_kernel(
        name=f"hash_encode_grad_res{int(resolution)}",
        input_names=["points", "table", "d_out"],
        output_names=["d_points", "d_table"],
        source=source,
        atomic_outputs=True,
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
        backward_backend="metal",
    ):
        super().__init__()
        self.num_levels = num_levels
        self.table_size = table_size
        self.feature_dim = feature_dim
        if backward_backend not in {"metal", "mlx_scatter"}:
            raise ValueError("backward_backend must be 'metal' or 'mlx_scatter'")
        self.backward_backend = backward_backend
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
        self._backward_kernels = [
            _make_backward_kernel(res, table_size, feature_dim) for res in self.resolutions
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

    def _backward_raw(self, level, points, table, d_out):
        N = points.shape[0]
        outputs = self._backward_kernels[level](
            inputs=[points, table, d_out],
            grid=(N, 1, 1),
            threadgroup=(min(N, 256), 1, 1),
            output_shapes=[points.shape, table.shape],
            output_dtypes=[points.dtype, table.dtype],
            init_value=0,
        )
        return outputs[0], outputs[1]

    def _make_encode_fn(self, level, resolution):
        table_size = self.table_size
        feature_dim = self.feature_dim
        forward_raw = self._forward_raw
        backward_raw = self._backward_raw
        backward_backend = self.backward_backend

        @mx.custom_function
        def encode(points, table):
            return forward_raw(level, points, table)

        @encode.vjp
        def encode_vjp(primals, cotangent, output):
            # NOTE: with a single-output forward, MLX passes cotangent/output
            # as single arrays, not lists — and expects a plain tuple back.
            points, table = primals
            d_out = cotangent

            if backward_backend == "metal":
                return backward_raw(level, points, table, d_out)

            points_np = np.array(points)
            idx_np, w_np = _hash_corners_numpy(points_np, resolution, table_size)

            idx_mx = mx.array(idx_np)  # (N, 8)
            w_mx = mx.array(w_np)  # (N, 8)

            d_table = mx.zeros((table_size, feature_dim))
            for c in range(8):
                contrib = w_mx[:, c : c + 1] * d_out  # (N, F)
                d_table = d_table.at[idx_mx[:, c]].add(contrib)

            # Differentiate the trilinear interpolation weights with respect
            # to the input coordinates. Indices are piecewise constant; only
            # the fractional position inside the cell contributes a gradient.
            scaled = points * resolution
            frac = scaled - mx.floor(scaled)
            fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
            dx_terms = []
            dy_terms = []
            dz_terms = []
            for c in range(8):
                dx, dy, dz = (c >> 2) & 1, (c >> 1) & 1, c & 1
                wx = fx if dx else 1.0 - fx
                wy = fy if dy else 1.0 - fy
                wz = fz if dz else 1.0 - fz
                table_values = table[idx_mx[:, c]]
                weighted_cotangent = d_out * table_values
                dx_sign = 1.0 if dx else -1.0
                dy_sign = 1.0 if dy else -1.0
                dz_sign = 1.0 if dz else -1.0
                dx_terms.append(mx.sum(weighted_cotangent * (dx_sign * resolution * wy * wz)[:, None], axis=-1))
                dy_terms.append(mx.sum(weighted_cotangent * (dy_sign * resolution * wx * wz)[:, None], axis=-1))
                dz_terms.append(mx.sum(weighted_cotangent * (dz_sign * resolution * wx * wy)[:, None], axis=-1))

            d_points = mx.stack(
                [sum(dx_terms), sum(dy_terms), sum(dz_terms)], axis=-1
            )
            return d_points, d_table

        return encode
    
    def __call__(self, points):
        feats = []
        for l in range(self.num_levels):
            table_l = self.tables[l]
            feats.append(self._encode_fns[l](points, table_l))
        return mx.concatenate(feats, axis=-1)  # (N, num_levels * feature_dim)
