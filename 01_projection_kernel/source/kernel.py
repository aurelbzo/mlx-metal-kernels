import mlx.core as mx

source = """
    uint i = thread_position_in_grid.x;
    uint N = points_shape[0];
    if (i >= N) { return; }

    uint M = sdf_values_shape[1];
    float best_abs = 1e30;
    uint best_idx = 0;

    for (uint m = 0; m < M; m++) {
        float v = sdf_values[i * M + m];
        float av = fabs(v);
        if (av < best_abs) {
            best_abs = av;
            best_idx = m;
        }
    }

    float dist = sdf_values[i * M + best_idx];
    for (uint c = 0; c < 3; c++) {
        float g = sdf_grads[(i * M + best_idx) * 3 + c];
        float p = points[i * 3 + c];
        out[i * 3 + c] = p - dist * g;
    }
"""

fused_projection_kernel = mx.fast.metal_kernel(
    name="fused_projection",
    input_names=["points", "sdf_values", "sdf_grads"],
    output_names=["out"],
    source=source,
)

def fused_projection(points, sdf_values, sdf_grads):
    N = points.shape[0]
    outputs = fused_projection_kernel(
        inputs=[points, sdf_values, sdf_grads],
        grid=(N, 1, 1),
        threadgroup=(min(N, 256), 1, 1),
        output_shapes=[(N, 3)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


if __name__ == "__main__":
    from baseline import naive_projection
    from benchmark import make_data, time_fn

    # correctness check
    points, sdf_values, sdf_grads = make_data(1000)
    ref = naive_projection(points, sdf_values, sdf_grads)
    fused = fused_projection(points, sdf_values, sdf_grads)
    mx.eval(ref, fused)
    max_diff = mx.max(mx.abs(ref - fused)).item()
    print(f"max diff vs naive: {max_diff:.2e}")  # should be ~1e-6 or less

    # benchmark comparison
    for N in [1_000, 10_000, 100_000, 1_000_000]:
        points, sdf_values, sdf_grads = make_data(N)
        t_naive = time_fn(naive_projection, points, sdf_values, sdf_grads)
        t_fused = time_fn(fused_projection, points, sdf_values, sdf_grads)
        speedup = t_naive / t_fused
        print(f"N={N:>9}  naive: {t_naive*1000:7.4f} ms  "
              f"fused: {t_fused*1000:7.4f} ms  speedup: {speedup:.2f}x")