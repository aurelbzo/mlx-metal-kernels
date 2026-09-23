import mlx.core as mx

source= """
    uint elem = thread_position_in_grid.x;
    out[elem] = a[elem] * a[elem];
"""

kernel = mx.fast.metal_kernel(
    name="square",
    input_names=["a"],
    output_names=["out"],
    source=source
)

a = mx.array([1.0, 2.0, 3.0, 4.0])
outputs = kernel(
    inputs=[a],
    grid=(a.size, 1, 1),
    threadgroup=(4, 1, 1),
    output_shapes=[a.shape],
    output_dtypes=[a.dtype],
)
print(outputs[0])