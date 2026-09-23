import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from sdf_model import HashMLP_SDF
from data import ground_truth_sdf
from train import train

# train a fresh model (reuse train.py's loop)
model = HashMLP_SDF()
train(model, num_steps=500)

# sample a grid on the z=0.5 slice
res = 128
xs = np.linspace(0, 1, res)
ys = np.linspace(0, 1, res)
grid_x, grid_y = np.meshgrid(xs, ys)
grid_z = np.full_like(grid_x, 0.5)

points_np = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.float32)
points = mx.array(points_np)

pred = model(points)
gt = ground_truth_sdf(points)
mx.eval(pred, gt)

pred_img = np.array(pred).reshape(res, res)
gt_img = np.array(gt).reshape(res, res)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, img, title in [(axes[0], gt_img, "ground truth SDF"),
                         (axes[1], pred_img, "hash-encoder prediction")]:
    im = ax.imshow(img, extent=(0, 1, 0, 1), origin="lower", cmap="coolwarm", vmin=-0.3, vmax=0.3)
    ax.contour(grid_x, grid_y, img, levels=[0], colors="black", linewidths=1.5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("sdf_slice_comparison.png", dpi=150)
print("saved sdf_slice_comparison.png")