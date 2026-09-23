import math

import mlx.core as mx
import mlx.nn as nn

from hash_encoder import HashEncoder


class SDFNet(nn.Module):
    """Small coordinate MLP: encoded features -> scalar SDF value."""

    def __init__(self, in_dim, hidden=64, depth=3):
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)]
        self.out = nn.Linear(hidden, 1)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.out(x).squeeze(-1)


class FourierEncoder(nn.Module):
    """Fixed (non-learned) sinusoidal positional encoding. Baseline to
    compare the learned hash-grid encoder against."""

    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs = mx.array([2.0**i for i in range(num_freqs)], dtype=mx.float32)

    def __call__(self, points):
        # points in [0,1] -> center to [-1,1] for a nicer-conditioned encoding
        x = points * 2.0 - 1.0  # (N, 3)
        angles = x[:, :, None] * self.freqs[None, None, :] * math.pi  # (N, 3, F)
        s = mx.sin(angles)
        c = mx.cos(angles)
        enc = mx.concatenate([s, c], axis=-1).reshape(points.shape[0], -1)
        return mx.concatenate([x, enc], axis=-1)

    @property
    def out_dim(self):
        return 3 + 3 * 2 * self.num_freqs


class HashMLP_SDF(nn.Module):
    """Learned hash-grid encoder + MLP."""

    def __init__(self, num_levels=4, base_resolution=4, growth_factor=2.0,
                 table_size=2**12, feature_dim=2, hidden=64, depth=3):
        super().__init__()
        self.encoder = HashEncoder(
            num_levels=num_levels,
            base_resolution=base_resolution,
            growth_factor=growth_factor,
            table_size=table_size,
            feature_dim=feature_dim,
        )
        self.mlp = SDFNet(in_dim=num_levels * feature_dim, hidden=hidden, depth=depth)

    def __call__(self, points):
        feats = self.encoder(points)
        return self.mlp(feats)


class FourierMLP_SDF(nn.Module):
    """Baseline: fixed Fourier encoder + MLP (no custom kernel involved)."""

    def __init__(self, num_freqs=6, hidden=64, depth=3):
        super().__init__()
        self.encoder = FourierEncoder(num_freqs=num_freqs)
        self.mlp = SDFNet(in_dim=self.encoder.out_dim, hidden=hidden, depth=depth)

    def __call__(self, points):
        feats = self.encoder(points)
        return self.mlp(feats)
