import math

import torch
from torch import nn

from .encoding import fourier_encode, safe_normalize


class RFGeometryEncoder(nn.Module):
    """Build NeRF2-style propagation features for candidate Gaussian anchors."""

    def __init__(self, hidden_dim=128, num_frequencies=4, frequency_hz=915.0e6, scene_scale=1.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_frequencies = int(num_frequencies)
        self.frequency_hz = float(frequency_hz)
        self.scene_scale = float(max(scene_scale, 1.0e-6))

        base_dim = 3 + 3 + 3 + 3 + 3 + 5
        encoded_dim = base_dim * (1 + 2 * self.num_frequencies)
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def raw_features(self, xyz, tx, rx):
        tx = tx.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
        rx = rx.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
        to_rx = rx - xyz
        to_tx = tx - xyz
        from_rx = xyz - rx
        from_tx = xyz - tx

        d_rx = to_rx.norm(dim=-1, keepdim=True)
        d_tx = to_tx.norm(dim=-1, keepdim=True)
        path = d_rx + d_tx
        wavelength = 3.0e8 / self.frequency_hz
        phase = 2.0 * math.pi * path / wavelength

        scaled_xyz = xyz / self.scene_scale
        raw = torch.cat(
            [
                scaled_xyz,
                safe_normalize(to_rx),
                safe_normalize(to_tx),
                safe_normalize(from_rx),
                safe_normalize(from_tx),
                d_rx / self.scene_scale,
                d_tx / self.scene_scale,
                path / self.scene_scale,
                torch.sin(phase),
                torch.cos(phase),
            ],
            dim=-1,
        )
        return raw

    def forward(self, xyz, tx, rx):
        raw = self.raw_features(xyz, tx, rx)
        return self.net(fourier_encode(raw, self.num_frequencies))
