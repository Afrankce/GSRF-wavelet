import torch
from torch import nn


class RFGeometryAwareTriplaneEncoder(nn.Module):
    """Project point features into the three axis-aligned triplane token grids."""

    def __init__(self, hidden_dim=128, channels=32, resolution=128):
        super().__init__()
        self.channels = int(channels)
        self.resolution = int(resolution)
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, channels),
            nn.SiLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _scatter_plane(self, feat, coords01):
        res = self.resolution
        ij = (coords01.clamp(0.0, 1.0) * (res - 1)).round().long()
        flat_idx = ij[:, 1] * res + ij[:, 0]
        plane = feat.new_zeros(self.channels, res * res)
        count = feat.new_zeros(1, res * res)
        plane.index_add_(1, flat_idx, feat.transpose(0, 1))
        count.index_add_(1, flat_idx, torch.ones(1, feat.shape[0], device=feat.device, dtype=feat.dtype))
        plane = plane / count.clamp(min=1.0)
        return plane.view(self.channels, res, res)

    def forward(self, point_features, xyz_norm):
        feat = self.feature_proj(point_features)
        coords01 = (xyz_norm + 1.0) * 0.5
        xy = self._scatter_plane(feat, coords01[:, [0, 1]])
        xz = self._scatter_plane(feat, coords01[:, [0, 2]])
        yz = self._scatter_plane(feat, coords01[:, [1, 2]])
        return self.scale * torch.stack([xy, xz, yz], dim=0)
