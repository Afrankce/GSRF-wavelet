import torch
from torch import nn


class RFGaussianDecoder(nn.Module):
    """Decode RF Gaussian attributes from point, triplane, geometry and spectrum features."""

    def __init__(
        self,
        hidden_dim=128,
        triplane_channels=32,
        geometry_dim=128,
        local_dim=128,
        offset_radius=0.15,
        att_radius=0.05,
        scale_radius=0.10,
    ):
        super().__init__()
        self.offset_radius = float(offset_radius)
        self.att_radius = float(att_radius)
        self.scale_radius = float(scale_radius)
        in_dim = hidden_dim + 3 * triplane_channels + geometry_dim + local_dim + hidden_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.xyz_head = nn.Linear(hidden_dim, 3)
        self.att_head = nn.Linear(hidden_dim, 1)
        self.scale_head = nn.Linear(hidden_dim, 3)
        for layer in (self.xyz_head, self.att_head, self.scale_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, point_features, triplane_features, geometry_features, local_spectrum_features, global_token):
        global_expand = global_token.expand(point_features.shape[0], -1)
        x = torch.cat(
            [
                point_features,
                triplane_features,
                geometry_features,
                local_spectrum_features,
                global_expand,
            ],
            dim=-1,
        )
        h = self.shared(x)
        return {
            "delta_xyz": torch.tanh(self.xyz_head(h)) * self.offset_radius,
            "attenuation_delta": torch.tanh(self.att_head(h)) * self.att_radius,
            "log_scale_delta": torch.tanh(self.scale_head(h)) * self.scale_radius,
        }

