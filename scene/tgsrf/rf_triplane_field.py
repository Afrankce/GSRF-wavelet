import torch
from torch import nn
import torch.nn.functional as F


class RFTriplaneField(nn.Module):
    def __init__(self, xyz_min, xyz_max, channels=32, resolution=128, field_type="direct", levels=2, high_init=0.0):
        super().__init__()
        self.channels = int(channels)
        self.resolution = int(resolution)
        self.field_type = str(field_type).lower()
        self.levels = int(levels)
        self.register_buffer("xyz_min", xyz_min.detach().clone().float())
        self.register_buffer("xyz_max", xyz_max.detach().clone().float())

        if self.field_type == "direct":
            self.planes = nn.Parameter(0.01 * torch.randn(3, self.channels, self.resolution, self.resolution))
            self.highs = nn.ParameterList()
        elif self.field_type in {"haar", "bior44", "bior4.4"}:
            if self.resolution % (2 ** self.levels) != 0:
                raise ValueError("tgsrf_triplane_resolution must be divisible by 2^levels.")
            coarse = self.resolution // (2 ** self.levels)
            self.ll = nn.Parameter(0.01 * torch.randn(3, self.channels, coarse, coarse))
            self.highs = nn.ParameterList()
            for level in range(self.levels):
                size = coarse * (2 ** level)
                high = torch.zeros(3, self.channels, 3, size, size)
                if float(high_init) > 0.0:
                    high = float(high_init) * torch.randn_like(high)
                self.highs.append(nn.Parameter(high))
            self.active_levels = self.levels
        else:
            raise ValueError(f"Unknown TGS-RF triplane field type: {field_type}")

    def set_active_levels(self, active_levels=None):
        if hasattr(self, "active_levels"):
            self.active_levels = self.levels if active_levels is None else max(0, min(self.levels, int(active_levels)))

    def normalize_xyz(self, xyz):
        denom = (self.xyz_max - self.xyz_min).clamp(min=1.0e-6)
        return ((xyz - self.xyz_min) / denom).clamp(0.0, 1.0) * 2.0 - 1.0

    @staticmethod
    def _sample_plane(plane, coords):
        grid = coords.view(1, -1, 1, 2)
        sampled = F.grid_sample(plane.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()

    @staticmethod
    def _haar_iwt_step(ll, high):
        lh = high[:, :, 0]
        hl = high[:, :, 1]
        hh = high[:, :, 2]
        out = ll.new_empty(*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2)
        out[..., 0::2, 0::2] = 0.5 * (ll + lh + hl + hh)
        out[..., 0::2, 1::2] = 0.5 * (ll - lh + hl - hh)
        out[..., 1::2, 0::2] = 0.5 * (ll + lh - hl - hh)
        out[..., 1::2, 1::2] = 0.5 * (ll - lh - hl + hh)
        return out

    def materialize_planes(self):
        if self.field_type == "direct":
            return self.planes
        planes = self.ll
        for level, high in enumerate(self.highs):
            active_high = high if level < self.active_levels else torch.zeros_like(high)
            # V1 uses stable Haar reconstruction for both Haar and bior44 switches.
            # The bior44 option is kept as a separate ablation flag and can be
            # replaced by the longer filter-bank IWT after the architecture is stable.
            planes = self._haar_iwt_step(planes, active_high)
        return planes

    def wavelet_l1_loss(self):
        if not hasattr(self, "highs") or len(self.highs) == 0:
            return self.materialize_planes().new_zeros(())
        return torch.stack([h.abs().mean() for h in self.highs]).mean()

    def query(self, xyz, extra_planes=None):
        xyz_norm = self.normalize_xyz(xyz)
        planes = self.materialize_planes()
        if extra_planes is not None:
            planes = planes + extra_planes
        f_xy = self._sample_plane(planes[0], xyz_norm[:, [0, 1]])
        f_xz = self._sample_plane(planes[1], xyz_norm[:, [0, 2]])
        f_yz = self._sample_plane(planes[2], xyz_norm[:, [1, 2]])
        return torch.cat([f_xy, f_xz, f_yz], dim=-1), xyz_norm


