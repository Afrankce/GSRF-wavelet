import math

import torch
from torch import nn

from .encoding import make_angle_grid, safe_normalize, sample_feature_map_by_direction


def _ray_box_intersections(origin, dirs, xyz_min, xyz_max):
    inv = torch.where(dirs.abs() < 1.0e-8, torch.full_like(dirs, 1.0e8), 1.0 / dirs)
    t0 = (xyz_min.view(1, 3) - origin.view(1, 3)) * inv
    t1 = (xyz_max.view(1, 3) - origin.view(1, 3)) * inv
    t_min = torch.minimum(t0, t1).max(dim=-1).values
    t_max = torch.maximum(t0, t1).min(dim=-1).values
    valid = t_max > torch.clamp(t_min, min=0.0)
    return torch.clamp(t_min, min=0.0), t_max, valid


def build_rf_ray_seeds(
    train_views,
    xyz_min,
    xyz_max,
    num_points=12000,
    seed_views=32,
    samples_per_ray=8,
    ray_origin="tx",
    min_depth=0.05,
    ray_fraction=0.45,
    path_fraction=0.25,
    jitter_scale=0.05,
):
    """Create mixed non-grid RF seeds from angular rays, TX-RX paths and Sobol samples."""
    if not train_views:
        raise ValueError("TGS-RF seed generation requires at least one train spectrum.")

    device, dtype = xyz_min.device, xyz_min.dtype
    num_points = int(num_points)
    samples_per_ray = max(1, int(samples_per_ray))
    seed_views = min(len(train_views), max(1, int(seed_views)))
    if seed_views < len(train_views):
        ids = torch.linspace(0, len(train_views) - 1, seed_views).round().long().tolist()
        source_views = [train_views[i] for i in ids]
    else:
        source_views = list(train_views)

    ray_target = int(round(num_points * float(ray_fraction)))
    path_target = int(round(num_points * float(path_fraction)))
    ray_target = max(0, min(num_points, ray_target))
    path_target = max(0, min(num_points - ray_target, path_target))
    random_target = num_points - ray_target - path_target

    dirs_needed = max(1, math.ceil(max(1, ray_target) / (seed_views * samples_per_ray)))
    all_points = []
    for view in source_views:
        spectrum = view.spectrum.to(device=device, dtype=dtype)
        height, width = int(spectrum.shape[-2]), int(spectrum.shape[-1])
        _, _, dirs_hw = make_angle_grid(height, width, device, dtype)
        dirs = dirs_hw.reshape(-1, 3)
        strength = spectrum.reshape(-1)

        topk = min(dirs_needed, dirs.shape[0])
        _, top_idx = torch.topk(strength, k=topk, largest=True, sorted=False)
        dirs_top = dirs[top_idx]

        tx = view.T_tx.to(device=device, dtype=dtype)
        rx = view.T_rx.to(device=device, dtype=dtype)
        origin = tx if str(ray_origin).lower() == "tx" else rx
        t_min, t_max, valid = _ray_box_intersections(origin, dirs_top, xyz_min, xyz_max)
        if valid.sum() == 0:
            continue
        dirs_valid = dirs_top[valid]
        t_min = torch.clamp(t_min[valid], min=float(min_depth))
        t_max = t_max[valid]
        alpha = torch.linspace(0.15, 0.95, samples_per_ray, device=device, dtype=dtype)
        depth = t_min[:, None] * (1.0 - alpha[None, :]) + t_max[:, None] * alpha[None, :]
        pts = origin.view(1, 1, 3) + depth[..., None] * dirs_valid[:, None, :]
        all_points.append(pts.reshape(-1, 3))

    if all_points and ray_target > 0:
        points = torch.cat(all_points, dim=0)
        inside = ((points >= xyz_min.view(1, 3)) & (points <= xyz_max.view(1, 3))).all(dim=-1)
        points = points[inside]
        if points.shape[0] > ray_target:
            points = points[torch.randperm(points.shape[0], device=device)[:ray_target]]
    else:
        points = xyz_min.new_empty(0, 3)

    path_points = []
    if path_target > 0:
        per_view = max(1, math.ceil(path_target / seed_views))
        span = (xyz_max - xyz_min).norm().clamp(min=1.0e-6)
        for view in source_views:
            tx = view.T_tx.to(device=device, dtype=dtype)
            rx = view.T_rx.to(device=device, dtype=dtype)
            alpha = torch.linspace(0.05, 0.95, per_view, device=device, dtype=dtype)
            line = tx.view(1, 3) * (1.0 - alpha[:, None]) + rx.view(1, 3) * alpha[:, None]
            jitter = torch.randn_like(line) * (float(jitter_scale) * span)
            pts = (line + jitter).clamp(xyz_min.view(1, 3), xyz_max.view(1, 3))
            path_points.append(pts)
        path_points = torch.cat(path_points, dim=0)
        if path_points.shape[0] > path_target:
            path_points = path_points[torch.randperm(path_points.shape[0], device=device)[:path_target]]
        points = torch.cat([points, path_points], dim=0)

    if random_target > 0:
        sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        random01 = sobol.draw(random_target).to(device=device, dtype=dtype)
        random_pts = xyz_min + random01 * (xyz_max - xyz_min)
        points = torch.cat([points, random_pts], dim=0)

    if points.shape[0] < num_points:
        needed = num_points - points.shape[0]
        random_pts = xyz_min + torch.rand(needed, 3, device=device, dtype=dtype) * (xyz_max - xyz_min)
        points = torch.cat([points, random_pts], dim=0)
    if points.shape[0] > num_points:
        # Deterministic enough for a fixed process seed while avoiding spatial ordering artifacts.
        perm = torch.randperm(points.shape[0], device=device)[:num_points]
        points = points[perm]
    return points.contiguous()


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.mem_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(self, query, memory):
        attn_out, _ = self.attn(self.q_norm(query), self.mem_norm(memory), self.mem_norm(memory), need_weights=False)
        query = query + attn_out
        query = query + self.ffn(query)
        return query


class RFPointDecoder(nn.Module):
    """Refine non-grid RF seeds with spectrum-conditioned cross-attention."""

    def __init__(self, hidden_dim=128, geometry_dim=128, local_dim=128, num_layers=2, num_heads=4, offset_radius=0.25):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.offset_radius = float(offset_radius)
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim + geometry_dim + local_dim + 3, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads) for _ in range(int(num_layers))]
        )
        self.delta_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, seed_xyz, seed_coord, point_latent, geometry_feature, spectrum_feature, spectrum_tokens, enable_offset=True):
        x = torch.cat([seed_coord, point_latent, geometry_feature, spectrum_feature], dim=-1)
        query = self.input_proj(x).unsqueeze(0)
        memory = spectrum_tokens
        for block in self.blocks:
            query = block(query, memory)
        point_features = query.squeeze(0)
        if enable_offset:
            delta = torch.tanh(self.delta_head(point_features)) * self.offset_radius
        else:
            delta = torch.zeros_like(seed_xyz)
        return seed_xyz + delta, delta, point_features


def point_to_local_spectrum(feature_map, xyz, origin):
    direction = safe_normalize(xyz - origin.view(1, 3), dim=-1)
    return sample_feature_map_by_direction(feature_map, direction)
