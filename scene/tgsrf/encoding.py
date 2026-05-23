import math

import torch


def safe_normalize(x, dim=-1, eps=1.0e-8):
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def inverse_sigmoid(x, eps=1.0e-6):
    x = torch.as_tensor(x).clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def fourier_encode(x, num_frequencies=4, include_input=True):
    parts = [x] if include_input else []
    if num_frequencies <= 0:
        return torch.cat(parts, dim=-1) if parts else x.new_empty(*x.shape[:-1], 0)
    freq = (2.0 ** torch.arange(num_frequencies, device=x.device, dtype=x.dtype))
    xb = x[..., None, :] * freq[:, None] * math.pi
    parts.append(torch.sin(xb).flatten(start_dim=-2))
    parts.append(torch.cos(xb).flatten(start_dim=-2))
    return torch.cat(parts, dim=-1)


def make_angle_grid(height, width, device, dtype):
    az = torch.linspace(1.0, 360.0, width, device=device, dtype=dtype) * math.pi / 180.0
    el = torch.linspace(1.0, 90.0, height, device=device, dtype=dtype) * math.pi / 180.0
    el_grid, az_grid = torch.meshgrid(el, az, indexing="ij")
    dirs = torch.stack(
        [
            torch.cos(el_grid) * torch.cos(az_grid),
            torch.cos(el_grid) * torch.sin(az_grid),
            torch.sin(el_grid),
        ],
        dim=-1,
    )
    return az_grid, el_grid, safe_normalize(dirs, dim=-1)


def direction_to_grid(direction):
    direction = safe_normalize(direction, dim=-1)
    az = torch.atan2(direction[..., 1], direction[..., 0])
    az = torch.remainder(az, 2.0 * math.pi)
    el = torch.asin(direction[..., 2].clamp(-1.0, 1.0)).clamp(0.0, 0.5 * math.pi)
    grid_x = az / (2.0 * math.pi) * 2.0 - 1.0
    grid_y = el / (0.5 * math.pi) * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def sample_feature_map_by_direction(feature_map, direction):
    if feature_map.ndim != 4:
        raise ValueError("feature_map must be [B, C, H, W]")
    if direction.ndim != 2:
        raise ValueError("direction must be [N, 3]")
    grid = direction_to_grid(direction).view(1, -1, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def scene_bounds_from_gaussians(gaussians, pad_scale=0.02):
    xyz = gaussians.get_xyz.detach()
    xyz_min = xyz.min(dim=0).values
    xyz_max = xyz.max(dim=0).values
    span = (xyz_max - xyz_min).clamp(min=1.0e-3)
    pad = span * float(pad_scale)
    return xyz_min - pad, xyz_max + pad

