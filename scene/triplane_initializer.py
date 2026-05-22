import random

import torch
from torch import nn
import torch.nn.functional as F

from utils.loss_utils import l1_loss, ssim, fourier_loss


class TriPlaneField(nn.Module):
    """Learnable factorized 3D feature field queried at Gaussian centers."""

    def __init__(self, xyz_min, xyz_max, channels=16, resolution=64):
        super().__init__()

        self.channels = int(channels)
        self.resolution = int(resolution)

        self.register_buffer("xyz_min", xyz_min.detach().clone().float())
        self.register_buffer("xyz_max", xyz_max.detach().clone().float())

        self.planes = nn.Parameter(
            0.01 * torch.randn(3, self.channels, self.resolution, self.resolution)
        )

    def normalize_xyz(self, xyz):
        denom = (self.xyz_max - self.xyz_min).clamp(min=1e-6)
        xyz01 = (xyz - self.xyz_min) / denom
        return xyz01.clamp(0.0, 1.0) * 2.0 - 1.0

    @staticmethod
    def _sample_plane(plane, coords):
        # grid_sample expects [N, H_out, W_out, 2]; we query all points as H_out=N, W_out=1.
        grid = coords.view(1, -1, 1, 2)
        sampled = F.grid_sample(
            plane.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()

    def forward(self, xyz):
        xyz_norm = self.normalize_xyz(xyz)

        xy = xyz_norm[:, [0, 1]]
        xz = xyz_norm[:, [0, 2]]
        yz = xyz_norm[:, [1, 2]]

        f_xy = self._sample_plane(self.planes[0], xy)
        f_xz = self._sample_plane(self.planes[1], xz)
        f_yz = self._sample_plane(self.planes[2], yz)

        return torch.cat([f_xy, f_xz, f_yz], dim=-1), xyz_norm


class HaarWaveletTriPlaneField(TriPlaneField):
    """TriPlane whose feature planes are reconstructed from Haar wavelet coefficients."""

    def __init__(
        self,
        xyz_min,
        xyz_max,
        channels=16,
        resolution=64,
        levels=2,
        high_init=0.0,
    ):
        nn.Module.__init__(self)

        self.channels = int(channels)
        self.resolution = int(resolution)
        self.levels = int(levels)
        if self.levels < 1:
            raise ValueError("Haar wavelet triplane requires at least one wavelet level.")
        if self.resolution % (2 ** self.levels) != 0:
            raise ValueError(
                f"triplane_resolution={self.resolution} must be divisible by 2^levels={2 ** self.levels}."
            )

        self.register_buffer("xyz_min", xyz_min.detach().clone().float())
        self.register_buffer("xyz_max", xyz_max.detach().clone().float())

        coarse_resolution = self.resolution // (2 ** self.levels)
        self.ll = nn.Parameter(
            0.01 * torch.randn(3, self.channels, coarse_resolution, coarse_resolution)
        )
        self.highs = nn.ParameterList()
        for level in range(self.levels):
            band_resolution = coarse_resolution * (2 ** level)
            high = torch.zeros(3, self.channels, 3, band_resolution, band_resolution)
            if float(high_init) > 0.0:
                high = float(high_init) * torch.randn_like(high)
            self.highs.append(nn.Parameter(high))

        self.active_levels = self.levels

    def set_active_levels(self, active_levels=None):
        if active_levels is None:
            self.active_levels = self.levels
        else:
            self.active_levels = max(0, min(self.levels, int(active_levels)))

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
        planes = self.ll
        for level, high in enumerate(self.highs):
            if level < self.active_levels:
                active_high = high
            else:
                active_high = torch.zeros_like(high)
            planes = self._haar_iwt_step(planes, active_high)
        return planes

    def wavelet_l1_loss(self):
        if not self.highs:
            return self.ll.new_zeros(())
        return torch.stack([high.abs().mean() for high in self.highs]).mean()

    def forward(self, xyz):
        xyz_norm = self.normalize_xyz(xyz)
        planes = self.materialize_planes()

        xy = xyz_norm[:, [0, 1]]
        xz = xyz_norm[:, [0, 2]]
        yz = xyz_norm[:, [1, 2]]

        f_xy = self._sample_plane(planes[0], xy)
        f_xz = self._sample_plane(planes[1], xz)
        f_yz = self._sample_plane(planes[2], yz)

        return torch.cat([f_xy, f_xz, f_yz], dim=-1), xyz_norm


class Bior44WaveletTriPlaneField(TriPlaneField):
    """Biorthogonal 4.4 wavelet triplane reconstructed by differentiable IWT."""

    _REC_LO = [
        0.0,
        -0.06453888262869706,
        -0.04068941760916406,
        0.41809227322161724,
        0.7884856164055829,
        0.41809227322161724,
        -0.04068941760916406,
        -0.06453888262869706,
        0.0,
        0.0,
    ]
    _REC_HI = [
        0.0,
        -0.03782845550726404,
        -0.023849465019556843,
        0.11062440441843718,
        0.37740285561283066,
        -0.8526986790088938,
        0.37740285561283066,
        0.11062440441843718,
        -0.023849465019556843,
        -0.03782845550726404,
    ]

    def __init__(
        self,
        xyz_min,
        xyz_max,
        channels=16,
        resolution=64,
        levels=2,
        high_init=0.0,
    ):
        nn.Module.__init__(self)

        self.channels = int(channels)
        self.resolution = int(resolution)
        self.levels = int(levels)
        if self.levels < 1:
            raise ValueError("Bior4.4 wavelet triplane requires at least one wavelet level.")
        if self.resolution % (2 ** self.levels) != 0:
            raise ValueError(
                f"triplane_resolution={self.resolution} must be divisible by 2^levels={2 ** self.levels}."
            )

        self.register_buffer("xyz_min", xyz_min.detach().clone().float())
        self.register_buffer("xyz_max", xyz_max.detach().clone().float())
        self.register_buffer("rec_lo", torch.tensor(self._REC_LO, dtype=torch.float32), persistent=False)
        self.register_buffer("rec_hi", torch.tensor(self._REC_HI, dtype=torch.float32), persistent=False)

        coarse_resolution = self.resolution // (2 ** self.levels)
        self.ll = nn.Parameter(
            0.01 * torch.randn(3, self.channels, coarse_resolution, coarse_resolution)
        )
        self.highs = nn.ParameterList()
        for level in range(self.levels):
            band_resolution = coarse_resolution * (2 ** level)
            high = torch.zeros(3, self.channels, 3, band_resolution, band_resolution)
            if float(high_init) > 0.0:
                high = float(high_init) * torch.randn_like(high)
            self.highs.append(nn.Parameter(high))

        self.active_levels = self.levels

    def set_active_levels(self, active_levels=None):
        if active_levels is None:
            self.active_levels = self.levels
        else:
            self.active_levels = max(0, min(self.levels, int(active_levels)))

    @staticmethod
    def _zero_upsample_2d(x):
        out = x.new_zeros(*x.shape[:-2], x.shape[-2] * 2, x.shape[-1] * 2)
        out[..., 0::2, 0::2] = x
        return out

    @staticmethod
    def _filter_same_2d(x, row_filter, col_filter):
        n_planes, n_channels, height, width = x.shape
        y = x.reshape(n_planes * n_channels, 1, height, width)

        row_kernel = row_filter.to(device=x.device, dtype=x.dtype).flip(0).view(1, 1, 1, -1)
        col_kernel = col_filter.to(device=x.device, dtype=x.dtype).flip(0).view(1, 1, -1, 1)
        pad_left = (row_kernel.shape[-1] - 1) // 2
        pad_right = row_kernel.shape[-1] - 1 - pad_left
        pad_top = (col_kernel.shape[-2] - 1) // 2
        pad_bottom = col_kernel.shape[-2] - 1 - pad_top

        y = F.pad(y, (pad_left, pad_right, 0, 0), mode="replicate")
        y = F.conv2d(y, row_kernel)
        y = F.pad(y, (0, 0, pad_top, pad_bottom), mode="replicate")
        y = F.conv2d(y, col_kernel)
        return y.reshape(n_planes, n_channels, height, width)

    def _bior44_iwt_step(self, ll, high):
        lh = high[:, :, 0]
        hl = high[:, :, 1]
        hh = high[:, :, 2]

        lo = self.rec_lo
        hi = self.rec_hi
        ll_up = self._zero_upsample_2d(ll)
        lh_up = self._zero_upsample_2d(lh)
        hl_up = self._zero_upsample_2d(hl)
        hh_up = self._zero_upsample_2d(hh)

        return (
            self._filter_same_2d(ll_up, lo, lo)
            + self._filter_same_2d(lh_up, hi, lo)
            + self._filter_same_2d(hl_up, lo, hi)
            + self._filter_same_2d(hh_up, hi, hi)
        )

    def materialize_planes(self):
        planes = self.ll
        for level, high in enumerate(self.highs):
            if level < self.active_levels:
                active_high = high
            else:
                active_high = torch.zeros_like(high)
            planes = self._bior44_iwt_step(planes, active_high)
        return planes

    def wavelet_l1_loss(self):
        if not self.highs:
            return self.ll.new_zeros(())
        return torch.stack([high.abs().mean() for high in self.highs]).mean()

    def forward(self, xyz):
        xyz_norm = self.normalize_xyz(xyz)
        planes = self.materialize_planes()

        xy = xyz_norm[:, [0, 1]]
        xz = xyz_norm[:, [0, 2]]
        yz = xyz_norm[:, [1, 2]]

        f_xy = self._sample_plane(planes[0], xy)
        f_xz = self._sample_plane(planes[1], xz)
        f_yz = self._sample_plane(planes[2], yz)

        return torch.cat([f_xy, f_xz, f_yz], dim=-1), xyz_norm


class FourierGeometryEncoding(nn.Module):
    """Fixed Fourier features for low-dimensional RF geometry descriptors."""

    def __init__(self, num_frequencies=4, include_input=True):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        freq_bands = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_multiplier(self):
        return (1 if self.include_input else 0) + 2 * self.num_frequencies

    def forward(self, x):
        encoded = []
        if self.include_input:
            encoded.append(x)

        if self.num_frequencies > 0:
            freq_bands = self.freq_bands.to(device=x.device, dtype=x.dtype)
            xb = x.unsqueeze(-1) * freq_bands.view(*([1] * x.dim()), -1)
            xb = xb.flatten(start_dim=-2)
            encoded.extend([torch.sin(torch.pi * xb), torch.cos(torch.pi * xb)])

        return torch.cat(encoded, dim=-1)


class RFMLPDecoder(nn.Module):
    """Original shallow concatenation decoder kept as an ablation."""

    def __init__(self, tri_dim, geom_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tri_dim + geom_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        self.reset_output()

    def reset_output(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, tri_feat, geom_feat):
        return self.net(torch.cat([tri_feat, geom_feat], dim=-1))


class FiLMResidualBlock(nn.Module):
    """TX/RX-conditioned residual block for RF geometry features."""

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(cond_dim, 2 * hidden_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, h, cond):
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        x = self.norm(self.linear(h))
        x = x * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        return h + self.act(x)
    


class RFFiLMFourierDecoder(nn.Module):
    """Fourier geometry encoder with TX/RX-conditioned FiLM residual blocks."""

    def __init__(self, tri_dim, geom_dim, hidden_dim, num_layers=3, num_frequencies=4):
        super().__init__()
        self.fourier = FourierGeometryEncoding(num_frequencies=num_frequencies, include_input=True)
        geom_encoded_dim = geom_dim * self.fourier.out_multiplier

        self.input_proj = nn.Sequential(
            nn.Linear(tri_dim + geom_encoded_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim, hidden_dim) for _ in range(int(num_layers))]
        )
        self.out = nn.Linear(hidden_dim, 4)
        self.reset_output()

    def reset_output(self):
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tri_feat, geom_feat):
        geom_encoded = self.fourier(geom_feat)
        cond = self.cond_proj(geom_feat)
        h = self.input_proj(torch.cat([tri_feat, geom_encoded], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h)


class RFTriplaneInitializer(nn.Module):
    """Predict bounded Gaussian center offsets and attenuation initialization."""

    def __init__(
        self,
        xyz_min,
        xyz_max,
        channels=16,
        resolution=64,
        hidden_dim=64,
        offset_radius=0.25,
        attenuation_radius=0.1,
        decoder_type="mlp",
        fourier_frequencies=4,
        film_layers=3,
        output_mode="offset_att",
        field_type="direct",
        wavelet_levels=2,
        wavelet_high_init=0.0,
    ):
        super().__init__()

        self.field_type = str(field_type).lower()
        if self.field_type in {"direct", "triplane"}:
            self.triplane = TriPlaneField(
                xyz_min=xyz_min,
                xyz_max=xyz_max,
                channels=channels,
                resolution=resolution,
            )
        elif self.field_type in {"haar", "wavelet_haar"}:
            self.triplane = HaarWaveletTriPlaneField(
                xyz_min=xyz_min,
                xyz_max=xyz_max,
                channels=channels,
                resolution=resolution,
                levels=wavelet_levels,
                high_init=wavelet_high_init,
            )
        elif self.field_type in {"bior4.4", "bior44", "wavelet_bior44"}:
            self.triplane = Bior44WaveletTriPlaneField(
                xyz_min=xyz_min,
                xyz_max=xyz_max,
                channels=channels,
                resolution=resolution,
                levels=wavelet_levels,
                high_init=wavelet_high_init,
            )
        else:
            raise ValueError(f"Unknown triplane field type: {self.field_type}")

        self.offset_radius = float(offset_radius)
        self.attenuation_radius = float(attenuation_radius)
        self.decoder_type = str(decoder_type)
        self.output_mode = str(output_mode)
        if self.output_mode not in {"offset_att", "gating_only"}:
            raise ValueError(f"Unknown triplane output mode: {self.output_mode}")

        geom_dim = 18
        self.use_plane_features = not self.decoder_type.endswith("_no_triplane")
        base_decoder_type = self.decoder_type.removesuffix("_no_triplane")
        tri_dim = int(channels) * 3 if self.use_plane_features else 0

        if base_decoder_type == "mlp":
            self.head = RFMLPDecoder(tri_dim=tri_dim, geom_dim=geom_dim, hidden_dim=hidden_dim)
        elif base_decoder_type == "film_fourier":
            self.head = RFFiLMFourierDecoder(
                tri_dim=tri_dim,
                geom_dim=geom_dim,
                hidden_dim=hidden_dim,
                num_layers=film_layers,
                num_frequencies=fourier_frequencies,
            )
        else:
            raise ValueError(f"Unknown triplane decoder type: {self.decoder_type}")

    def _geometry_features(self, xyz, xyz_norm, tx_context, rx_context):
        tx = tx_context.to(device=xyz.device, dtype=xyz.dtype).view(1, 3).expand_as(xyz)
        rx = rx_context.to(device=xyz.device, dtype=xyz.dtype).view(1, 3).expand_as(xyz)

        tx_norm = self.triplane.normalize_xyz(tx)
        rx_norm = self.triplane.normalize_xyz(rx)

        scene_scale = (self.triplane.xyz_max - self.triplane.xyz_min).norm().clamp(min=1e-6)

        vec_tx = xyz - tx
        vec_rx = rx - xyz
        d_tx = vec_tx.norm(dim=-1, keepdim=True)
        d_rx = vec_rx.norm(dim=-1, keepdim=True)
        d_total = d_tx + d_rx

        dir_tx = vec_tx / d_tx.clamp(min=1e-6)
        dir_rx = vec_rx / d_rx.clamp(min=1e-6)

        return torch.cat(
            [
                xyz_norm,
                tx_norm,
                rx_norm,
                d_tx / scene_scale,
                d_rx / scene_scale,
                d_total / scene_scale,
                dir_tx,
                dir_rx,
            ],
            dim=-1,
        )

    def forward(self, base_xyz, base_attenuation_raw, tx_context, rx_context):
        if self.use_plane_features:
            tri_feat, xyz_norm = self.triplane(base_xyz)
        else:
            xyz_norm = self.triplane.normalize_xyz(base_xyz)
            tri_feat = base_xyz.new_empty((base_xyz.shape[0], 0))
        geom_feat = self._geometry_features(base_xyz, xyz_norm, tx_context, rx_context)

        pred = self.head(tri_feat, geom_feat)
        if self.output_mode == "gating_only":
            delta_xyz = torch.zeros_like(base_xyz)
        else:
            delta_xyz = self.offset_radius * torch.tanh(pred[:, :3])
        attenuation_delta = self.attenuation_radius * torch.tanh(pred[:, 3:4])
        attenuation_raw = base_attenuation_raw + attenuation_delta

        return base_xyz + delta_xyz, attenuation_raw, delta_xyz, attenuation_delta

    def set_wavelet_active_levels(self, active_levels=None):
        if hasattr(self.triplane, "set_active_levels"):
            self.triplane.set_active_levels(active_levels)

    def wavelet_l1_loss(self):
        if hasattr(self.triplane, "wavelet_l1_loss"):
            return self.triplane.wavelet_l1_loss()
        return next(self.parameters()).new_zeros(())


def _scene_context(scene, device, dtype):
    train_views = scene.getTrainSpectrums()
    tx_stack = torch.stack([view.T_tx for view in train_views], dim=0).to(device=device, dtype=dtype)
    rx_context = train_views[0].T_rx.to(device=device, dtype=dtype)
    tx_context = tx_stack.mean(dim=0)
    return tx_context, rx_context


def _make_warmup_optimizer(init_model, gaussians, model_args, optim_args):
    params = [{"params": init_model.parameters(), "lr": getattr(model_args, "triplane_lr", 1.0e-3)}]

    if getattr(model_args, "triplane_update_gaussian_attrs", False):
        params.extend(
            [
                {"params": [gaussians._features_dc], "lr": optim_args.feature_lr},
                {
                    "params": [gaussians._features_rest],
                    "lr": optim_args.feature_lr * getattr(optim_args, "_rest_lr_ratio", 1.0),
                },
                {"params": [gaussians._attenuation], "lr": optim_args.opacity_lr},
                {"params": [gaussians._scaling], "lr": optim_args.scaling_lr},
                {"params": [gaussians._rotation], "lr": optim_args.rotation_lr},
            ]
        )

    return torch.optim.Adam(params, lr=0.0, eps=1e-15)


def _finalize_with_tx_average(init_model, gaussians, train_views, base_xyz, base_attenuation_raw, tx_context, rx_context, model_args):
    use_view_tx = getattr(model_args, "triplane_use_view_tx", True)
    n_samples = int(getattr(model_args, "triplane_finalize_tx_samples", 32))

    if (not use_view_tx) or n_samples <= 1 or len(train_views) <= 1:
        return init_model(base_xyz, base_attenuation_raw, tx_context, rx_context)

    n_samples = min(n_samples, len(train_views))
    sample_ids = torch.linspace(0, len(train_views) - 1, steps=n_samples).long().tolist()

    delta_sum = torch.zeros_like(base_xyz)
    attenuation_delta_sum = torch.zeros_like(base_attenuation_raw)

    for idx in sample_ids:
        _, _, delta, attenuation_delta = init_model(
            base_xyz,
            base_attenuation_raw,
            train_views[idx].T_tx.to(device=base_xyz.device, dtype=base_xyz.dtype),
            rx_context,
        )
        delta_sum += delta
        attenuation_delta_sum += attenuation_delta

    final_delta = delta_sum / float(n_samples)
    final_attenuation_delta = attenuation_delta_sum / float(n_samples)
    final_xyz = base_xyz + final_delta
    final_attenuation_raw = base_attenuation_raw + final_attenuation_delta

    return final_xyz, final_attenuation_raw, final_delta, final_attenuation_delta


def run_triplane_init_warmup(scene, gaussians, model_args, optim_args, pipe_args, render_fn):
    if not getattr(model_args, "use_triplane_init", False):
        return

    warmup_iters = int(getattr(model_args, "triplane_warmup_iters", 0))
    if warmup_iters <= 0:
        return

    base_xyz = gaussians.get_xyz.detach()
    xyz_min = base_xyz.min(dim=0).values
    xyz_max = base_xyz.max(dim=0).values
    bbox_pad_scale = float(getattr(model_args, "triplane_bbox_pad_scale", 0.0))
    if bbox_pad_scale < 0.0:
        raise ValueError("triplane_bbox_pad_scale must be non-negative.")
    if bbox_pad_scale > 0.0:
        bbox_pad = (xyz_max - xyz_min) * bbox_pad_scale
        xyz_min = xyz_min - bbox_pad
        xyz_max = xyz_max + bbox_pad

    frequency = float(getattr(model_args, "frequency", 915.0e6))
    voxel_size_scale = float(getattr(model_args, "voxel_size_scale", 1.0))
    offset_scale = float(getattr(model_args, "triplane_offset_radius_scale", 0.5))
    offset_radius = (3.0e8 / frequency) * voxel_size_scale * offset_scale

    init_model = RFTriplaneInitializer(
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        channels=int(getattr(model_args, "triplane_channels", 16)),
        resolution=int(getattr(model_args, "triplane_resolution", 64)),
        hidden_dim=int(getattr(model_args, "triplane_hidden_dim", 64)),
        offset_radius=offset_radius,
        attenuation_radius=float(getattr(model_args, "triplane_att_radius", 0.1)),
        decoder_type=getattr(model_args, "triplane_decoder_type", "mlp"),
        fourier_frequencies=int(getattr(model_args, "triplane_fourier_freqs", 4)),
        film_layers=int(getattr(model_args, "triplane_film_layers", 3)),
        output_mode=getattr(model_args, "triplane_output_mode", "offset_att"),
        field_type=getattr(model_args, "triplane_field_type", "direct"),
        wavelet_levels=int(getattr(model_args, "triplane_wavelet_levels", 2)),
        wavelet_high_init=float(getattr(model_args, "triplane_wavelet_high_init", 0.0)),
    ).to(device=base_xyz.device, dtype=base_xyz.dtype)

    optimizer = _make_warmup_optimizer(init_model, gaussians, model_args, optim_args)
    train_views = scene.getTrainSpectrums()
    tx_context, rx_context = _scene_context(scene, base_xyz.device, base_xyz.dtype)
    update_gaussian_attrs = getattr(model_args, "triplane_update_gaussian_attrs", False)
    use_view_tx = getattr(model_args, "triplane_use_view_tx", True)

    lambda_ssim = optim_args.lambda_dssim
    lambda_fourier = optim_args.lambda_dfourier
    lambda_offset = float(getattr(model_args, "triplane_offset_l2", 0.05))
    lambda_attenuation = float(getattr(model_args, "triplane_att_l2", 1.0e-3))
    lambda_wavelet = float(getattr(model_args, "triplane_wavelet_l1", 0.0))
    wavelet_c2f = bool(getattr(model_args, "triplane_wavelet_c2f", False))
    wavelet_levels = int(getattr(model_args, "triplane_wavelet_levels", 2))

    print(
        "\n[TriPlane Init] field={} decoder={} output_mode={} warmup_iters={} resolution={} channels={} hidden={} bbox_pad_scale={} offset_radius={:.4f}m att_radius={} offset_l2={} att_l2={} wavelet_l1={} wavelet_c2f={} view_tx={} update_gaussian_attrs={}\n".format(
            getattr(model_args, "triplane_field_type", "direct"),
            getattr(model_args, "triplane_decoder_type", "mlp"),
            getattr(model_args, "triplane_output_mode", "offset_att"),
            warmup_iters,
            int(getattr(model_args, "triplane_resolution", 64)),
            int(getattr(model_args, "triplane_channels", 16)),
            int(getattr(model_args, "triplane_hidden_dim", 64)),
            bbox_pad_scale,
            offset_radius,
            float(getattr(model_args, "triplane_att_radius", 0.1)),
            lambda_offset,
            lambda_attenuation,
            lambda_wavelet,
            wavelet_c2f,
            use_view_tx,
            update_gaussian_attrs,
        )
    )

    for step in range(1, warmup_iters + 1):
        if wavelet_c2f:
            # Keep an LL-only phase, then give each high-frequency level
            # a real optimization window during warmup.
            active_levels = min(
                wavelet_levels,
                ((step - 1) * (wavelet_levels + 1)) // max(1, warmup_iters),
            )
            init_model.set_wavelet_active_levels(active_levels)

        viewpoint = train_views[random.randrange(len(train_views))]

        base_xyz = gaussians.get_xyz.detach()
        base_attenuation = gaussians._attenuation if update_gaussian_attrs else gaussians._attenuation.detach()
        tx_step = viewpoint.T_tx.to(device=base_xyz.device, dtype=base_xyz.dtype) if use_view_tx else tx_context

        xyz_init, attenuation_raw_init, delta_xyz, attenuation_delta = init_model(
            base_xyz,
            base_attenuation,
            tx_step,
            rx_context,
        )

        render_pkg = render_fn(
            viewpoint,
            gaussians,
            pipe_args,
            override_xyz=xyz_init.contiguous(),
            override_attenuation=torch.sigmoid(attenuation_raw_init),
        )

        spectrum = render_pkg["render"]
        gt_spectrum = viewpoint.spectrum.to(device=spectrum.device, dtype=spectrum.dtype)

        ll1 = l1_loss(spectrum, gt_spectrum)
        ssim_loss = 1.0 - ssim(spectrum.unsqueeze(0).unsqueeze(0), gt_spectrum.unsqueeze(0).unsqueeze(0))
        lf = fourier_loss(spectrum, gt_spectrum)
        recon_loss = (1.0 - lambda_ssim - lambda_fourier) * ll1 + lambda_ssim * ssim_loss + lambda_fourier * lf
        offset_reg = (delta_xyz / max(offset_radius, 1.0e-6)).pow(2).sum(dim=-1).mean()
        attenuation_reg = attenuation_delta.pow(2).mean()
        wavelet_reg = init_model.wavelet_l1_loss()
        loss = (
            recon_loss
            + lambda_offset * offset_reg
            + lambda_attenuation * attenuation_reg
            + lambda_wavelet * wavelet_reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(1, warmup_iters // 5) == 0:
            mean_offset = delta_xyz.norm(dim=-1).mean().item()
            print(
                "[TriPlane Init] step {}/{} loss={:.6f} recon={:.6f} offset_reg={:.6f} att_reg={:.6f} wavelet_reg={:.6f} active_wavelet_levels={} mean_offset={:.4f}m".format(
                    step,
                    warmup_iters,
                    loss.item(),
                    recon_loss.item(),
                    offset_reg.item(),
                    attenuation_reg.item(),
                    wavelet_reg.item(),
                    getattr(getattr(init_model, "triplane", None), "active_levels", "-"),
                    mean_offset,
                )
            )

    with torch.no_grad():
        init_model.set_wavelet_active_levels()
        base_xyz = gaussians.get_xyz.detach()
        base_attenuation = gaussians._attenuation.detach()
        final_xyz, final_attenuation_raw, final_delta, final_attenuation_delta = _finalize_with_tx_average(
            init_model,
            gaussians,
            train_views,
            base_xyz,
            base_attenuation,
            tx_context,
            rx_context,
            model_args,
        )
        gaussians.apply_initial_xyz_attenuation(final_xyz, final_attenuation_raw)

        attenuation_delta_abs = final_attenuation_delta.abs()
        print(
            "[TriPlane Init] materialized: mean_offset={:.4f}m max_offset={:.4f}m mean_att_delta={:.4f} max_att_delta={:.4f}\n".format(
                final_delta.norm(dim=-1).mean().item(),
                final_delta.norm(dim=-1).max().item(),
                attenuation_delta_abs.mean().item(),
                attenuation_delta_abs.max().item(),
            )
        )
