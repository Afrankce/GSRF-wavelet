import random

import torch
from torch import nn
import torch.nn.functional as F

from utils.loss_utils import l1_loss, ssim, fourier_loss


def _normalize_quaternion(q):
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)


def _axis_angle_to_quaternion(axis_angle):
    angle = axis_angle.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    axis = axis_angle / angle.clamp(min=1.0e-8)
    xyz = axis * torch.sin(half_angle)
    xyz = torch.where(angle < 1.0e-8, 0.5 * axis_angle, xyz)
    return _normalize_quaternion(torch.cat([torch.cos(half_angle), xyz], dim=-1))


def _quaternion_multiply(q_left, q_right):
    w1, x1, y1, z1 = q_left.unbind(dim=-1)
    w2, x2, y2, z2 = q_right.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


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

    def __init__(self, tri_dim, geom_dim, hidden_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tri_dim + geom_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, int(output_dim)),
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

    def __init__(self, tri_dim, geom_dim, hidden_dim, num_layers=3, num_frequencies=4, output_dim=4):
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
        self.out = nn.Linear(hidden_dim, int(output_dim))
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


class RFSeparateFiLMFourierDecoder(nn.Module):
    """Independent FiLM-Fourier decoders for different Gaussian parameter types."""

    def __init__(
        self,
        tri_dim,
        geom_dim,
        hidden_dim,
        output_mode,
        candidate_dim=0,
        num_layers=3,
        num_frequencies=4,
        predict_rotation=False,
        feature_dim=1,
    ):
        super().__init__()
        self.output_mode = str(output_mode)
        self.candidate_dim = int(candidate_dim)
        self.predict_rotation = bool(predict_rotation)
        self.feature_dim = int(feature_dim)

        def make_branch(output_dim):
            return RFFiLMFourierDecoder(
                tri_dim=tri_dim,
                geom_dim=geom_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_frequencies=num_frequencies,
                output_dim=output_dim,
            )

        if self.output_mode == "multi_param":
            self.xyz_head = make_branch(3)
        elif self.output_mode == "candidate_multi":
            self.candidate_head = make_branch(self.candidate_dim)
        else:
            raise ValueError(
                "film_fourier_multihead is currently intended for multi_param or candidate_multi modes."
            )

        self.attenuation_head = make_branch(1)
        self.scale_head = make_branch(3)
        self.rotation_head = make_branch(3) if self.predict_rotation else None
        self.feature_head = make_branch(self.feature_dim) if self.feature_dim > 0 else None

    def forward(self, tri_feat, geom_feat):
        tails = [
            self.attenuation_head(tri_feat, geom_feat),
            self.scale_head(tri_feat, geom_feat),
        ]
        if self.rotation_head is not None:
            tails.append(self.rotation_head(tri_feat, geom_feat))
        if self.feature_head is not None:
            tails.append(self.feature_head(tri_feat, geom_feat))

        if self.output_mode == "multi_param":
            return torch.cat([self.xyz_head(tri_feat, geom_feat), *tails], dim=-1)

        return torch.cat(
            [
                self.candidate_head(tri_feat, geom_feat),
                *tails,
            ],
            dim=-1,
        )


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
        candidate_pattern="axis7",
        candidate_temperature=1.0,
        scale_radius=0.1,
        feature_radius=0.1,
        rotation_radius=0.05,
        predict_rotation=False,
        feature_mode="scalar",
        max_fle_degree=9,
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
        self.scale_radius = float(scale_radius)
        self.feature_radius = float(feature_radius)
        self.rotation_radius = float(rotation_radius)
        self.predict_rotation = bool(predict_rotation)
        self.decoder_type = str(decoder_type)
        self.output_mode = str(output_mode)
        if self.output_mode not in {"offset_att", "gating_only", "candidate_score", "multi_param", "candidate_multi"}:
            raise ValueError(f"Unknown triplane output mode: {self.output_mode}")
        if self.predict_rotation and self.output_mode not in {"multi_param", "candidate_multi"}:
            raise ValueError("triplane_predict_rotation requires multi_param or candidate_multi output mode.")
        self.max_fle_degree = int(max_fle_degree)
        self.feature_mode = str(feature_mode).lower()
        if self.feature_mode not in {"none", "scalar", "degree"}:
            raise ValueError("triplane_feature_mode must be one of: none, scalar, degree.")
        if self.output_mode in {"multi_param", "candidate_multi"}:
            if self.feature_mode == "scalar":
                self.feature_dim = 1
            elif self.feature_mode == "degree":
                self.feature_dim = self.max_fle_degree + 1
            else:
                self.feature_dim = 0
        else:
            self.feature_dim = 0
        degree_ids = []
        for degree in range(self.max_fle_degree + 1):
            degree_ids.extend([degree] * (2 * degree + 1))
        self.register_buffer("fle_degree_ids", torch.tensor(degree_ids, dtype=torch.long))
        self.candidate_temperature = float(candidate_temperature)
        if self.candidate_temperature <= 0.0:
            raise ValueError("triplane_candidate_temperature must be positive.")
        self.candidate_pattern = str(candidate_pattern)
        self.register_buffer("candidate_offsets", self._make_candidate_offsets(self.candidate_pattern))

        geom_dim = 18
        self.use_plane_features = not self.decoder_type.endswith("_no_triplane")
        base_decoder_type = self.decoder_type.removesuffix("_no_triplane")
        tri_dim = int(channels) * 3 if self.use_plane_features else 0
        if self.output_mode == "candidate_score":
            output_dim = int(self.candidate_offsets.shape[0]) + 1
        elif self.output_mode == "multi_param":
            output_dim = 7 + (3 if self.predict_rotation else 0) + self.feature_dim
        elif self.output_mode == "candidate_multi":
            output_dim = int(self.candidate_offsets.shape[0]) + 4 + (3 if self.predict_rotation else 0) + self.feature_dim
        else:
            output_dim = 4

        if base_decoder_type == "mlp":
            self.head = RFMLPDecoder(
                tri_dim=tri_dim,
                geom_dim=geom_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
            )
        elif base_decoder_type == "film_fourier":
            self.head = RFFiLMFourierDecoder(
                tri_dim=tri_dim,
                geom_dim=geom_dim,
                hidden_dim=hidden_dim,
                num_layers=film_layers,
                num_frequencies=fourier_frequencies,
                output_dim=output_dim,
            )
        elif base_decoder_type in {"film_fourier_multihead", "film_fourier_separate"}:
            self.head = RFSeparateFiLMFourierDecoder(
                tri_dim=tri_dim,
                geom_dim=geom_dim,
                hidden_dim=hidden_dim,
                output_mode=self.output_mode,
                candidate_dim=int(self.candidate_offsets.shape[0]),
                num_layers=film_layers,
                num_frequencies=fourier_frequencies,
                predict_rotation=self.predict_rotation,
                feature_dim=self.feature_dim,
            )
        else:
            raise ValueError(f"Unknown triplane decoder type: {self.decoder_type}")

    @staticmethod
    def _make_candidate_offsets(pattern):
        pattern = str(pattern).lower()
        if pattern in {"axis7", "axis_7", "cross7"}:
            offsets = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        elif pattern in {"cube27", "cube_27"}:
            offsets = [
                [float(dx), float(dy), float(dz)]
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            ]
        else:
            raise ValueError(f"Unknown triplane candidate pattern: {pattern}")
        return torch.tensor(offsets, dtype=torch.float32)

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

    def _apply_rotation_delta(self, base_rotation_raw, rotation_delta):
        if base_rotation_raw is None or not self.predict_rotation:
            return None
        base_q = _normalize_quaternion(base_rotation_raw)
        delta_q = _axis_angle_to_quaternion(rotation_delta)
        return _normalize_quaternion(_quaternion_multiply(delta_q, base_q))

    def _apply_feature_gain(self, base_features, feature_log_gain):
        if base_features is None:
            return None
        if self.feature_dim <= 0 or self.feature_mode == "none":
            return base_features
        if self.feature_mode == "scalar":
            return base_features * torch.exp(feature_log_gain).view(-1, 1, 1)
        degree_ids = self.fle_degree_ids[: base_features.shape[1]].to(device=base_features.device)
        coeff_gain = feature_log_gain[:, degree_ids]
        return base_features * torch.exp(coeff_gain).unsqueeze(-1)

    def predict_params(
        self,
        base_xyz,
        base_attenuation_raw,
        tx_context,
        rx_context,
        base_scaling=None,
        base_features=None,
        base_rotation_raw=None,
    ):
        if self.use_plane_features:
            tri_feat, xyz_norm = self.triplane(base_xyz)
        else:
            xyz_norm = self.triplane.normalize_xyz(base_xyz)
            tri_feat = base_xyz.new_empty((base_xyz.shape[0], 0))
        geom_feat = self._geometry_features(base_xyz, xyz_norm, tx_context, rx_context)

        pred = self.head(tri_feat, geom_feat)
        log_scale_delta = base_xyz.new_zeros(base_xyz.shape)
        rotation_delta = base_xyz.new_zeros(base_xyz.shape)
        feature_gain_dim = max(1, self.feature_dim)
        feature_log_gain = base_xyz.new_zeros((base_xyz.shape[0], feature_gain_dim))

        if self.output_mode in {"candidate_score", "candidate_multi"}:
            candidate_offsets = self.candidate_offsets.to(device=base_xyz.device, dtype=base_xyz.dtype)
            score_logits = pred[:, : candidate_offsets.shape[0]] / self.candidate_temperature
            candidate_probs = torch.softmax(score_logits, dim=-1)
            delta_xyz = self.offset_radius * (candidate_probs @ candidate_offsets)
            cursor = candidate_offsets.shape[0]
            attenuation_pred = pred[:, cursor : cursor + 1]
            cursor += 1
            if self.output_mode == "candidate_multi":
                log_scale_delta = self.scale_radius * torch.tanh(pred[:, cursor : cursor + 3])
                cursor += 3
                if self.predict_rotation:
                    rotation_delta = self.rotation_radius * torch.tanh(pred[:, cursor : cursor + 3])
                    cursor += 3
                if self.feature_dim > 0:
                    feature_log_gain = self.feature_radius * torch.tanh(pred[:, cursor : cursor + self.feature_dim])
        elif self.output_mode == "gating_only":
            delta_xyz = torch.zeros_like(base_xyz)
            attenuation_pred = pred[:, 3:4]
        elif self.output_mode == "multi_param":
            cursor = 0
            delta_xyz = self.offset_radius * torch.tanh(pred[:, cursor : cursor + 3])
            cursor += 3
            attenuation_pred = pred[:, cursor : cursor + 1]
            cursor += 1
            log_scale_delta = self.scale_radius * torch.tanh(pred[:, cursor : cursor + 3])
            cursor += 3
            if self.predict_rotation:
                rotation_delta = self.rotation_radius * torch.tanh(pred[:, cursor : cursor + 3])
                cursor += 3
            if self.feature_dim > 0:
                feature_log_gain = self.feature_radius * torch.tanh(pred[:, cursor : cursor + self.feature_dim])
        else:
            delta_xyz = self.offset_radius * torch.tanh(pred[:, :3])
            attenuation_pred = pred[:, 3:4]
        attenuation_delta = self.attenuation_radius * torch.tanh(attenuation_pred)
        attenuation_raw = base_attenuation_raw + attenuation_delta
        rotation_raw = self._apply_rotation_delta(base_rotation_raw, rotation_delta)
        scaling = None
        features = None
        if base_scaling is not None:
            scaling = base_scaling * torch.exp(log_scale_delta)
        features = self._apply_feature_gain(base_features, feature_log_gain)

        return {
            "xyz": base_xyz + delta_xyz,
            "attenuation_raw": attenuation_raw,
            "delta_xyz": delta_xyz,
            "attenuation_delta": attenuation_delta,
            "scaling": scaling,
            "log_scale_delta": log_scale_delta,
            "rotation_raw": rotation_raw,
            "rotation_delta": rotation_delta,
            "features": features,
            "feature_log_gain": feature_log_gain,
        }

    def forward(self, base_xyz, base_attenuation_raw, tx_context, rx_context):
        params = self.predict_params(base_xyz, base_attenuation_raw, tx_context, rx_context)

        return (
            params["xyz"],
            params["attenuation_raw"],
            params["delta_xyz"],
            params["attenuation_delta"],
        )

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


def build_triplane_initializer_from_gaussians(gaussians, model_args):
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

    return RFTriplaneInitializer(
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
        candidate_pattern=getattr(model_args, "triplane_candidate_pattern", "axis7"),
        candidate_temperature=float(getattr(model_args, "triplane_candidate_temperature", 1.0)),
        scale_radius=float(getattr(model_args, "triplane_scale_radius", 0.1)),
        feature_radius=float(getattr(model_args, "triplane_feature_radius", 0.1)),
        rotation_radius=float(getattr(model_args, "triplane_rotation_radius", 0.05)),
        predict_rotation=bool(getattr(model_args, "triplane_predict_rotation", False)),
        feature_mode=getattr(model_args, "triplane_feature_mode", "scalar"),
        max_fle_degree=int(getattr(model_args, "fle_degree", 9)),
    ).to(device=base_xyz.device, dtype=base_xyz.dtype)


def make_persistent_triplane_runtime(init_model, scene, gaussians, model_args):
    if init_model is None or not getattr(model_args, "triplane_persistent_modulation", False):
        return None

    tx_context, rx_context = _scene_context(scene, gaussians.get_xyz.device, gaussians.get_xyz.dtype)
    init_model.set_wavelet_active_levels()
    init_model.train()

    return {
        "model": init_model,
        "tx_context": tx_context.detach(),
        "rx_context": rx_context.detach(),
        "use_view_tx": bool(getattr(model_args, "triplane_use_view_tx", True)),
        "lambda_att": float(getattr(model_args, "triplane_persistent_att_l2", getattr(model_args, "triplane_att_l2", 0.0))),
        "lambda_wavelet": float(getattr(model_args, "triplane_persistent_wavelet_l1", getattr(model_args, "triplane_wavelet_l1", 0.0))),
    }


def load_persistent_triplane_runtime(scene, gaussians, model_args, state_path):
    if not getattr(model_args, "triplane_persistent_modulation", False):
        return None
    if not state_path:
        return None

    init_model = build_triplane_initializer_from_gaussians(gaussians, model_args)
    payload = torch.load(state_path, map_location=gaussians.get_xyz.device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    init_model.load_state_dict(state_dict)
    return make_persistent_triplane_runtime(init_model, scene, gaussians, model_args)


def render_with_persistent_triplane(viewpoint, gaussians, pipe_args, render_fn, runtime, return_regularizers=False):
    if runtime is None:
        render_pkg = render_fn(viewpoint, gaussians, pipe_args)
        if return_regularizers:
            zero = gaussians.get_xyz.new_zeros(())
            return render_pkg, {"att_reg": zero, "wavelet_reg": zero}
        return render_pkg

    init_model = runtime["model"]
    init_model.set_wavelet_active_levels()

    base_xyz = gaussians.get_xyz
    base_attenuation_raw = gaussians._attenuation
    tx_context = (
        viewpoint.T_tx.to(device=base_xyz.device, dtype=base_xyz.dtype)
        if runtime["use_view_tx"]
        else runtime["tx_context"].to(device=base_xyz.device, dtype=base_xyz.dtype)
    )
    rx_context = runtime["rx_context"].to(device=base_xyz.device, dtype=base_xyz.dtype)

    _, attenuation_raw, _, attenuation_delta = init_model(
        base_xyz,
        base_attenuation_raw,
        tx_context,
        rx_context,
    )

    render_pkg = render_fn(
        viewpoint,
        gaussians,
        pipe_args,
        override_attenuation=torch.sigmoid(attenuation_raw).contiguous(),
    )

    if return_regularizers:
        regularizers = {
            "att_reg": attenuation_delta.pow(2).mean(),
            "wavelet_reg": init_model.wavelet_l1_loss(),
        }
        return render_pkg, regularizers
    return render_pkg


def _finalize_with_tx_average(init_model, gaussians, train_views, base_xyz, base_attenuation_raw, tx_context, rx_context, model_args):
    use_view_tx = getattr(model_args, "triplane_use_view_tx", True)
    n_samples = int(getattr(model_args, "triplane_finalize_tx_samples", 32))
    base_scaling = gaussians.get_scaling.detach()
    base_features = gaussians.get_features.detach()
    base_rotation_raw = gaussians._rotation.detach()

    if (not use_view_tx) or n_samples <= 1 or len(train_views) <= 1:
        params = init_model.predict_params(
            base_xyz,
            base_attenuation_raw,
            tx_context,
            rx_context,
            base_scaling=base_scaling,
            base_features=base_features,
            base_rotation_raw=base_rotation_raw,
        )
        final_scaling_raw = gaussians._scaling.detach() + params["log_scale_delta"]
        final_features = params["features"] if params["features"] is not None else base_features
        final_rotation_raw = params["rotation_raw"]
        return (
            params["xyz"],
            params["attenuation_raw"],
            params["delta_xyz"],
            params["attenuation_delta"],
            final_scaling_raw,
            params["log_scale_delta"],
            final_rotation_raw,
            params["rotation_delta"],
            final_features,
            params["feature_log_gain"],
        )

    n_samples = min(n_samples, len(train_views))
    sample_ids = torch.linspace(0, len(train_views) - 1, steps=n_samples).long().tolist()

    delta_sum = torch.zeros_like(base_xyz)
    attenuation_delta_sum = torch.zeros_like(base_attenuation_raw)
    log_scale_delta_sum = torch.zeros_like(gaussians._scaling.detach())
    rotation_delta_sum = torch.zeros_like(base_xyz)
    feature_gain_dim = max(1, init_model.feature_dim)
    feature_log_gain_sum = torch.zeros((base_xyz.shape[0], feature_gain_dim), device=base_xyz.device, dtype=base_xyz.dtype)

    for idx in sample_ids:
        params = init_model.predict_params(
            base_xyz,
            base_attenuation_raw,
            train_views[idx].T_tx.to(device=base_xyz.device, dtype=base_xyz.dtype),
            rx_context,
            base_scaling=base_scaling,
            base_features=base_features,
            base_rotation_raw=base_rotation_raw,
        )
        delta_sum += params["delta_xyz"]
        attenuation_delta_sum += params["attenuation_delta"]
        log_scale_delta_sum += params["log_scale_delta"]
        rotation_delta_sum += params["rotation_delta"]
        feature_log_gain_sum += params["feature_log_gain"]

    final_delta = delta_sum / float(n_samples)
    final_attenuation_delta = attenuation_delta_sum / float(n_samples)
    final_log_scale_delta = log_scale_delta_sum / float(n_samples)
    final_rotation_delta = rotation_delta_sum / float(n_samples)
    final_feature_log_gain = feature_log_gain_sum / float(n_samples)
    final_xyz = base_xyz + final_delta
    final_attenuation_raw = base_attenuation_raw + final_attenuation_delta
    final_scaling_raw = gaussians._scaling.detach() + final_log_scale_delta
    final_rotation_raw = init_model._apply_rotation_delta(base_rotation_raw, final_rotation_delta)
    final_features = init_model._apply_feature_gain(base_features, final_feature_log_gain)

    return (
        final_xyz,
        final_attenuation_raw,
        final_delta,
        final_attenuation_delta,
        final_scaling_raw,
        final_log_scale_delta,
        final_rotation_raw,
        final_rotation_delta,
        final_features,
        final_feature_log_gain,
    )


def run_triplane_init_warmup(scene, gaussians, model_args, optim_args, pipe_args, render_fn):
    if not getattr(model_args, "use_triplane_init", False):
        return None

    warmup_iters = int(getattr(model_args, "triplane_warmup_iters", 0))
    if warmup_iters <= 0:
        return None

    base_xyz = gaussians.get_xyz.detach()
    init_model = build_triplane_initializer_from_gaussians(gaussians, model_args)
    offset_radius = init_model.offset_radius

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
    lambda_scale = float(getattr(model_args, "triplane_scale_l2", 0.0))
    lambda_rotation = float(getattr(model_args, "triplane_rotation_l2", 0.0))
    lambda_feature = float(getattr(model_args, "triplane_feature_l2", 0.0))
    wavelet_c2f = bool(getattr(model_args, "triplane_wavelet_c2f", False))
    wavelet_levels = int(getattr(model_args, "triplane_wavelet_levels", 2))
    bbox_pad_scale = float(getattr(model_args, "triplane_bbox_pad_scale", 0.0))

    print(
        "\n[TriPlane Init] field={} decoder={} output_mode={} warmup_iters={} resolution={} channels={} hidden={} bbox_pad_scale={} offset_radius={:.4f}m att_radius={} scale_radius={} rotation_radius={} feature_radius={} predict_rotation={} feature_mode={} offset_l2={} att_l2={} scale_l2={} rotation_l2={} feature_l2={} wavelet_l1={} wavelet_c2f={} view_tx={} update_gaussian_attrs={}\n".format(
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
            float(getattr(model_args, "triplane_scale_radius", 0.1)),
            float(getattr(model_args, "triplane_rotation_radius", 0.05)),
            float(getattr(model_args, "triplane_feature_radius", 0.1)),
            bool(getattr(model_args, "triplane_predict_rotation", False)),
            getattr(model_args, "triplane_feature_mode", "scalar"),
            lambda_offset,
            lambda_attenuation,
            lambda_scale,
            lambda_rotation,
            lambda_feature,
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
        base_scaling = gaussians.get_scaling.detach()
        base_features = gaussians.get_features.detach()
        base_rotation_raw = gaussians._rotation.detach()
        tx_step = viewpoint.T_tx.to(device=base_xyz.device, dtype=base_xyz.dtype) if use_view_tx else tx_context

        params = init_model.predict_params(
            base_xyz,
            base_attenuation,
            tx_step,
            rx_context,
            base_scaling=base_scaling,
            base_features=base_features,
            base_rotation_raw=base_rotation_raw,
        )

        render_pkg = render_fn(
            viewpoint,
            gaussians,
            pipe_args,
            override_xyz=params["xyz"].contiguous(),
            override_attenuation=torch.sigmoid(params["attenuation_raw"]),
            override_scaling=params["scaling"],
            override_features=params["features"],
            override_rotation=params["rotation_raw"],
        )

        spectrum = render_pkg["render"]
        gt_spectrum = viewpoint.spectrum.to(device=spectrum.device, dtype=spectrum.dtype)

        ll1 = l1_loss(spectrum, gt_spectrum)
        ssim_loss = 1.0 - ssim(spectrum.unsqueeze(0).unsqueeze(0), gt_spectrum.unsqueeze(0).unsqueeze(0))
        lf = fourier_loss(spectrum, gt_spectrum)
        recon_loss = (1.0 - lambda_ssim - lambda_fourier) * ll1 + lambda_ssim * ssim_loss + lambda_fourier * lf
        offset_reg = (params["delta_xyz"] / max(offset_radius, 1.0e-6)).pow(2).sum(dim=-1).mean()
        attenuation_reg = params["attenuation_delta"].pow(2).mean()
        scale_reg = params["log_scale_delta"].pow(2).mean()
        rotation_reg = (params["rotation_delta"] / max(init_model.rotation_radius, 1.0e-6)).pow(2).sum(dim=-1).mean()
        feature_reg = params["feature_log_gain"].pow(2).mean()
        wavelet_reg = init_model.wavelet_l1_loss()
        loss = (
            recon_loss
            + lambda_offset * offset_reg
            + lambda_attenuation * attenuation_reg
            + lambda_scale * scale_reg
            + lambda_rotation * rotation_reg
            + lambda_feature * feature_reg
            + lambda_wavelet * wavelet_reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(1, warmup_iters // 5) == 0:
            mean_offset = params["delta_xyz"].norm(dim=-1).mean().item()
            print(
                "[TriPlane Init] step {}/{} loss={:.6f} recon={:.6f} offset_reg={:.6f} att_reg={:.6f} scale_reg={:.6f} rotation_reg={:.6f} feature_reg={:.6f} wavelet_reg={:.6f} active_wavelet_levels={} mean_offset={:.4f}m".format(
                    step,
                    warmup_iters,
                    loss.item(),
                    recon_loss.item(),
                    offset_reg.item(),
                    attenuation_reg.item(),
                    scale_reg.item(),
                    rotation_reg.item(),
                    feature_reg.item(),
                    wavelet_reg.item(),
                    getattr(getattr(init_model, "triplane", None), "active_levels", "-"),
                    mean_offset,
                )
            )

    with torch.no_grad():
        init_model.set_wavelet_active_levels()
        base_xyz = gaussians.get_xyz.detach()
        base_attenuation = gaussians._attenuation.detach()
        (
            final_xyz,
            final_attenuation_raw,
            final_delta,
            final_attenuation_delta,
            final_scaling_raw,
            final_log_scale_delta,
            final_rotation_raw,
            final_rotation_delta,
            final_features,
            final_feature_log_gain,
        ) = _finalize_with_tx_average(
            init_model,
            gaussians,
            train_views,
            base_xyz,
            base_attenuation,
            tx_context,
            rx_context,
            model_args,
        )
        gaussians.apply_initial_xyz_attenuation(
            final_xyz,
            final_attenuation_raw,
            scaling_raw=final_scaling_raw,
            rotation_raw=final_rotation_raw,
            features=final_features,
        )

        attenuation_delta_abs = final_attenuation_delta.abs()
        log_scale_delta_abs = final_log_scale_delta.abs()
        rotation_delta_abs = final_rotation_delta.norm(dim=-1)
        feature_log_gain_abs = final_feature_log_gain.abs()
        print(
            "[TriPlane Init] materialized: mean_offset={:.4f}m max_offset={:.4f}m mean_att_delta={:.4f} max_att_delta={:.4f} mean_log_scale_delta={:.4f} max_log_scale_delta={:.4f} mean_rotation_delta={:.4f}rad max_rotation_delta={:.4f}rad mean_feature_log_gain={:.4f} max_feature_log_gain={:.4f}\n".format(
                final_delta.norm(dim=-1).mean().item(),
                final_delta.norm(dim=-1).max().item(),
                attenuation_delta_abs.mean().item(),
                attenuation_delta_abs.max().item(),
                log_scale_delta_abs.mean().item(),
                log_scale_delta_abs.max().item(),
                rotation_delta_abs.mean().item(),
                rotation_delta_abs.max().item(),
                feature_log_gain_abs.mean().item(),
                feature_log_gain_abs.max().item(),
            )
        )

    return init_model
