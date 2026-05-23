import random

import torch
from torch import nn

from simple_knn._C import distCUDA2
from utils.loss_utils import l1_loss, ssim, fourier_loss

from .encoding import inverse_sigmoid, scene_bounds_from_gaussians
from .rf_spectrum_encoder import RFSpectrumEncoder
from .rf_geometry_encoder import RFGeometryEncoder
from .rf_point_decoder import RFPointDecoder, build_rf_ray_seeds, point_to_local_spectrum
from .rf_geometry_aware_triplane import RFGeometryAwareTriplaneEncoder
from .rf_triplane_field import RFTriplaneField
from .rf_gaussian_decoder import RFGaussianDecoder


class TGSRFInitializer(nn.Module):
    def __init__(self, gaussians, train_views, model_args, pipe_args):
        super().__init__()
        base_xyz = gaussians.get_xyz.detach()
        device, dtype = base_xyz.device, base_xyz.dtype
        pad = float(getattr(model_args, "tgsrf_bbox_pad_scale", 0.02))
        xyz_min, xyz_max = scene_bounds_from_gaussians(gaussians, pad)
        self.register_buffer("xyz_min", xyz_min)
        self.register_buffer("xyz_max", xyz_max)

        self.hidden_dim = int(getattr(model_args, "tgsrf_hidden_dim", 128))
        self.mode = str(getattr(model_args, "tgsrf_mode", "refine")).lower()
        self.frequency = float(getattr(model_args, "frequency", 915.0e6))
        self.ray_origin = str(getattr(model_args, "tgsrf_ray_origin", "tx")).lower()
        scene_scale = float((xyz_max - xyz_min).norm().item())

        if self.mode == "refine":
            seed_xyz = base_xyz.detach().clone()
            self.num_points = seed_xyz.shape[0]
        elif self.mode == "generate":
            self.num_points = int(getattr(model_args, "tgsrf_num_points", 12000))
            seed_xyz = build_rf_ray_seeds(
                train_views,
                xyz_min,
                xyz_max,
                num_points=self.num_points,
                seed_views=int(getattr(model_args, "tgsrf_seed_views", 32)),
                samples_per_ray=int(getattr(model_args, "tgsrf_samples_per_ray", 8)),
                ray_origin=self.ray_origin,
                min_depth=float(getattr(model_args, "tgsrf_min_depth", 0.05)),
                ray_fraction=float(getattr(model_args, "tgsrf_ray_seed_fraction", 0.45)),
                path_fraction=float(getattr(model_args, "tgsrf_path_seed_fraction", 0.25)),
                jitter_scale=float(getattr(model_args, "tgsrf_path_jitter_scale", 0.05)),
            )
        else:
            raise ValueError(f"Unknown tgsrf_mode={self.mode}; expected refine or generate.")
        self.register_buffer("seed_xyz", seed_xyz)

        self.point_latent = nn.Parameter(0.01 * torch.randn(self.num_points, self.hidden_dim, device=device, dtype=dtype))
        n_coeffs = (gaussians.max_fle_degree + 1) ** 2
        refine_features = bool(getattr(model_args, "tgsrf_refine_features", False))
        if self.mode == "refine":
            feature_init = gaussians.get_features.detach().clone()
        else:
            fle_scale = float(getattr(model_args, "tgsrf_fle_init_scale", getattr(model_args, "_fle_init_scale", 0.05)))
            feature_init = fle_scale * torch.randn(self.num_points, n_coeffs, gaussians.num_channels, device=device, dtype=dtype)
        self.fle_features = nn.Parameter(feature_init, requires_grad=refine_features or self.mode == "generate")

        train_base_attrs = bool(getattr(model_args, "tgsrf_train_base_attrs", False))
        if self.mode == "refine":
            attenuation_raw = gaussians._attenuation.detach().clone()
            scaling_raw = gaussians._scaling.detach().clone()
            rot = gaussians._rotation.detach().clone()
        else:
            init_att = float(getattr(model_args, "tgsrf_init_attenuation", 0.08))
            attenuation_raw = inverse_sigmoid(torch.full((self.num_points, 1), init_att, device=device, dtype=dtype))
            init_scale = float(getattr(model_args, "tgsrf_init_scale", 0.0))
            if init_scale > 0.0:
                scaling_raw = torch.log(torch.full((self.num_points, 3), init_scale, device=device, dtype=dtype))
            else:
                dist2 = torch.clamp_min(distCUDA2(seed_xyz.float()), 1.0e-7).to(device=device, dtype=dtype)
                scaling_raw = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            rot = torch.zeros(self.num_points, 4, device=device, dtype=dtype)
            rot[:, 0] = 1.0
        self.base_attenuation_raw = nn.Parameter(attenuation_raw, requires_grad=train_base_attrs or self.mode == "generate")
        self.base_scaling_raw = nn.Parameter(scaling_raw, requires_grad=train_base_attrs or self.mode == "generate")
        self.register_buffer("base_rotation_raw", rot)

        self.spectrum_encoder = RFSpectrumEncoder(
            hidden_dim=self.hidden_dim,
            patch_stride=int(getattr(model_args, "tgsrf_spectrum_patch_stride", 8)),
            txrx_freqs=int(getattr(model_args, "tgsrf_txrx_freqs", 4)),
        )
        self.geometry_encoder = RFGeometryEncoder(
            hidden_dim=self.hidden_dim,
            num_frequencies=int(getattr(model_args, "tgsrf_geometry_freqs", 4)),
            frequency_hz=self.frequency,
            scene_scale=scene_scale,
        )
        self.point_decoder = RFPointDecoder(
            hidden_dim=self.hidden_dim,
            geometry_dim=self.hidden_dim,
            local_dim=self.hidden_dim,
            num_layers=int(getattr(model_args, "tgsrf_point_layers", 2)),
            num_heads=int(getattr(model_args, "tgsrf_transformer_heads", 4)),
            offset_radius=float(getattr(model_args, "tgsrf_point_offset_radius", 0.25)),
        )
        tri_channels = int(getattr(model_args, "tgsrf_triplane_channels", 32))
        self.triplane = RFTriplaneField(
            xyz_min,
            xyz_max,
            channels=tri_channels,
            resolution=int(getattr(model_args, "tgsrf_triplane_resolution", 128)),
            field_type=getattr(model_args, "tgsrf_triplane_type", "direct"),
            levels=int(getattr(model_args, "tgsrf_wavelet_levels", 2)),
            high_init=float(getattr(model_args, "tgsrf_wavelet_high_init", 0.0)),
        )
        self.geometry_triplane = RFGeometryAwareTriplaneEncoder(
            hidden_dim=self.hidden_dim,
            channels=tri_channels,
            resolution=int(getattr(model_args, "tgsrf_triplane_resolution", 128)),
        )
        self.gaussian_decoder = RFGaussianDecoder(
            hidden_dim=self.hidden_dim,
            triplane_channels=tri_channels,
            geometry_dim=self.hidden_dim,
            local_dim=self.hidden_dim,
            offset_radius=float(getattr(model_args, "tgsrf_gaussian_offset_radius", 0.05 if self.mode == "refine" else 0.15)),
            att_radius=float(getattr(model_args, "tgsrf_att_radius", 0.05)),
            scale_radius=float(getattr(model_args, "tgsrf_scale_radius", 0.10)),
        )

    def normalize_xyz(self, xyz):
        denom = (self.xyz_max - self.xyz_min).clamp(min=1.0e-6)
        return ((xyz - self.xyz_min) / denom).clamp(0.0, 1.0) * 2.0 - 1.0

    def _origin_for_view(self, viewpoint, device, dtype):
        tx = viewpoint.T_tx.to(device=device, dtype=dtype)
        rx = viewpoint.T_rx.to(device=device, dtype=dtype)
        origin = tx if self.ray_origin == "tx" else rx
        return tx, rx, origin

    def set_wavelet_active_levels(self, active_levels=None):
        self.triplane.set_active_levels(active_levels)

    def wavelet_l1_loss(self):
        return self.triplane.wavelet_l1_loss()

    def predict_params(self, viewpoint, enable_point_offset=True, enable_gaussian_offset=True):
        seed_xyz = self.seed_xyz
        device, dtype = seed_xyz.device, seed_xyz.dtype
        tx, rx, origin = self._origin_for_view(viewpoint, device, dtype)
        spectrum = viewpoint.spectrum.to(device=device, dtype=dtype)
        spectrum_pack = self.spectrum_encoder(spectrum, tx, rx)

        seed_geom = self.geometry_encoder(seed_xyz, tx, rx)
        seed_local = point_to_local_spectrum(spectrum_pack["feature_map"], seed_xyz, origin)
        point_xyz, point_delta, point_features = self.point_decoder(
            seed_xyz,
            self.normalize_xyz(seed_xyz),
            self.point_latent,
            seed_geom,
            seed_local,
            spectrum_pack["tokens"],
            enable_offset=enable_point_offset,
        )

        point_geom = self.geometry_encoder(point_xyz, tx, rx)
        point_local = point_to_local_spectrum(spectrum_pack["feature_map"], point_xyz, origin)
        _, xyz_norm = self.triplane.query(point_xyz)
        geometry_planes = self.geometry_triplane(point_features, xyz_norm)
        triplane_features, _ = self.triplane.query(point_xyz, extra_planes=geometry_planes)
        decoded = self.gaussian_decoder(
            point_features,
            triplane_features,
            point_geom,
            point_local,
            spectrum_pack["global"],
        )

        gaussian_delta = decoded["delta_xyz"] if enable_gaussian_offset else torch.zeros_like(decoded["delta_xyz"])
        final_xyz = point_xyz + gaussian_delta
        final_xyz = torch.minimum(torch.maximum(final_xyz, self.xyz_min.view(1, 3)), self.xyz_max.view(1, 3))
        attenuation_raw = self.base_attenuation_raw + decoded["attenuation_delta"]
        scaling_raw = self.base_scaling_raw + decoded["log_scale_delta"]
        attenuation_raw = attenuation_raw.clamp(-8.0, 2.0)
        scaling_raw = scaling_raw.clamp(-6.0, 0.0)
        return {
            "xyz": final_xyz,
            "attenuation_raw": attenuation_raw,
            "attenuation": torch.sigmoid(attenuation_raw),
            "scaling_raw": scaling_raw,
            "scaling": torch.exp(scaling_raw),
            "rotation_raw": self.base_rotation_raw,
            "features": self.fle_features,
            "point_delta": point_delta,
            "gaussian_delta": gaussian_delta,
            "attenuation_delta": decoded["attenuation_delta"],
            "log_scale_delta": decoded["log_scale_delta"],
        }


def _make_optimizer(model, model_args):
    lr = float(getattr(model_args, "tgsrf_lr", 2.0e-4))
    return torch.optim.Adam(model.parameters(), lr=lr, eps=1.0e-15)


def _render_loss(pred, gt, optim_args):
    ll1 = l1_loss(pred, gt)
    ssim_loss = 1.0 - ssim(pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))
    lf = fourier_loss(pred, gt)
    lambda_ssim = optim_args.lambda_dssim
    lambda_fourier = optim_args.lambda_dfourier
    loss = (1.0 - lambda_ssim - lambda_fourier) * ll1 + lambda_ssim * ssim_loss + lambda_fourier * lf
    return loss, ll1, ssim_loss, lf


def run_tgsrf_init_warmup(scene, gaussians, model_args, optim_args, pipe_args, render_fn):
    if not bool(getattr(model_args, "use_tgsrf_initializer", False)):
        return None
    warmup_iters = int(getattr(model_args, "tgsrf_warmup_iters", 0))
    if warmup_iters <= 0:
        return None

    train_views = scene.getTrainSpectrums()
    model = TGSRFInitializer(gaussians, train_views, model_args, pipe_args).cuda()
    optimizer = _make_optimizer(model, model_args)

    lambda_point = float(getattr(model_args, "tgsrf_point_offset_l2", 0.02))
    lambda_gauss = float(getattr(model_args, "tgsrf_gaussian_offset_l2", 0.02))
    lambda_att = float(getattr(model_args, "tgsrf_att_l2", 0.02))
    lambda_scale = float(getattr(model_args, "tgsrf_scale_l2", 0.01))
    lambda_wavelet = float(getattr(model_args, "tgsrf_wavelet_l1", 0.0))
    grad_clip = float(getattr(model_args, "tgsrf_grad_clip", 1.0))
    point_start = int(getattr(model_args, "tgsrf_point_refine_start", max(1, warmup_iters + 1)))
    gaussian_offset_start = int(getattr(model_args, "tgsrf_gaussian_offset_start", max(1, warmup_iters // 2)))
    skipped_steps = 0
    wavelet_c2f = bool(getattr(model_args, "tgsrf_wavelet_c2f", False))
    wavelet_levels = int(getattr(model_args, "tgsrf_wavelet_levels", 2))

    print(
        "\n[TGS-RF Init] mode={} warmup_iters={} points={} ray_origin={} spectrum_encoder=conv_transformer triplane={} resolution={} channels={} geometry_aware=True\n".format(
            model.mode,
            warmup_iters,
            model.num_points,
            model.ray_origin,
            getattr(model_args, "tgsrf_triplane_type", "direct"),
            int(getattr(model_args, "tgsrf_triplane_resolution", 128)),
            int(getattr(model_args, "tgsrf_triplane_channels", 32)),
        )
    )

    for step in range(1, warmup_iters + 1):
        if wavelet_c2f:
            active = min(wavelet_levels, ((step - 1) * (wavelet_levels + 1)) // max(1, warmup_iters))
            model.set_wavelet_active_levels(active)

        viewpoint = train_views[random.randrange(len(train_views))]
        params = model.predict_params(
            viewpoint,
            enable_point_offset=step >= point_start,
            enable_gaussian_offset=step >= gaussian_offset_start,
        )
        render_pkg = render_fn(
            viewpoint,
            gaussians,
            pipe_args,
            override_xyz=params["xyz"].contiguous(),
            override_attenuation=params["attenuation"].contiguous(),
            override_scaling=params["scaling"].contiguous(),
            override_features=params["features"].contiguous(),
            override_rotation=params["rotation_raw"].contiguous(),
        )
        pred = render_pkg["render"]
        gt = viewpoint.spectrum.to(device=pred.device, dtype=pred.dtype)
        recon_loss, ll1, ssim_loss, lf = _render_loss(pred, gt, optim_args)
        point_reg = params["point_delta"].pow(2).sum(dim=-1).mean()
        gauss_reg = params["gaussian_delta"].pow(2).sum(dim=-1).mean()
        att_reg = params["attenuation_delta"].pow(2).mean()
        scale_reg = params["log_scale_delta"].pow(2).mean()
        wavelet_reg = model.wavelet_l1_loss()
        loss = (
            recon_loss
            + lambda_point * point_reg
            + lambda_gauss * gauss_reg
            + lambda_att * att_reg
            + lambda_scale * scale_reg
            + lambda_wavelet * wavelet_reg
        )

        if not torch.isfinite(loss):
            skipped_steps += 1
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % max(1, warmup_iters // 5) == 0:
                print(f"[TGS-RF Init] step {step}/{warmup_iters} skipped non-finite loss")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        with torch.no_grad():
            model.base_attenuation_raw.clamp_(-8.0, 2.0)
            model.base_scaling_raw.clamp_(-6.0, 0.0)
            model.fle_features.nan_to_num_(0.0, posinf=0.0, neginf=0.0)

        if step == 1 or step % max(1, warmup_iters // 5) == 0:
            mean_delta = (params["point_delta"] + params["gaussian_delta"]).norm(dim=-1).mean().item()
            print(
                "[TGS-RF Init] step {}/{} loss={:.6f} recon={:.6f} l1={:.6f} ssim={:.6f} fourier={:.6f} point_reg={:.6f} gauss_reg={:.6f} att_reg={:.6f} scale_reg={:.6f} wavelet_reg={:.6f} mean_delta={:.4f}m".format(
                    step,
                    warmup_iters,
                    loss.item(),
                    recon_loss.item(),
                    ll1.item(),
                    ssim_loss.item(),
                    lf.item(),
                    point_reg.item(),
                    gauss_reg.item(),
                    att_reg.item(),
                    scale_reg.item(),
                    wavelet_reg.item(),
                    mean_delta,
                )
            )

    with torch.no_grad():
        model.set_wavelet_active_levels()
        n_finalize = min(len(train_views), int(getattr(model_args, "tgsrf_finalize_views", 16)))
        if n_finalize < len(train_views):
            ids = torch.linspace(0, len(train_views) - 1, n_finalize).round().long().tolist()
            final_views = [train_views[i] for i in ids]
        else:
            final_views = list(train_views)

        xyz_list, att_list, scale_list = [], [], []
        delta_norms = []
        for view in final_views:
            params = model.predict_params(view, enable_point_offset=True, enable_gaussian_offset=True)
            xyz_list.append(params["xyz"])
            att_list.append(params["attenuation_raw"])
            scale_list.append(params["scaling_raw"])
            delta_norms.append((params["point_delta"] + params["gaussian_delta"]).norm(dim=-1).mean())

        final_xyz = torch.stack(xyz_list, dim=0).mean(dim=0)
        final_att_raw = torch.stack(att_list, dim=0).mean(dim=0)
        final_scale_raw = torch.stack(scale_list, dim=0).mean(dim=0)
        if not (torch.isfinite(final_xyz).all() and torch.isfinite(final_att_raw).all() and torch.isfinite(final_scale_raw).all()):
            print("[TGS-RF Init] non-finite final parameters; skip materialization and keep original GaussianModel.\n")
            return model
        gaussians.apply_initial_xyz_attenuation(
            final_xyz,
            final_att_raw,
            scaling_raw=final_scale_raw,
            rotation_raw=model.base_rotation_raw,
            features=model.fle_features,
        )
        print(
            "[TGS-RF Init] materialized: points={} finalize_views={} mean_delta={:.4f}m mean_attenuation={:.4f} mean_scale={:.4f}\n".format(
                final_xyz.shape[0],
                len(final_views),
                torch.stack(delta_norms).mean().item(),
                torch.sigmoid(final_att_raw).mean().item(),
                torch.exp(final_scale_raw).mean().item(),
            )
        )
        if skipped_steps > 0:
            print(f"[TGS-RF Init] skipped_nonfinite_steps={skipped_steps}")

    return model
