
# standard library
import os
import json
from datetime import datetime
from random import randint
from argparse import ArgumentParser, Namespace
from typing import List

# third-party
import numpy as np
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# project
from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state
from scene import Scene, GaussianModel
from gaussian_renderer.render_csi import render_csi
from scene.csi_dataset import load_csi_data
from scene.csi_model import CSIEncoder, CSIAutoDecoder
from scene.dataset_readers import SpectrumInfo
from utils.train_utils import setup_fle_only_optimizer, init_gaussians_from_reference


def capture_geometry_to_cpu_dict(gaussians):
    """Capture only geometry tensors to CPU for low-memory cross-antenna reuse."""
    return {
        'xyz': gaussians._xyz.detach().cpu().contiguous(),
        'scaling': gaussians._scaling.detach().cpu().contiguous(),
        'rotation': gaussians._rotation.detach().cpu().contiguous(),
        'attenuation': gaussians._attenuation.detach().cpu().contiguous(),
        'spatial_lr_scale': float(gaussians.spatial_lr_scale),
    }


def _build_wavelet_cfg(args):
    enabled = bool(getattr(args, 'wavelet_enabled', False))
    if not enabled:
        return None

    mode = str(getattr(args, 'wavelet_mode', 'none')).lower()
    if mode not in {'none', 'cdb4', 'cdb4_strict'}:
        mode = 'none'

    return {
        'enabled': True,
        'mode': mode,
        'use_lpr': bool(getattr(args, 'wavelet_use_lpr', False)),
        'use_lhf': bool(getattr(args, 'wavelet_use_lhf', False)),
        'lpr_weight': float(getattr(args, 'wavelet_lpr_weight', 0.05)),
        'lhf_weight': float(getattr(args, 'wavelet_lhf_weight', 0.1)),
        'lpr_warmup': int(getattr(args, 'wavelet_lpr_warmup_iters', 2000)),
        'lhf_warmup': int(getattr(args, 'wavelet_lhf_warmup_iters', 2000)),
    }


def _build_wavelet_alpha(mode: str, device, dtype):
    alpha_logit = torch.nn.Parameter(torch.tensor(-2.2, device=device, dtype=dtype), requires_grad=True)
    g_ref = torch.tensor(_DB4_G, device=device, dtype=dtype)
    if mode == 'cdb4_strict':
        g_ref_r, g_ref_i = _qmf_complex_taps(_CDB4_STRICT_H0_R, _CDB4_STRICT_H0_I)
        g_ref = torch.sqrt(torch.tensor(g_ref_r, device=device, dtype=dtype) ** 2 +
                           torch.tensor(g_ref_i, device=device, dtype=dtype) ** 2)
    return alpha_logit, g_ref


def _wavelet_l1_proxy_from_complex_channels(h_complex_channels: torch.Tensor) -> torch.Tensor:
    """Proxy for wavelet-domain sparsity (high-pass along subcarrier axis)."""
    if h_complex_channels.numel() <= 2:
        return h_complex_channels.new_tensor(0.0)
    hp = h_complex_channels[2:] - h_complex_channels[:-2]
    return hp.abs().mean()


# db4 analysis filter bank (8 taps)
_DB4_H = [
    0.16291765, 0.50547227, 0.44610042, -0.01980037,
    -0.13228957, 0.02178025, 0.02328089, -0.00749349
]
_DB4_G = [
    -0.00749349, -0.02328089, 0.02178025, 0.13228957,
    -0.01980037, -0.44610042, 0.50547227, -0.16291765
]

# strict complex cdb4 comparison bank (8 taps, fixed complex taps)
_CDB4_STRICT_H0_R = _DB4_H
_CDB4_STRICT_H0_I = [
    0.0121, -0.0184, 0.0219, -0.0150,
    0.0150, -0.0219, 0.0184, -0.0121
]


def _complex_soft_threshold(z: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    mag = torch.abs(z)
    scale = torch.relu(mag - lam) / (mag + eps)
    return z * scale


def _symmetric_pad_1d(x: torch.Tensor, pad_len: int) -> torch.Tensor:
    return F.pad(x, (pad_len, pad_len), mode='reflect')


def _make_real_filter(coeffs, device, dtype):
    return torch.tensor(coeffs, device=device, dtype=dtype).view(1, 1, -1)


def _complex_conv1d(xr, xi, hr, hi, stride=1):
    real = F.conv1d(xr, hr, stride=stride) - F.conv1d(xi, hi, stride=stride)
    imag = F.conv1d(xr, hi, stride=stride) + F.conv1d(xi, hr, stride=stride)
    return real, imag


def _qmf_complex_taps(real_coeffs, imag_coeffs):
    signs = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.float32)
    h0_r = torch.tensor(real_coeffs, dtype=torch.float32)
    h0_i = torch.tensor(imag_coeffs, dtype=torch.float32)
    g0_r = torch.flip(h0_r, dims=[0]) * signs
    g0_i = torch.flip(h0_i, dims=[0]) * signs
    return g0_r.tolist(), g0_i.tolist()


def _dwt_complex_db4(h_complex: torch.Tensor, alpha: torch.Tensor = None):
    """One-level DWT for complex CSI: input [B, N] or [N], output cA/cD."""
    squeeze_back = False
    if h_complex.dim() == 1:
        h_complex = h_complex.unsqueeze(0)
        squeeze_back = True
    elif h_complex.dim() != 2:
        raise ValueError(f"Expected complex CSI shape [N] or [B,N], got {tuple(h_complex.shape)}")

    B, N = h_complex.shape
    xr = h_complex.real.view(B, 1, N)
    xi = h_complex.imag.view(B, 1, N)

    h = _make_real_filter(_DB4_H, h_complex.device, h_complex.real.dtype)
    g_coeff = torch.tensor(_DB4_G, device=h_complex.device, dtype=h_complex.real.dtype)
    if alpha is not None:
        g_coeff = alpha.to(device=h_complex.device, dtype=h_complex.real.dtype) * g_coeff
    g = g_coeff.view(1, 1, -1)

    pad_len = 4
    xr_p = _symmetric_pad_1d(xr, pad_len)
    xi_p = _symmetric_pad_1d(xi, pad_len)

    cA_r = F.conv1d(xr_p, h, stride=2)
    cA_i = F.conv1d(xi_p, h, stride=2)
    cD_r = F.conv1d(xr_p, g, stride=2)
    cD_i = F.conv1d(xi_p, g, stride=2)

    cA = torch.complex(cA_r, cA_i).squeeze(1)
    cD = torch.complex(cD_r, cD_i).squeeze(1)

    if squeeze_back:
        return cA.squeeze(0), cD.squeeze(0)
    return cA, cD


def _idwt_complex_db4(cA: torch.Tensor, cD: torch.Tensor, output_len: int) -> torch.Tensor:
    """One-level IDWT for complex CSI with zero-insertion upsampling."""
    squeeze_back = False
    if cA.dim() == 1:
        cA = cA.unsqueeze(0)
        cD = cD.unsqueeze(0)
        squeeze_back = True

    B, L = cA.shape
    h = _make_real_filter(_DB4_H, cA.device, cA.real.dtype)
    g = _make_real_filter(_DB4_G, cA.device, cA.real.dtype)

    upA_r = torch.zeros((B, 1, L * 2), device=cA.device, dtype=cA.real.dtype)
    upA_i = torch.zeros((B, 1, L * 2), device=cA.device, dtype=cA.real.dtype)
    upD_r = torch.zeros((B, 1, L * 2), device=cA.device, dtype=cA.real.dtype)
    upD_i = torch.zeros((B, 1, L * 2), device=cA.device, dtype=cA.real.dtype)

    upA_r[:, :, ::2] = cA.real.view(B, 1, L)
    upA_i[:, :, ::2] = cA.imag.view(B, 1, L)
    upD_r[:, :, ::2] = cD.real.view(B, 1, L)
    upD_i[:, :, ::2] = cD.imag.view(B, 1, L)

    pad_len = 4
    yA_r = F.conv1d(_symmetric_pad_1d(upA_r, pad_len), h)
    yA_i = F.conv1d(_symmetric_pad_1d(upA_i, pad_len), h)
    yD_r = F.conv1d(_symmetric_pad_1d(upD_r, pad_len), g)
    yD_i = F.conv1d(_symmetric_pad_1d(upD_i, pad_len), g)

    y_r = (yA_r + yD_r).squeeze(1)
    y_i = (yA_i + yD_i).squeeze(1)

    y_r = y_r[:, :output_len]
    y_i = y_i[:, :output_len]
    y = torch.complex(y_r, y_i)
    return y.squeeze(0) if squeeze_back else y


def _dwt_complex_cdb4_strict(h_complex: torch.Tensor, alpha: torch.Tensor = None):
    """One-level strict complex DWT: complex analysis filters, complex convolution."""
    squeeze_back = False
    if h_complex.dim() == 1:
        h_complex = h_complex.unsqueeze(0)
        squeeze_back = True
    elif h_complex.dim() != 2:
        raise ValueError(f"Expected complex CSI shape [N] or [B,N], got {tuple(h_complex.shape)}")

    B, N = h_complex.shape
    xr = h_complex.real.view(B, 1, N)
    xi = h_complex.imag.view(B, 1, N)

    h0_r, h0_i = _make_real_filter(_CDB4_STRICT_H0_R, h_complex.device, h_complex.real.dtype), _make_real_filter(_CDB4_STRICT_H0_I, h_complex.device, h_complex.real.dtype)
    g0_r_coeffs, g0_i_coeffs = _qmf_complex_taps(_CDB4_STRICT_H0_R, _CDB4_STRICT_H0_I)
    g0_r_coeffs_t = torch.tensor(g0_r_coeffs, device=h_complex.device, dtype=h_complex.real.dtype)
    g0_i_coeffs_t = torch.tensor(g0_i_coeffs, device=h_complex.device, dtype=h_complex.real.dtype)
    if alpha is not None:
        a = alpha.to(device=h_complex.device, dtype=h_complex.real.dtype)
        g0_r_coeffs_t = a * g0_r_coeffs_t
        g0_i_coeffs_t = a * g0_i_coeffs_t
    g0_r, g0_i = g0_r_coeffs_t.view(1, 1, -1), g0_i_coeffs_t.view(1, 1, -1)

    pad_len = 4
    xr_p = _symmetric_pad_1d(xr, pad_len)
    xi_p = _symmetric_pad_1d(xi, pad_len)

    cA_r, cA_i = _complex_conv1d(xr_p, xi_p, h0_r, h0_i, stride=2)
    cD_r, cD_i = _complex_conv1d(xr_p, xi_p, g0_r, g0_i, stride=2)

    cA = torch.complex(cA_r, cA_i).squeeze(1)
    cD = torch.complex(cD_r, cD_i).squeeze(1)

    if squeeze_back:
        return cA.squeeze(0), cD.squeeze(0)
    return cA, cD


def _idwt_complex_cdb4_strict(cA: torch.Tensor, cD: torch.Tensor, output_len: int) -> torch.Tensor:
    """Strict complex IDWT with zero-insertion upsampling."""
    squeeze_back = False
    if cA.dim() == 1:
        cA = cA.unsqueeze(0)
        cD = cD.unsqueeze(0)
        squeeze_back = True

    B, L = cA.shape
    h0_r, h0_i = _make_real_filter(_CDB4_STRICT_H0_R, cA.device, cA.real.dtype), _make_real_filter(_CDB4_STRICT_H0_I, cA.device, cA.real.dtype)
    g0_r_coeffs, g0_i_coeffs = _qmf_complex_taps(_CDB4_STRICT_H0_R, _CDB4_STRICT_H0_I)
    g0_r, g0_i = _make_real_filter(g0_r_coeffs, cA.device, cA.real.dtype), _make_real_filter(g0_i_coeffs, cA.device, cA.real.dtype)

    upA_r = torch.zeros((B, 1, L * 2), device=cA.device, dtype=cA.real.dtype)
    upA_i = torch.zeros((B, 1, L * 2), device=cA.device, dtype=cA.real.dtype)
    upD_r = torch.zeros((B, 1, L * 2), device=cA.device, dtype=cA.real.dtype)
    upD_i = torch.zeros((B, 1, L * 2), device=cA.device, dtype=cA.real.dtype)

    upA_r[:, :, ::2] = cA.real.view(B, 1, L)
    upA_i[:, :, ::2] = cA.imag.view(B, 1, L)
    upD_r[:, :, ::2] = cD.real.view(B, 1, L)
    upD_i[:, :, ::2] = cD.imag.view(B, 1, L)

    pad_len = 4
    yA_r, yA_i = _complex_conv1d(_symmetric_pad_1d(upA_r, pad_len), _symmetric_pad_1d(upA_i, pad_len), h0_r, h0_i)
    yD_r, yD_i = _complex_conv1d(_symmetric_pad_1d(upD_r, pad_len), _symmetric_pad_1d(upD_i, pad_len), g0_r, g0_i)

    y_r = (yA_r + yD_r).squeeze(1)
    y_i = (yA_i + yD_i).squeeze(1)

    y_r = y_r[:, :output_len]
    y_i = y_i[:, :output_len]
    y = torch.complex(y_r, y_i)
    return y.squeeze(0) if squeeze_back else y


def _wavelet_dwt(h_complex: torch.Tensor, mode: str, alpha: torch.Tensor = None):
    if mode == 'cdb4_strict':
        return _dwt_complex_cdb4_strict(h_complex, alpha=alpha)
    return _dwt_complex_db4(h_complex, alpha=alpha)


def _wavelet_idwt(cA: torch.Tensor, cD: torch.Tensor, output_len: int, mode: str) -> torch.Tensor:
    if mode == 'cdb4_strict':
        return _idwt_complex_cdb4_strict(cA, cD, output_len=output_len)
    return _idwt_complex_db4(cA, cD, output_len=output_len)


def _dwt_idwt_cdb4_forward_complex(h_complex: torch.Tensor, lam: float, mode: str = 'cdb4') -> torch.Tensor:
    output_len = h_complex.shape[-1]
    cA, cD = _wavelet_dwt(h_complex, mode)
    lam_t = torch.as_tensor(lam, device=h_complex.device, dtype=h_complex.real.dtype)
    cD_thr = _complex_soft_threshold(cD, lam_t)
    return _wavelet_idwt(cA, cD_thr, output_len=output_len, mode=mode)


def compute_wavelet_losses(h_pred_complex: torch.Tensor,
                           h_gt_complex: torch.Tensor,
                           mode: str,
                           alpha: torch.Tensor,
                           eps: float = 1e-8) -> (torch.Tensor, torch.Tensor):
    """Paper-aligned wavelet terms: L_wave (cD mismatch) and lambda_H."""
    _, cD_pred = _wavelet_dwt(h_pred_complex, mode, alpha=alpha)
    _, cD_gt = _wavelet_dwt(h_gt_complex, mode, alpha=alpha)
    l_wave = torch.sum(torch.abs(cD_pred - cD_gt))
    lambda_h = l_wave / (torch.sum(torch.abs(cD_gt)) + eps)
    return l_wave, lambda_h


def _apply_cdb4_forward_branch(pred_re: torch.Tensor, pred_im: torch.Tensor, iteration: int, mode: str) -> (torch.Tensor, torch.Tensor):
    h = torch.complex(pred_re, pred_im)

    if iteration <= 5000:
        lam = 0.05
    elif iteration <= 25000:
        t = (iteration - 5000) / 20000.0
        lam = 0.05 + (0.02 - 0.05) * t
    else:
        lam = 0.005

    h_wave = _dwt_idwt_cdb4_forward_complex(h, lam=lam, mode=mode)
    return h_wave.real, h_wave.imag


def preprocess_uplink_pair(uplink_re: torch.Tensor, uplink_im: torch.Tensor,
                          wavelet_cfg):
    """Wavelet hook; disabled when wavelet_cfg is None."""
    if wavelet_cfg is None:
        return uplink_re, uplink_im

    if not wavelet_cfg.get('enabled', False):
        return uplink_re, uplink_im

    mode = wavelet_cfg.get('mode', 'none')
    if mode in {'cdb4', 'cdb4_strict'}:
        return uplink_re, uplink_im

    return uplink_re, uplink_im


# ---- viewpoint helper ----

def make_viewpoint(tx_pos, rx_pos, n_elevation=9, n_azimuth=36):
    R = torch.eye(3, dtype=torch.float32)
    dummy = torch.zeros(n_elevation, n_azimuth, dtype=torch.float32)
    return SpectrumInfo(R=R, T_rx=rx_pos, T_tx=tx_pos,
                        spectrum=dummy, spectrum_path="csi",
                        spectrum_name="00001", height=n_elevation, width=n_azimuth)


# ---- phase 1: autoencoder pretraining (uplink CSI -> TX position) ----

def pretrain_autoencoder(scene_info, args, model_path, wavelet_cfg=None):
    device = torch.device(args.data_device)
    pretrain_iters = getattr(args, 'pretrain_iters', 10000)

    encoder = CSIEncoder(n_antennas=scene_info.n_antennas,
                         n_subcarriers=scene_info.n_subcarriers).to(device)
    auto_decoder = CSIAutoDecoder(n_antennas=scene_info.n_antennas,
                                  n_subcarriers=scene_info.n_subcarriers).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(auto_decoder.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_iters, eta_min=1e-5)

    train_samples = scene_info.train_samples
    loss_log = []
    batch_size = 32

    all_up_re = torch.stack([s.uplink_re for s in train_samples]).to(device)
    all_up_im = torch.stack([s.uplink_im for s in train_samples]).to(device)
    n_train = all_up_re.shape[0]

    print(f"\n  Phase 1: Autoencoder pretraining ({pretrain_iters} iters)\n")

    progress_bar = tqdm(range(pretrain_iters), desc="  Pretrain")
    for iteration in range(1, pretrain_iters + 1):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        batch_re = all_up_re[idx]
        batch_im = all_up_im[idx]
        batch_re, batch_im = preprocess_uplink_pair(batch_re, batch_im, wavelet_cfg or {})

        positions = encoder(batch_re, batch_im)
        pred_re, pred_im = auto_decoder(positions)

        recon_loss = torch.nn.functional.mse_loss(pred_re, batch_re) \
                   + torch.nn.functional.mse_loss(pred_im, batch_im)

        dists = torch.cdist(positions, positions)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        spread_loss = torch.relu(0.1 - dists[mask].view(batch_size, -1).min(dim=1)[0]).mean()

        loss = recon_loss + 0.01 * spread_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"}); progress_bar.update(10)
        if iteration % 500 == 0:
            loss_log.append((iteration, loss.item()))

    progress_bar.close()

    encoder.eval()
    test_re = torch.stack([s.uplink_re for s in scene_info.test_samples]).to(device)
    test_im = torch.stack([s.uplink_im for s in scene_info.test_samples]).to(device)
    test_re, test_im = preprocess_uplink_pair(test_re, test_im, wavelet_cfg or {})
    with torch.no_grad():
        test_pos = encoder(test_re, test_im)
        test_pred_re, test_pred_im = auto_decoder(test_pos)
        test_loss = (torch.nn.functional.mse_loss(test_pred_re, test_re)
                    + torch.nn.functional.mse_loss(test_pred_im, test_im)).item()
    print(f"\n  Pretrain done: test_loss={test_loss:.6f}")

    fixed_path = os.path.join(os.path.dirname(model_path), "pretrained_encoder.pth")
    torch.save({'encoder': encoder.state_dict()}, fixed_path)
    print(f"  Saved: {fixed_path}")

    if loss_log:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(*zip(*loss_log)); ax.set_xlabel("Iter"); ax.set_ylabel("Loss")
        ax.set_title("Autoencoder Pretrain"); ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(model_path, "pretrain_loss.png"), dpi=200); plt.close()

    encoder.train()
    return encoder


# ---- per-antenna evaluation ----

def evaluate_one_antenna(ant_idx, gaussians, encoder, scene_info, pipe_args,
                         n_azimuth, n_elevation, ant_output_dir, iteration,
                         wavelet_cfg=None):
    device = next(encoder.parameters()).device
    rx_pos = scene_info.antenna_positions[ant_idx].to(device)
    test_samples = scene_info.test_samples
    all_snr = []
    all_nmse = []
    all_pred_re = []
    all_pred_im = []
    all_gt_re = []
    all_gt_im = []

    with torch.no_grad():
        for sample in test_samples:
            up_re = sample.uplink_re.to(device)
            up_im = sample.uplink_im.to(device)
            up_re, up_im = preprocess_uplink_pair(up_re, up_im, wavelet_cfg or {})
            tx_pos = encoder(up_re, up_im)
            viewpoint = make_viewpoint(tx_pos, rx_pos, n_elevation, n_azimuth)
            render_pkg = render_csi(viewpoint, gaussians, pipe_args,
                                   n_azimuth=n_azimuth, n_elevation=n_elevation)
            rendered_csi = render_pkg["render"].mean(dim=(1, 2))
            pred_re = rendered_csi[0::2]
            pred_im = rendered_csi[1::2]

            gt_re = sample.downlink_re[ant_idx].to(device)
            gt_im = sample.downlink_im[ant_idx].to(device)

            # denormalize for fair SNR comparison
            down_std = scene_info.down_std
            pred_re_dn = pred_re * down_std + scene_info.down_re_mean
            pred_im_dn = pred_im * down_std + scene_info.down_im_mean
            gt_re_dn = gt_re * down_std + scene_info.down_re_mean
            gt_im_dn = gt_im * down_std + scene_info.down_im_mean

            all_pred_re.append(pred_re_dn.cpu().numpy())
            all_pred_im.append(pred_im_dn.cpu().numpy())
            all_gt_re.append(gt_re_dn.cpu().numpy())
            all_gt_im.append(gt_im_dn.cpu().numpy())

            err = ((pred_re_dn - gt_re_dn)**2 + (pred_im_dn - gt_im_dn)**2).sum().item()
            gt_pwr = (gt_re_dn**2 + gt_im_dn**2).sum().item()
            all_snr.append(-10 * np.log10(err / (gt_pwr + 1e-8) + 1e-10))
            all_nmse.append(err / (gt_pwr + 1e-8))

    snr_arr = np.array(all_snr)
    nmse_arr = np.array(all_nmse)
    result = {
        "antenna": ant_idx, "iteration": iteration,
        "num_gaussians": gaussians.get_xyz.shape[0],
        "SNR_dB_mean": round(float(snr_arr.mean()), 2),
        "SNR_dB_std": round(float(snr_arr.std()), 2),
        "SNR_dB_min": round(float(snr_arr.min()), 2),
        "SNR_dB_p25": round(float(np.percentile(snr_arr, 25)), 2),
        "SNR_dB_p50": round(float(np.percentile(snr_arr, 50)), 2),
        "SNR_dB_p90": round(float(np.percentile(snr_arr, 90)), 2),
        "SNR_dB_p95": round(float(np.percentile(snr_arr, 95)), 2),
        "SNR_dB_max": round(float(snr_arr.max()), 2),
        "NMSE_mean": round(float(nmse_arr.mean()), 6),
        "NMSE_std": round(float(nmse_arr.std()), 6),
        "NMSE_min": round(float(nmse_arr.min()), 6),
        "NMSE_p25": round(float(np.percentile(nmse_arr, 25)), 6),
        "NMSE_p50": round(float(np.percentile(nmse_arr, 50)), 6),
        "NMSE_p90": round(float(np.percentile(nmse_arr, 90)), 6),
        "NMSE_p95": round(float(np.percentile(nmse_arr, 95)), 6),
        "NMSE_max": round(float(nmse_arr.max()), 6),
    }

    out_dir = os.path.join(ant_output_dir, f"eval_iter{iteration}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=2)

    np.savez(os.path.join(out_dir, "csi_results.npz"),
             pred=np.array(all_pred_re) + 1j * np.array(all_pred_im),
             gt=np.array(all_gt_re) + 1j * np.array(all_gt_im),
             snr_db=snr_arr,
             nmse=nmse_arr)

    print(f"    Ant{ant_idx} iter{iteration}: SNR={snr_arr.mean():.2f}±{snr_arr.std():.2f} dB, "
          f"NMSE={nmse_arr.mean():.6f}±{nmse_arr.std():.6f} (#G={gaussians.get_xyz.shape[0]})")
    return result


# ---- phase 2: per-antenna Gaussian splatting training ----

def train_one_antenna(ant_idx, encoder, scene_info, args, ant_output_dir, test_iterations,
                      ref_gaussians=None, wavelet_cfg=None):
    """
    Train one antenna model.
    - ref_gaussians=None (antenna 0): train all parameters from scratch
    - ref_gaussians provided (antenna 1+): reuse geometry, only train FLE coefficients
    """

    device = torch.device(args.data_device)
    freeze_geometry = ref_gaussians is not None

    # be explicit about cache cleanup before each antenna starts
    torch.cuda.empty_cache()

    args.num_channels_override = scene_info.n_subcarriers * 2
    gaussians = GaussianModel(args)

    if freeze_geometry:
        # reuse geometry from antenna 0, reinitialize FLE coefficients
        args.gene_init_point = False
        scene = Scene(args, gaussians, load_iteration=None, shuffle=True)
        init_gaussians_from_reference(gaussians, ref_gaussians, args)
        setup_fle_only_optimizer(gaussians, args)
        print(f"    (reusing geometry from antenna 0, training FLE only)")
    else:
        # train everything from scratch
        scene = Scene(args, gaussians, load_iteration=None, shuffle=True)
        gaussians.training_setup(args)

    rx_pos = scene_info.antenna_positions[ant_idx].to(device)
    n_azimuth = getattr(args, 'n_azimuth', 36)
    n_elevation = getattr(args, 'n_elevation', 9)
    wavelet_cfg = _build_wavelet_cfg(args)
    if wavelet_cfg is None:
        print("  Uplink preprocessing: disabled")
    else:
        print(f"  Uplink preprocessing: enabled={wavelet_cfg['enabled']}, mode={wavelet_cfg['mode']}")

    pipe_args = Namespace(**{k: getattr(args, k) for k in
                            ['convert_SHs_python', 'compute_cov3D_python', 'debug', 'radius_rx']
                            if hasattr(args, k)})

    train_samples = scene_info.train_samples
    iters = args.iterations

    wavelet_alpha_logit = None
    g_ref = None
    wavelet_alpha_opt = None
    if wavelet_cfg is not None and wavelet_cfg.get('enabled', False):
        wavelet_alpha_logit, g_ref = _build_wavelet_alpha(wavelet_cfg.get('mode', 'cdb4'), device, torch.float32)
        wavelet_alpha_opt = torch.optim.Adam([wavelet_alpha_logit], lr=getattr(args, 'wavelet_alpha_lr', 1e-4))

    progress_bar = tqdm(range(iters), desc=f"  Ant{ant_idx}", leave=False)

    # training loop
    for iteration in range(1, iters + 1):
        if not freeze_geometry:
            gaussians.update_learning_rate(iteration)

        # progressively increase FLE degree
        fle_ramp = getattr(args, '_fle_degree_ramp', 500)
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        sample = train_samples[randint(0, len(train_samples) - 1)]

        with torch.no_grad():
            up_re = sample.uplink_re.to(device)
            up_im = sample.uplink_im.to(device)
            up_re, up_im = preprocess_uplink_pair(up_re, up_im, wavelet_cfg or {})
            tx_pos = encoder(up_re, up_im)
        viewpoint = make_viewpoint(tx_pos, rx_pos, n_elevation, n_azimuth)

        render_pkg = render_csi(viewpoint, gaussians, pipe_args,
                               n_azimuth=n_azimuth, n_elevation=n_elevation)
        rendered = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        rendered_csi = rendered.mean(dim=(1, 2))
        pred_re = rendered_csi[0::2]
        pred_im = rendered_csi[1::2]
        h_pred = torch.complex(pred_re, pred_im)

        # wavelet branch is auxiliary: never overwrite the main CSI prediction used by SNR/MSE
        h_wavelet = None
        if wavelet_cfg is not None and wavelet_cfg.get('enabled', False) and wavelet_cfg.get('mode', 'none') in {'cdb4', 'cdb4_strict'}:
            h_wavelet = _dwt_idwt_cdb4_forward_complex(h_pred, lam=0.05, mode=wavelet_cfg.get('mode', 'cdb4'))

        l_hf = rendered_csi.new_tensor(0.0)
        l_pr = rendered_csi.new_tensor(0.0)
        lambda_h = rendered_csi.new_tensor(0.0)

        gt_re = sample.downlink_re[ant_idx].to(device)
        gt_im = sample.downlink_im[ant_idx].to(device)
        h_gt = torch.complex(gt_re, gt_im)

        alpha = None
        if wavelet_alpha_logit is not None:
            alpha = torch.sigmoid(wavelet_alpha_logit)

        if wavelet_cfg is not None and wavelet_cfg.get('enabled', False) and alpha is not None:
            l_hf, lambda_h = compute_wavelet_losses(
                h_pred, h_gt, mode=wavelet_cfg.get('mode', 'cdb4'), alpha=alpha)

            if wavelet_cfg.get('use_lpr', False):
                # PR-style proxy: encourage bounded alpha toward full high-pass recovery (alpha -> 1)
                l_pr = (alpha - 1.0) ** 2

        expected_channels = int(scene_info.n_subcarriers) * 2
        if rendered_csi.numel() != expected_channels:
            raise RuntimeError(
                f"Rendered channel count mismatch: got {rendered_csi.numel()}, "
                f"expected {expected_channels} (=2*{scene_info.n_subcarriers}). "
                f"This usually means tracer output channels are fixed and do not match dataset subcarriers."
            )
        if pred_re.shape != gt_re.shape or pred_im.shape != gt_im.shape:
            raise RuntimeError(
                f"Prediction/GT shape mismatch at antenna {ant_idx}: "
                f"pred_re={tuple(pred_re.shape)} vs gt_re={tuple(gt_re.shape)}, "
                f"pred_im={tuple(pred_im.shape)} vs gt_im={tuple(gt_im.shape)}, "
                f"rendered_csi={tuple(rendered_csi.shape)}, n_subcarriers={scene_info.n_subcarriers}."
            )

        mse_loss = torch.nn.functional.mse_loss(pred_re, gt_re) \
                 + torch.nn.functional.mse_loss(pred_im, gt_im)

        if wavelet_cfg is not None and wavelet_cfg.get('enabled', False):
            lpr_on = wavelet_cfg.get('use_lpr', False) and iteration >= wavelet_cfg.get('lpr_warmup', 0)
            lhf_on = wavelet_cfg.get('use_lhf', False) and iteration >= wavelet_cfg.get('lhf_warmup', 0)

            lpr_weight = wavelet_cfg.get('lpr_weight', 0.05) if lpr_on else 0.0
            lhf_weight = wavelet_cfg.get('lhf_weight', 0.1) if lhf_on else 0.0

            loss = mse_loss + lpr_weight * l_pr + lhf_weight * (lambda_h * l_hf)
        else:
            loss = mse_loss

        if wavelet_alpha_opt is not None:
            wavelet_alpha_opt.zero_grad(set_to_none=True)

        loss.backward()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"}); progress_bar.update(10)
            if iteration == iters:
                progress_bar.close()

            if iteration in test_iterations:
                scene.save(iteration)
                torch.save({
                    'gaussians': gaussians.capture(),
                    'iteration': iteration,
                }, os.path.join(ant_output_dir, f"chkpnt{iteration}.pth"))

                evaluate_one_antenna(ant_idx, gaussians, encoder, scene_info,
                                     pipe_args, n_azimuth, n_elevation,
                                     ant_output_dir, iteration,
                                     wavelet_cfg=wavelet_cfg)

            # densification only for antenna 0 (full training)
            if not freeze_geometry:
                densify_until = getattr(args, 'densify_until_iter', iters // 2)
                if iteration < densify_until:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(gaussians.get_xyz, visibility_filter)

                    densify_from = getattr(args, 'densify_from_iter', 500)
                    densify_interval = getattr(args, 'densification_interval', 100)
                    if iteration >= densify_from and iteration % densify_interval == 0:
                        size_threshold = getattr(args, 'raddi_size_threshold', 10) \
                            if iteration > getattr(args, 'opacity_reset_interval', 3000) else None

                        grad_threshold = getattr(args, 'densify_grad_threshold', 0.0002)
                        min_att_threshold = getattr(args, 'min_attenuation_threshold', 0.004)

                        gaussians.densify_and_prune(
                            grad_threshold,
                            min_att_threshold,
                            scene.cameras_extent, size_threshold)

                    if iteration % getattr(args, 'opacity_reset_interval', 3000) == 0:
                        gaussians.reset_attenuation()

            if iteration < iters:
                gaussians.optimizer.step()
                if wavelet_alpha_opt is not None:
                    wavelet_alpha_opt.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    return gaussians, pipe_args


# ---- main: train all antennas ----

if __name__ == '__main__':

    # parse config file first, then build full argument parser
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/csi/exp1.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="CSI Training (per-antenna models)")
    parser.add_argument("--config", type=str, default="arguments/configs/csi/exp1.yaml")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--pretrain_iters", type=int, default=10000)
    parser.add_argument("--pretrained_encoder", type=str, default=None)
    parser.add_argument("--max_gpu_mem_gb", type=float, default=0.0,
                        help="Per-process PyTorch memory fraction cap in GB; 0 disables cap")
    parser.add_argument("--wavelet_enabled", action='store_true',
                        default=(yaml_cfg or {}).get("wavelet_enabled", False),
                        help="Enable wavelet auxiliary losses")
    parser.add_argument("--wavelet_alpha_lr", type=float,
                        default=(yaml_cfg or {}).get("wavelet_alpha_lr", 1e-4),
                        help="Learning rate for bounded wavelet alpha parameter")
    parser.add_argument("--wavelet_use_lpr", action='store_true',
                        default=(yaml_cfg or {}).get("wavelet_use_lpr", False),
                        help="Enable PR-style wavelet regularization")
    parser.add_argument("--wavelet_use_lhf", action='store_true',
                        default=(yaml_cfg or {}).get("wavelet_use_lhf", False),
                        help="Enable high-frequency subband consistency loss")
    parser.add_argument("--wavelet_lpr_weight", type=float,
                        default=(yaml_cfg or {}).get("wavelet_lpr_weight", 0.05),
                        help="Weight for L_pr term")
    parser.add_argument("--wavelet_lhf_weight", type=float,
                        default=(yaml_cfg or {}).get("wavelet_lhf_weight", 0.1),
                        help="Weight for L_hf term")
    parser.add_argument("--wavelet_lpr_warmup_iters", type=int,
                        default=(yaml_cfg or {}).get("wavelet_lpr_warmup_iters", 2000),
                        help="Warmup iterations before enabling L_pr")
    parser.add_argument("--wavelet_lhf_warmup_iters", type=int,
                        default=(yaml_cfg or {}).get("wavelet_lhf_warmup_iters", 2000),
                        help="Warmup iterations before enabling L_hf")
    parser.add_argument("--wavelet_mode", type=str,
                        default=(yaml_cfg or {}).get("wavelet_mode", "none"),
                        choices=["none", "cdb4", "cdb4_strict"],
                        help="Wavelet mode")

    cmd_args = parser.parse_args()

    # set up data and output paths
    dataset_name = cmd_args.dataset
    exp_name = cmd_args.exp_name
    data_dir = os.path.join(cmd_args.input_data_folder, dataset_name)
    cmd_args.source_path = data_dir

    # logs/<dataset>/<exp_name>[_YYYYmmdd-HHMMSS]/
    model_path = os.path.join(cmd_args.log_base_folder, dataset_name, exp_name)
    if os.path.exists(model_path):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(cmd_args.log_base_folder, dataset_name, f"{exp_name}_{ts}")
    os.makedirs(model_path, exist_ok=False)

    cmd_args.model_path = model_path

    cmd_args.densify_until_iter = cmd_args.iterations // 2
    cmd_args.position_lr_max_steps = cmd_args.iterations

    safe_state(cmd_args.quiet, random_seed, torch.device(cmd_args.data_device))

    if cmd_args.max_gpu_mem_gb and cmd_args.max_gpu_mem_gb > 0:
        device = torch.device(cmd_args.data_device)
        if device.type == "cuda":
            total_bytes = torch.cuda.get_device_properties(device).total_memory
            fraction = min(max((cmd_args.max_gpu_mem_gb * (1024 ** 3)) / total_bytes, 0.05), 0.98)
            torch.cuda.set_per_process_memory_fraction(fraction, device=device)
            print(f"  Set CUDA memory fraction cap: {fraction:.3f} (~{cmd_args.max_gpu_mem_gb} GB)")

    # load CSI data
    split_method = getattr(cmd_args, 'split_method', 'random')
    ratio_train = getattr(cmd_args, 'ratio_train', 0.7)
    scene_info = load_csi_data(data_dir, split_method=split_method,
                               ratio_train=ratio_train, seed=random_seed)

    print(f"\n{'='*60}")
    print(f"  CSI Training (per-antenna models)")
    print(f"  Data: {data_dir}")
    print(f"  Output: {model_path}")
    print(f"  Samples: train={len(scene_info.train_samples)}, test={len(scene_info.test_samples)}")
    print(f"  Antennas: {scene_info.n_antennas} (one model each)")
    print(f"  Phase 1: {cmd_args.pretrain_iters} iters")
    print(f"  Phase 2: {cmd_args.iterations} iters × {scene_info.n_antennas} antennas")
    print(f"{'='*60}")

    with open(os.path.join(model_path, "config.json"), 'w') as f:
        config_dict = {k: v for k, v in vars(cmd_args).items() if not k.startswith('_')}
        json.dump(config_dict, f, indent=2, default=str)

    args = model_para_cls.extract(cmd_args)
    for k in ['wavelet_enabled', 'wavelet_mode', 'wavelet_use_lpr', 'wavelet_use_lhf',
              'wavelet_lpr_weight', 'wavelet_lhf_weight',
              'wavelet_lpr_warmup_iters', 'wavelet_lhf_warmup_iters',
              'wavelet_alpha_lr']:
        setattr(args, k, getattr(cmd_args, k))
    for k, v in vars(optimization_para_cls.extract(cmd_args)).items():
        setattr(args, k, v)
    for k, v in vars(pipeline_para_cls.extract(cmd_args)).items():
        setattr(args, k, v)
    for k in ['model_path', 'densify_until_iter', 'position_lr_max_steps',
              'data_device', 'pretrain_iters']:
        if hasattr(cmd_args, k):
            setattr(args, k, getattr(cmd_args, k))

    n_azimuth = getattr(args, 'n_azimuth', 36)
    n_elevation = getattr(args, 'n_elevation', 9)
    wavelet_cfg = _build_wavelet_cfg(args)

    # load or pretrain encoder
    device = torch.device(cmd_args.data_device)
    encoder_path = cmd_args.pretrained_encoder
    if encoder_path is None:
        default_path = os.path.join(cmd_args.log_base_folder, dataset_name, "pretrained_encoder.pth")
        if os.path.exists(default_path):
            encoder_path = default_path

    if encoder_path and os.path.exists(encoder_path):
        print(f"\n  Loading pretrained encoder: {encoder_path}")
        encoder = CSIEncoder(n_antennas=scene_info.n_antennas,
                             n_subcarriers=scene_info.n_subcarriers).to(device)
        ckpt = torch.load(encoder_path)
        encoder.load_state_dict(ckpt['encoder'])
        print(f"  Loaded. Skipping Phase 1.\n")
    else:
        encoder = pretrain_autoencoder(scene_info, args, model_path, wavelet_cfg=wavelet_cfg)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    iters = args.iterations
    test_iters_cfg = getattr(args, '_test_iterations', None)
    if test_iters_cfg:
        test_iters = sorted(set(test_iters_cfg + [iters]))
    else:
        test_step = max(iters // 5, 5)
        test_iters = sorted(set([i for i in range(test_step, iters + 1, test_step)] + [iters]))

    # train one model per antenna in fixed order: 0..n-1
    # antenna 0: full training (geometry + FLE)
    # antenna 1+: reuse antenna 0 geometry, train FLE only
    ref_geometry_cpu = None

    for ant_idx in range(scene_info.n_antennas):
        ant_dir = os.path.join(model_path, f"antenna_{ant_idx}")
        os.makedirs(ant_dir, exist_ok=True)

        args.model_path = ant_dir

        mode = "full" if ant_idx == 0 else "FLE-only"
        print(f"\n  Training antenna {ant_idx} [{mode}] (RX={scene_info.antenna_positions[ant_idx].tolist()})")

        gaussians, _ = train_one_antenna(
            ant_idx, encoder, scene_info, args, ant_dir, test_iters,
            ref_gaussians=ref_geometry_cpu,
            wavelet_cfg=wavelet_cfg
        )

        # save antenna 0 geometry as reference for subsequent antennas
        if ant_idx == 0:
            ref_geometry_cpu = capture_geometry_to_cpu_dict(gaussians)

        del gaussians
        torch.cuda.empty_cache()

    # aggregate results across all antennas for this run
    print(f"\n  Collecting summaries...")
    for test_iter in test_iters:
        ant_results = []
        all_found = True
        for ant_idx in range(scene_info.n_antennas):
            result_path = os.path.join(model_path, f"antenna_{ant_idx}",
                                       f"eval_iter{test_iter}", "result.json")
            if not os.path.exists(result_path):
                all_found = False
                break
            with open(result_path) as f:
                ant_results.append(json.load(f))

        if not all_found:
            continue

        all_snr = []
        per_ant_means = []
        for ant_idx in range(scene_info.n_antennas):
            eval_dir = os.path.join(model_path, f"antenna_{ant_idx}", f"eval_iter{test_iter}")
            data = np.load(os.path.join(eval_dir, "csi_results.npz"))
            pred_re, pred_im = data['pred'].real, data['pred'].imag
            gt_re, gt_im = data['gt'].real, data['gt'].imag

            if ant_idx == 0:
                n_test = pred_re.shape[0]
                total_err = np.zeros(n_test)
                total_gt = np.zeros(n_test)

            err = ((pred_re - gt_re)**2 + (pred_im - gt_im)**2).sum(axis=1)
            gt_pwr = (gt_re**2 + gt_im**2).sum(axis=1)
            total_err += err
            total_gt += gt_pwr
            per_ant_means.append(next(r["SNR_dB_mean"] for r in ant_results if r["antenna"] == ant_idx))

            all_snr = -10 * np.log10(total_err / (total_gt + 1e-8) + 1e-10)
            all_nmse = total_err / (total_gt + 1e-8)

            summary = {
                "iteration": test_iter,
                "num_test": len(all_snr),
                "SNR_dB": {
                    "mean": round(float(all_snr.mean()), 2),
                    "std": round(float(all_snr.std()), 2),
                    "min": round(float(all_snr.min()), 2),
                    "p25": round(float(np.percentile(all_snr, 25)), 2),
                    "p50": round(float(np.percentile(all_snr, 50)), 2),
                    "p90": round(float(np.percentile(all_snr, 90)), 2),
                    "p95": round(float(np.percentile(all_snr, 95)), 2),
                    "max": round(float(all_snr.max()), 2),
                },
                "NMSE": {
                    "mean": round(float(all_nmse.mean()), 6),
                    "std": round(float(all_nmse.std()), 6),
                    "min": round(float(all_nmse.min()), 6),
                    "p25": round(float(np.percentile(all_nmse, 25)), 6),
                    "p50": round(float(np.percentile(all_nmse, 50)), 6),
                    "p90": round(float(np.percentile(all_nmse, 90)), 6),
                    "p95": round(float(np.percentile(all_nmse, 95)), 6),
                    "max": round(float(all_nmse.max()), 6),
                },
                "per_antenna_SNR_dB": {f"ant{i}": m for i, m in enumerate(per_ant_means)},
            }

            out_path = os.path.join(model_path, f"summary_iter{test_iter}.json")
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"  iter{test_iter}: SNR={all_snr.mean():.2f}±{all_snr.std():.2f} dB, "
                  f"NMSE={all_nmse.mean():.6f}±{all_nmse.std():.6f}  "
                  f"per-ant={['%.1f' % m for m in per_ant_means]}")


    print(f"\n  CSI Training complete. Results in: {model_path}\n")
