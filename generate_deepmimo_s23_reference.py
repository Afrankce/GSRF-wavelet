"""Generate the minimal CSI dataset used by this repository from a DeepMIMO v4 scenario.

The current CSI training code only needs the following files under data/csi-like
folder:

    csidata.npy          complex array with shape [num_samples, 8, 52]
                         first 26 subcarriers: uplink, last 26: downlink
    base-station.yml     YAML with key "base_station" containing 8 antenna xyzs
    train_index.txt      integer train sample indices
    test_index.txt       integer test sample indices

Optional points3D.ply is intentionally not generated here because the CSI config
currently has gene_init_point: true, so training regenerates it automatically.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_SCENARIO = "i1_2p4"


def _check_gpu_memory_or_exit(limit_gib: float) -> None:
    """Exit early when used GPU memory exceeds limit_gib."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        # If nvidia-smi is unavailable, skip this guard silently.
        return

    used_mib_values = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            used_mib_values.append(float(line))
        except ValueError:
            continue

    if not used_mib_values:
        return

    max_used_gib = max(used_mib_values) / 1024.0
    if max_used_gib > float(limit_gib):
        raise SystemExit(
            f"GPU memory guard triggered: used={max_used_gib:.2f} GiB exceeds limit={float(limit_gib):.2f} GiB. Exiting."
        )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _slice_users(arr: np.ndarray, user_first: int, user_last: int) -> np.ndarray:
    """DeepMIMO examples use 1-based inclusive user ids."""
    if arr.dtype == object:
        raise RuntimeError(
            "User slicing received an object array (likely unresolved per-set DeepMIMO field). "
            "Please ensure field flattening produced a numeric ndarray."
        )
    start = max(user_first - 1, 0)
    stop = user_last
    return arr[start:stop]


def _load_deepmimo_dataset(scenario: str, download: bool) -> Any:
    try:
        import deepmimo as dm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "DeepMIMO is not installed in this Python environment. "
            "Install it in the same environment used to run this script."
        ) from exc

    if download:
        dm.download(scenario)

    dataset = dm.load(scenario)
    if hasattr(dataset, "info"):
        dataset.info()
    return dataset


def _compute_channels(dataset: Any, args: argparse.Namespace) -> None:
    if args.skip_channel_computation:
        return
    if not hasattr(dataset, "compute_channels"):
        raise RuntimeError("Loaded DeepMIMO dataset has no compute_channels() method.")

    # DeepMIMO v4 official channel generation API:
    #   ch_params = dm.ChannelParameters()
    #   ... set OFDM/antenna fields ...
    #   dataset.compute_channels(ch_params)
    try:
        import deepmimo as dm  # type: ignore
    except ImportError as exc:
        raise ImportError("DeepMIMO is required to compute channels.") from exc

    ch_params = dm.ChannelParameters()
    ch_params.freq_domain = True

    # OFDM settings
    ch_params.ofdm.subcarriers = int(args.num_deepmimo_subcarriers)
    ch_params.ofdm.bandwidth = float(args.num_deepmimo_subcarriers) * float(args.subcarrier_spacing)

    # keep all subcarriers by default
    if hasattr(ch_params.ofdm, "selected_subcarriers"):
        ch_params.ofdm.selected_subcarriers = np.arange(int(args.num_deepmimo_subcarriers), dtype=int)

    # Antenna settings
    ch_params.bs_antenna.shape = np.array(args.bs_antenna_shape, dtype=int)
    ch_params.ue_antenna.shape = np.array(args.ue_antenna_shape, dtype=int)

    if hasattr(ch_params.bs_antenna, "spacing"):
        ch_params.bs_antenna.spacing = float(args.antenna_spacing)
    if hasattr(ch_params.ue_antenna, "spacing"):
        ch_params.ue_antenna.spacing = 0.5

    if hasattr(ch_params, "enable_dual_polarization"):
        ch_params.enable_dual_polarization = bool(args.enable_dual_polarization)

    dataset.compute_channels(ch_params)


def _coerce_dataset_array(value: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except Exception:
        pass

    # DeepMIMO v4 may expose fields as list/tuple of per-set arrays with
    # inhomogeneous lengths. Flatten sets along sample axis without changing
    # downstream logic.
    if isinstance(value, (list, tuple)):
        parts = []
        for idx, part in enumerate(value):
            arr = np.asarray(part)
            if arr.size == 0:
                continue
            parts.append(arr)
        if not parts:
            raise RuntimeError(f"DeepMIMO field '{name}' contains no usable entries.")

        # Concatenate along sample axis (axis=0) if ranks match.
        rank_set = {p.ndim for p in parts}
        if len(rank_set) == 1:
            try:
                return np.concatenate(parts, axis=0)
            except Exception:
                pass

        # Fallback: keep object array with explicit warning upstream.
        return np.array(parts, dtype=object)

    return np.asarray(value)


def _get_required_array(dataset: Any, name: str) -> np.ndarray:
    if not hasattr(dataset, name):
        raise RuntimeError(f"DeepMIMO dataset is missing required attribute: {name}")
    return _coerce_dataset_array(getattr(dataset, name), name)


def _get_optional_array(dataset: Any, candidates: list[str]) -> tuple[str, np.ndarray] | tuple[None, None]:
    for name in candidates:
        if hasattr(dataset, name):
            return name, _coerce_dataset_array(getattr(dataset, name), name)
    return None, None


def _array_summary(arr: np.ndarray, name: str) -> dict[str, Any]:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    finite_ratio = float(finite.mean()) if arr.size else 1.0
    nonzero_ratio = float((arr != 0).mean()) if arr.size else 0.0
    stats: dict[str, Any] = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_ratio": finite_ratio,
        "nonzero_ratio": nonzero_ratio,
        "size": int(arr.size),
    }
    if arr.size:
        if np.iscomplexobj(arr):
            mag = np.abs(arr)
            stats.update({
                "abs_min": float(np.nanmin(mag)),
                "abs_p50": float(np.nanpercentile(mag, 50)),
                "abs_p95": float(np.nanpercentile(mag, 95)),
                "abs_max": float(np.nanmax(mag)),
            })
        else:
            stats.update({
                "min": float(np.nanmin(arr)),
                "p50": float(np.nanpercentile(arr, 50)),
                "p95": float(np.nanpercentile(arr, 95)),
                "max": float(np.nanmax(arr)),
            })
    return stats


def _find_sample_axis(shape: tuple[int, ...], expected_samples: int) -> int:
    exact = [i for i, size in enumerate(shape) if size == expected_samples]
    if exact:
        return exact[0]
    larger = [i for i, size in enumerate(shape) if size >= expected_samples]
    if larger:
        return larger[0]
    raise RuntimeError(
        f"Cannot infer sample axis from channel shape {shape}; no axis has at least {expected_samples} samples."
    )


def _extract_csi_from_channel(
    channel: np.ndarray,
    sample_count: int,
    user_first: int,
    user_last: int,
    num_antennas: int,
    total_subcarriers: int,
    require_native_ofdm: bool,
) -> np.ndarray | None:
    """Convert a DeepMIMO channel tensor to [N, num_antennas, total_subcarriers].

    Returns None if the computed channel does not contain an OFDM subcarrier
    dimension. Some DeepMIMO v4 builds generate narrowband channel tensors such
    as [num_rx, 1, num_bs_ant, 1] even when num_subcarriers is provided; in that
    case this script falls back to synthesizing OFDM CSI from path parameters.
    """
    channel = np.asarray(channel)
    if not np.iscomplexobj(channel):
        channel = channel.astype(np.complex64)

    sample_axis = _find_sample_axis(channel.shape, sample_count)
    channel = np.moveaxis(channel, sample_axis, 0)
    channel = _slice_users(channel, user_first, user_last) if channel.shape[0] >= user_last else channel[:sample_count]

    if channel.shape[0] != sample_count:
        raise RuntimeError(
            f"Expected {sample_count} selected channel samples, got {channel.shape[0]} from shape {channel.shape}."
        )

    candidate_sub_axes = [i for i, size in enumerate(channel.shape[1:], start=1) if size >= total_subcarriers]
    if not candidate_sub_axes:
        msg = (
            f"Computed DeepMIMO channel shape {channel.shape} has no subcarrier axis with "
            f"{total_subcarriers} entries."
        )
        if require_native_ofdm:
            raise RuntimeError(msg + " Native OFDM is required; aborting without fallback.")
        print(msg + " Falling back to path-parameter synthesis.")
        return None
    subcarrier_axis = max(candidate_sub_axes, key=lambda i: channel.shape[i])

    channel = np.moveaxis(channel, subcarrier_axis, -1)
    channel = channel[..., :total_subcarriers]
    channel = channel.reshape(sample_count, -1, total_subcarriers)

    if channel.shape[1] < num_antennas:
        raise RuntimeError(
            f"Only {channel.shape[1]} antenna/polarization channels were found, but {num_antennas} are required. "
            "Increase the DeepMIMO BS antenna shape or disable settings that collapse antenna dimensions."
        )

    return channel[:, :num_antennas, :].astype(np.complex64)


def _prepare_path_array(arr: np.ndarray, sample_count: int, user_first: int, user_last: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    sample_axis = _find_sample_axis(arr.shape, sample_count)
    arr = np.moveaxis(arr, sample_axis, 0)
    arr = _slice_users(arr, user_first, user_last) if arr.shape[0] >= user_last else arr[:sample_count]
    if arr.shape[0] != sample_count:
        raise RuntimeError(f"{name} sample count mismatch: expected {sample_count}, got {arr.shape[0]}")
    return arr


def _to_sample_path(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def _synthesize_csi_from_paths(
    dataset: Any,
    sample_count: int,
    user_first: int,
    user_last: int,
    num_antennas: int,
    total_subcarriers: int,
    subcarrier_spacing: float,
    tx_power_dbm: float,
    power_unit: str,
    bs_antenna_shape: list[int],
    antenna_spacing: float,
    carrier_frequency: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesize frequency-domain CSI from path parameters with array response.

    Strict fallback model:
      H_m[f] = sum_p sqrt(P_p) * exp(j*phi_p) * exp(-j*2*pi*f*tau_p) * a_m(theta_p, phi_p)

    where a_m is built from DeepMIMO path angles and BS array geometry.
    """
    power = _prepare_path_array(_get_required_array(dataset, "power"), sample_count, user_first, user_last, "power")
    phase = _prepare_path_array(_get_required_array(dataset, "phase"), sample_count, user_first, user_last, "phase")
    delay = _prepare_path_array(_get_required_array(dataset, "delay"), sample_count, user_first, user_last, "delay")

    power = _to_sample_path(power).astype(np.float64)
    phase = _to_sample_path(phase).astype(np.float64)
    delay = _to_sample_path(delay).astype(np.float64)

    az_name, doa_az = _get_optional_array(dataset, ["doa_az", "aoa_az", "DoA_az", "AoA_az"])
    el_name, doa_el = _get_optional_array(dataset, ["doa_el", "aoa_el", "DoA_el", "AoA_el"])
    if doa_az is None or doa_el is None:
        raise RuntimeError(
            "Strict fallback requires path angle arrays (doa_az/aoa_az and doa_el/aoa_el) "
            "to build physically consistent BS array response."
        )

    doa_az = _to_sample_path(_prepare_path_array(doa_az, sample_count, user_first, user_last, az_name)).astype(np.float64)
    doa_el = _to_sample_path(_prepare_path_array(doa_el, sample_count, user_first, user_last, el_name)).astype(np.float64)

    n_paths = min(power.shape[1], phase.shape[1], delay.shape[1], doa_az.shape[1], doa_el.shape[1])
    power = power[:, :n_paths]
    phase = phase[:, :n_paths]
    delay = delay[:, :n_paths]
    doa_az = doa_az[:, :n_paths]
    doa_el = doa_el[:, :n_paths]

    if power_unit == "auto":
        finite_power = power[np.isfinite(power)]
        detected = "linear"
        if finite_power.size > 0:
            p95 = float(np.percentile(finite_power, 95))
            p50 = float(np.percentile(finite_power, 50))
            if p95 < 1.0 and p50 < 0.0:
                detected = "dBW"
        power_unit_effective = detected
    else:
        power_unit_effective = power_unit

    if power_unit_effective == "dBW":
        power_dbm = power + 30.0
        tx_gain_db = float(tx_power_dbm) - 30.0
        power_dbm = power_dbm + tx_gain_db
        power_watt = np.power(10.0, (power_dbm - 30.0) / 10.0)
    elif power_unit_effective == "dBm":
        power_dbm = power
        power_watt = np.power(10.0, (power_dbm - 30.0) / 10.0)
    elif power_unit_effective == "watt":
        power_watt = power
    else:
        raise ValueError(f"Unsupported --power-unit: {power_unit_effective}")

    valid = np.isfinite(power_watt) & np.isfinite(phase) & np.isfinite(delay) & np.isfinite(doa_az) & np.isfinite(doa_el) & (power_watt > 0)

    valid_paths_per_sample = valid.sum(axis=1)
    invalid_path_ratio = 1.0 - float(valid.mean())

    power_watt = np.where(valid, power_watt, 0.0)
    phase = np.where(valid, phase, 0.0)
    delay = np.where(valid, delay, 0.0)
    doa_az = np.where(valid, doa_az, 0.0)
    doa_el = np.where(valid, doa_el, 0.0)

    amp = np.sqrt(np.maximum(power_watt, 0.0))
    path_gain = amp * np.exp(1j * phase)

    freqs = (np.arange(total_subcarriers, dtype=np.float64) - (total_subcarriers - 1) / 2.0) * float(subcarrier_spacing)
    response = np.exp(-1j * 2.0 * np.pi * delay[:, :, None] * freqs[None, None, :])

    rows, cols = int(bs_antenna_shape[0]), int(bs_antenna_shape[1])
    if rows * cols < num_antennas:
        raise RuntimeError(f"bs_antenna_shape={bs_antenna_shape} provides only {rows*cols} elements; need {num_antennas}.")

    wavelength = 3.0e8 / float(carrier_frequency)
    d = float(antenna_spacing) * wavelength
    elem = []
    for r in range(rows):
        for c in range(cols):
            y = (c - (cols - 1) / 2.0) * d
            z = (r - (rows - 1) / 2.0) * d
            elem.append([0.0, y, z])
    elem_pos = np.asarray(elem[:num_antennas], dtype=np.float64)

    az = np.deg2rad(doa_az)
    el = np.deg2rad(doa_el)
    kx = np.cos(el) * np.cos(az)
    ky = np.cos(el) * np.sin(az)
    kz = np.sin(el)
    kvec = np.stack([kx, ky, kz], axis=-1)

    phase_shift = 2.0 * np.pi / wavelength * np.einsum("npa,ma->npm", kvec, elem_pos)
    array_resp = np.exp(1j * phase_shift)

    path_array_gain = path_gain[:, :, None] * array_resp
    csidata = np.sum(path_array_gain[:, :, :, None] * response[:, :, None, :], axis=1)

    re = np.nan_to_num(csidata.real, nan=0.0, posinf=0.0, neginf=0.0)
    im = np.nan_to_num(csidata.imag, nan=0.0, posinf=0.0, neginf=0.0)
    csidata = (re + 1j * im).astype(np.complex64)

    finite_mask = np.isfinite(csidata.real) & np.isfinite(csidata.imag)
    stats = {
        "power_unit_effective": power_unit_effective,
        "tx_power_dbm": float(tx_power_dbm),
        "invalid_path_ratio": invalid_path_ratio,
        "valid_paths_per_sample_min": int(valid_paths_per_sample.min()) if valid_paths_per_sample.size else 0,
        "valid_paths_per_sample_p50": float(np.percentile(valid_paths_per_sample, 50)) if valid_paths_per_sample.size else 0.0,
        "valid_paths_per_sample_p95": float(np.percentile(valid_paths_per_sample, 95)) if valid_paths_per_sample.size else 0.0,
        "valid_paths_per_sample_max": int(valid_paths_per_sample.max()) if valid_paths_per_sample.size else 0,
        "finite_ratio_after_synthesis": float(finite_mask.mean()),
        "array_response_angles": {"azimuth_field": az_name, "elevation_field": el_name},
    }

    return csidata, stats


def _make_base_station_positions(
    bs_center: np.ndarray,
    num_antennas: int,
    bs_antenna_shape: list[int],
    spacing_wavelengths: float,
    carrier_frequency: float,
) -> np.ndarray:
    """Create per-antenna xyz positions around the BS center.

    The CSI loader expects 8 receiver antenna positions. DeepMIMO commonly gives
    a single BS position, so we expand it into a small planar array. The spacing
    is converted from wavelengths to meters.
    """
    center = np.asarray(bs_center, dtype=np.float32).reshape(-1)[:3]
    rows, cols = int(bs_antenna_shape[0]), int(bs_antenna_shape[1])
    wavelength = 3.0e8 / float(carrier_frequency)
    spacing_m = float(spacing_wavelengths) * wavelength

    coords = []
    for r in range(rows):
        for c in range(cols):
            y = (c - (cols - 1) / 2.0) * spacing_m
            z = (r - (rows - 1) / 2.0) * spacing_m
            coords.append(center + np.array([0.0, y, z], dtype=np.float32))

    if len(coords) < num_antennas:
        raise RuntimeError(f"bs_antenna_shape={bs_antenna_shape} creates only {len(coords)} antennas; need {num_antennas}.")
    return np.stack(coords[:num_antennas], axis=0).astype(np.float32)


def _write_split_indices(output_dir: Path, sample_count: int, train_ratio: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count, dtype=np.int64)
    rng.shuffle(indices)
    train_count = int(sample_count * train_ratio)
    train_indices = np.sort(indices[:train_count])
    test_indices = np.sort(indices[train_count:])
    np.savetxt(output_dir / "train_index.txt", train_indices, fmt="%d")
    np.savetxt(output_dir / "test_index.txt", test_indices, fmt="%d")


def _concat_or_init(dst: np.ndarray | None, src: np.ndarray) -> np.ndarray:
    if dst is None:
        return src
    return np.concatenate([dst, src], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate current GSRF CSI training files from DeepMIMO.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="DeepMIMO scenario name passed to dm.load().")
    parser.add_argument("--scenario-dir", default="deepmimo_scenarios/asu_campus_3p5", help="Kept for provenance only; dm.load uses DeepMIMO's configured scenario path.")
    parser.add_argument("--output-dir", default="data/csi", help="Output directory containing csidata.npy and base-station.yml.")
    parser.add_argument("--user-first", type=int, default=1, help="First DeepMIMO user id, 1-based inclusive.")
    parser.add_argument("--user-last", type=int, default=1000, help="Last DeepMIMO user id, 1-based inclusive.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=8371)
    parser.add_argument("--chunk-size", type=int, default=0, help="If >0, process users in chunks to reduce peak memory (e.g., 500 or 1000).")
    parser.add_argument("--download", action="store_true", help="Call dm.download(scenario) before loading.")
    parser.add_argument("--skip-channel-computation", action="store_true", help="Use an already-computed dataset.channel if available.")
    parser.add_argument("--num-antennas", type=int, default=8, help="CSI loader expects 8 antennas by default.")
    parser.add_argument("--num-uplink-subcarriers", type=int, default=26)
    parser.add_argument("--num-downlink-subcarriers", type=int, default=26)
    parser.add_argument("--num-deepmimo-subcarriers", type=int, default=52, help="Number of OFDM subcarriers requested from DeepMIMO.")
    parser.add_argument("--subcarrier-spacing", type=float, default=15e3)
    parser.add_argument("--tx-power-dbm", type=float, default=20.0, help="Target transmit power in dBm for path-power based synthesis.")
    parser.add_argument("--power-unit", type=str, default="dBm", choices=["auto", "dBW", "dBm", "watt"],
                        help="Unit of DeepMIMO path power in fallback synthesis. Default is dBm.")
    parser.add_argument("--normalize-mode", type=str, default="nerf2_global", choices=["none", "nerf2_global"],
                        help="Optional CSI normalization. Default is 'nerf2_global' (divide by global max abs like NeRF2CSI preprocessing).")
    parser.add_argument("--bs-antenna-shape", nargs=2, type=int, default=[8, 8], help="Planar BS array shape. Default creates exactly 8 antennas.")
    parser.add_argument("--ue-antenna-shape", nargs=2, type=int, default=[1, 1], help="UE planar array shape. Default is 1x1.")
    parser.add_argument("--antenna-spacing", type=float, default=0.5, help="BS/UE antenna spacing in wavelengths.")
    parser.add_argument("--carrier-frequency", type=float, default=2.4e9)
    parser.add_argument("--enable-dual-polarization", action="store_true")
    parser.add_argument("--require-native-ofdm", dest="require_native_ofdm", action="store_true",
                        help="Require DeepMIMO native OFDM channel dimension; abort if channel has no subcarrier axis.")
    parser.add_argument("--no-require-native-ofdm", dest="require_native_ofdm", action="store_false",
                        help="Disable strict native-OFDM requirement.")
    parser.set_defaults(require_native_ofdm=True)
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Allow fallback synthesis from power/phase/delay when native OFDM channel is unavailable.")
    parser.add_argument("--gpu-mem-limit-gb", type=float, default=20.0,
                        help="Exit if current GPU memory usage exceeds this threshold in GiB (default: 20).")
    args = parser.parse_args()

    if args.user_last < args.user_first:
        raise ValueError("--user-last must be >= --user-first")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")

    if args.allow_fallback:
        args.require_native_ofdm = False

    _check_gpu_memory_or_exit(args.gpu_mem_limit_gb)

    repo_root = Path(__file__).resolve().parent
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_deepmimo_dataset(args.scenario, args.download)
    _compute_channels(dataset, args)

    rx_pos = _get_required_array(dataset, "rx_pos")
    tx_pos = _get_required_array(dataset, "tx_pos")
    channel = _get_required_array(dataset, "channel")

    selected_rx_pos = _slice_users(np.asarray(rx_pos), args.user_first, args.user_last)
    sample_count = selected_rx_pos.shape[0]
    if sample_count == 0:
        raise RuntimeError("Selected user range produced zero samples.")

    total_subcarriers = args.num_uplink_subcarriers + args.num_downlink_subcarriers
    if args.num_deepmimo_subcarriers < total_subcarriers:
        raise ValueError("--num-deepmimo-subcarriers must be at least uplink + downlink subcarriers.")

    csidata = _extract_csi_from_channel(
        channel=channel,
        sample_count=sample_count,
        user_first=args.user_first,
        user_last=args.user_last,
        num_antennas=args.num_antennas,
        total_subcarriers=total_subcarriers,
        require_native_ofdm=args.require_native_ofdm,
    )
    csi_source = "computed_channel"
    synthesis_stats = {}
    path_input_stats = {}
    if csidata is None:
        power_raw = _prepare_path_array(_get_required_array(dataset, "power"), sample_count, args.user_first, args.user_last, "power")
        phase_raw = _prepare_path_array(_get_required_array(dataset, "phase"), sample_count, args.user_first, args.user_last, "phase")
        delay_raw = _prepare_path_array(_get_required_array(dataset, "delay"), sample_count, args.user_first, args.user_last, "delay")

        path_input_stats = {
            "power": _array_summary(power_raw, "power"),
            "phase": _array_summary(phase_raw, "phase"),
            "delay": _array_summary(delay_raw, "delay"),
        }
        print("Path input stats:")
        print(f"  power finite={path_input_stats['power']['finite_ratio']:.4f}, nonzero={path_input_stats['power']['nonzero_ratio']:.4f}, p95={path_input_stats['power'].get('p95', 0.0):.6e}")
        print(f"  phase finite={path_input_stats['phase']['finite_ratio']:.4f}, nonzero={path_input_stats['phase']['nonzero_ratio']:.4f}")
        print(f"  delay finite={path_input_stats['delay']['finite_ratio']:.4f}, nonzero={path_input_stats['delay']['nonzero_ratio']:.4f}, p95={path_input_stats['delay'].get('p95', 0.0):.6e}")

        csidata, synthesis_stats = _synthesize_csi_from_paths(
            dataset=dataset,
            sample_count=sample_count,
            user_first=args.user_first,
            user_last=args.user_last,
            num_antennas=args.num_antennas,
            total_subcarriers=total_subcarriers,
            subcarrier_spacing=args.subcarrier_spacing,
            tx_power_dbm=args.tx_power_dbm,
            power_unit=args.power_unit,
            bs_antenna_shape=args.bs_antenna_shape,
            antenna_spacing=args.antenna_spacing,
            carrier_frequency=args.carrier_frequency,
        )
        csi_source = "path_parameter_synthesis"

    # final output sanitize and finite check regardless of source
    csidata = (np.nan_to_num(csidata.real, nan=0.0, posinf=0.0, neginf=0.0)
               + 1j * np.nan_to_num(csidata.imag, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.complex64)

    normalization = {"mode": args.normalize_mode, "global_max_abs": 1.0}
    if args.normalize_mode == "nerf2_global":
        global_max_abs = float(np.max(np.abs(csidata)))
        if global_max_abs > 0:
            csidata = (csidata / global_max_abs).astype(np.complex64)
            normalization["global_max_abs"] = global_max_abs
        print(f"Applied nerf2_global normalization with global_max_abs={normalization['global_max_abs']:.6e}")

    finite_mask = np.isfinite(csidata.real) & np.isfinite(csidata.imag)
    finite_ratio = float(finite_mask.mean())
    bad_samples = int((finite_mask.reshape(csidata.shape[0], -1).mean(axis=1) < 1.0).sum())
    print(f"CSI finite ratio: {finite_ratio:.6f} (bad samples: {bad_samples}/{csidata.shape[0]})")

    np.save(output_dir / "csidata.npy", csidata)

    tx_pos = np.asarray(tx_pos, dtype=np.float32)
    if tx_pos.ndim == 1:
        bs_center = tx_pos[:3]
    else:
        bs_center = tx_pos.reshape(-1, tx_pos.shape[-1])[0, :3]
    antenna_positions = _make_base_station_positions(
        bs_center=bs_center,
        num_antennas=args.num_antennas,
        bs_antenna_shape=args.bs_antenna_shape,
        spacing_wavelengths=args.antenna_spacing,
        carrier_frequency=args.carrier_frequency,
    )

    with (output_dir / "base-station.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"base_station": antenna_positions.tolist()}, f, sort_keys=False)

    _write_split_indices(output_dir, sample_count, args.train_ratio, args.seed)

    metadata = {
        "scenario": args.scenario,
        "scenario_dir": args.scenario_dir,
        "user_first": args.user_first,
        "user_last": args.user_last,
        "sample_count": int(sample_count),
        "csidata_shape": list(csidata.shape),
        "csidata_dtype": str(csidata.dtype),
        "csi_source": csi_source,
        "format": f"[sample, antenna, subcarrier], first {args.num_uplink_subcarriers} uplink and last {args.num_downlink_subcarriers} downlink",
        "rx_positions_shape": list(np.asarray(selected_rx_pos).shape),
        "bs_center": np.asarray(bs_center).astype(float).tolist(),
        "base_station_shape": list(antenna_positions.shape),
        "finite_ratio": finite_ratio,
        "bad_samples": bad_samples,
        "path_input_stats": path_input_stats,
        "path_synthesis_stats": synthesis_stats,
        "normalization": normalization,
        "args": vars(args),
    }
    with (output_dir / "deepmimo_csi_generation.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, indent=2)

    print(f"Generated CSI dataset: {output_dir}")
    print(f"  csidata.npy: shape={csidata.shape}, dtype={csidata.dtype}")
    print(f"  base-station.yml: {antenna_positions.shape[0]} antenna positions")
    print("  train_index.txt / test_index.txt written with integer sample indices")


if __name__ == "__main__":
    main()
