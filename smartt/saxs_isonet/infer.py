"""Inference for the SAXS-TT missing-wedge correction pipeline.

Called as a subprocess by pipeline.py each round:

    python -m smartt.saxs_isonet.infer \\
        --volume_dir      /path/to/round_N_volumes/ \\
        --rsm_dirs_path   /path/to/rsm_dirs.npy \\
        --norm_stats_path /path/to/norm_stats.npy \\
        --checkpoint      /path/to/checkpoint.pt \\
        --alpha_deg       45.0 \\
        --half_space      y \\
        --output_dir      /path/to/round_N1_volumes/

Each RSM volume's wedge points along ``rsm_dirs[k]`` in an arbitrary orientation.
Since the model was trained on **canonical** wedges, we work in the canonical
frame by rotating the *whole volume* (not per-patch — that would reintroduce
corner artifacts):

  1. normalize by the fixed round_00 stats, zero-pad to fit the rotation
  2. rotate the whole volume into the canonical frame
  3. run axis-aligned cube_size patches through the model, aggregate
  4. rotate the filled volume back, crop to the original shape, denormalize
  5. enforce Fourier consistency with the input (measured freqs from data,
     missing-wedge freqs from the model — cascaded across rounds)

Volumes are spherically masked (exterior = 0) before entering the pipeline, so
the model always sees near-zero corners after rotation. The sphere mask is
re-applied after step 5 to keep outputs consistent with training.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch

from smartt.saxs_isonet.augment import carve_shifted, rotate_batch
from smartt.saxs_isonet.preprocess import make_sphere_mask
from smartt.saxs_isonet.train import build_model
from smartt.saxs_isonet.wedge import (
    canonical_goniometer_axis,
    canonical_rotation,
    goniometer_axis_for_half_space,
    missing_arc_length,
    missing_wedge_mask_3d,
    sinusoidal_wedge_embedding,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch extraction / aggregation (axis-aligned, in the canonical frame)
# ---------------------------------------------------------------------------

def _extract_patches(vol: torch.Tensor, cube: int, stride: int):
    """Extract overlapping cube_size patches + top-left positions from (X,Y,Z)."""
    X, Y, Z = vol.shape
    patches, positions = [], []
    axes = []
    for dim in (X, Y, Z):
        starts = list(range(0, max(1, dim - cube + 1), stride))
        if starts[-1] + cube < dim:
            starts.append(dim - cube)
        axes.append(starts)
    for x0 in axes[0]:
        for y0 in axes[1]:
            for z0 in axes[2]:
                p = vol[x0:x0 + cube, y0:y0 + cube, z0:z0 + cube]
                if p.shape != (cube, cube, cube):
                    pad = torch.zeros(cube, cube, cube, dtype=vol.dtype)
                    pad[:p.shape[0], :p.shape[1], :p.shape[2]] = p
                    p = pad
                patches.append(p)
                positions.append((x0, y0, z0))
    return patches, positions


def _aggregate(preds, positions, vol_shape, cube):
    """Average overlapping cube predictions into a full (X,Y,Z) volume."""
    X, Y, Z = vol_shape
    acc = torch.zeros(X, Y, Z, dtype=torch.float32)
    cnt = torch.zeros(X, Y, Z, dtype=torch.float32)
    for pred, (x0, y0, z0) in zip(preds, positions):
        x1, y1, z1 = min(x0 + cube, X), min(y0 + cube, Y), min(z0 + cube, Z)
        acc[x0:x1, y0:y1, z0:z1] += pred[:x1 - x0, :y1 - y0, :z1 - z0]
        cnt[x0:x1, y0:y1, z0:z1] += 1.0
    return acc / cnt.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Fourier consistency enforcement
# ---------------------------------------------------------------------------

def enforce_fourier_consistency(
    model_output: torch.Tensor,
    measured_input: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Replace measured frequencies of model_output with those of measured_input.

    mask : (X, Y, Z) bool — True = measured frequency (unshifted layout).
    """
    fft_pred = torch.fft.fftn(model_output)
    fft_meas = torch.fft.fftn(measured_input)
    fft_pred[mask] = fft_meas[mask]
    return torch.fft.ifftn(fft_pred).real


# ---------------------------------------------------------------------------
# Per-volume inference (full-volume canonical rotation)
# ---------------------------------------------------------------------------

def _pad_to_fit_rotation(vol_shape: tuple) -> list[tuple[int, int]]:
    """Symmetric (before, after) padding per axis so an arbitrary rotation about
    the centre does not clip the data (pad each axis to the volume diagonal)."""
    diag = int(math.ceil(math.sqrt(sum(d * d for d in vol_shape))))
    diag += diag % 2                          # make even
    pads = []
    for d in vol_shape:
        total = max(0, diag - d)
        before = total // 2
        pads.append((before, total - before))
    return pads


def infer_volume(
    vol_norm: torch.Tensor,
    rsm_dir: np.ndarray,
    model: torch.nn.Module,
    alpha_deg: float,
    goniometer_axis: np.ndarray,
    cube_size: int,
    stride: int,
    conditioning_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Canonical-frame inference on a single normalized volume → filled (normalized).

    Returns the model-filled volume in the original frame (same shape as input,
    normalized).  Fourier consistency is NOT applied here (done by the caller in
    raw space).
    """
    orig_shape = tuple(vol_norm.shape)
    pads = _pad_to_fit_rotation(orig_shape)
    vol_p = torch.nn.functional.pad(
        vol_norm,
        # F.pad takes last-dim-first: (z_b,z_a, y_b,y_a, x_b,x_a)
        (pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]),
        mode='constant', value=0.0,
    ).to(device)

    R_can = torch.from_numpy(canonical_rotation(rsm_dir, goniometer_axis)).float().to(device)
    R_can = R_can.unsqueeze(0)                       # (1,3,3)
    R_inv = R_can.transpose(1, 2)

    # Rotate the whole padded volume into the canonical frame.
    vol_can = rotate_batch(vol_p.unsqueeze(0), R_can, mode='bilinear')[0]  # (D,D,D)

    # Carve canonical wedge exactly after rotation — matches training order (rotate then carve).
    _can_keep = torch.from_numpy(
        np.fft.fftshift(missing_wedge_mask_3d(
            np.array([0., 1., 0.]), alpha_deg, tuple(vol_can.shape),
            canonical_goniometer_axis(rsm_dir, goniometer_axis),
        ))
    ).to(device)
    vol_can = carve_shifted(vol_can.unsqueeze(0), _can_keep.unsqueeze(0)).squeeze(0)

    arc = missing_arc_length(rsm_dir, alpha_deg, goniometer_axis)
    cond = sinusoidal_wedge_embedding(arc, dim=conditioning_dim, device=device)  # (1,1,dim)
    timesteps = torch.zeros(1, device=device, dtype=torch.long)

    patches, positions = _extract_patches(vol_can.cpu(), cube_size, stride)
    model.eval()
    preds = []
    with torch.no_grad():
        for p in patches:
            x = p.to(device)[None, None]             # (1,1,C,C,C)
            y = model(x, timestep=timesteps, encoder_hidden_states=cond,
                      return_dict=False)[0]
            preds.append(y.squeeze().cpu())
    filled_can = _aggregate(preds, positions, tuple(vol_can.shape), cube_size).to(device)

    # Rotate back and crop to the original shape.
    filled_p = rotate_batch(filled_can.unsqueeze(0), R_inv, mode='bilinear')[0].cpu()
    sl = tuple(slice(b, b + d) for (b, _), d in zip(pads, orig_shape))
    return filled_p[sl]                              # (X,Y,Z) normalized


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def infer(
    volume_dir: str,
    rsm_dirs: np.ndarray,
    checkpoint: str,
    alpha_deg: float,
    half_space: str,
    output_dir: str,
    cube_size: int = 32,
    stride: int | None = None,
    conditioning_dim: int = 128,
    norm_stats: np.ndarray | None = None,
    device: torch.device | None = None,
) -> list[Path]:
    """Run inference on all K volumes and save corrected .npy files."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if stride is None:
        stride = cube_size // 2

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    goniometer_axis = goniometer_axis_for_half_space(half_space)

    model = build_model(cross_attention_dim=conditioning_dim).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint)

    paths = sorted(Path(volume_dir).glob('vol_*.npy'))
    K = len(rsm_dirs)
    if len(paths) != K:
        raise ValueError(f"Found {len(paths)} vol_*.npy files, expected K={K}")

    out_paths = []
    for k, (p, rsm_dir) in enumerate(zip(paths, rsm_dirs)):
        vol_raw = np.load(p).astype(np.float32)
        if norm_stats is not None:
            mean, std = float(norm_stats[k][0]), float(norm_stats[k][1])
        else:                                        # fallback: per-volume own stats
            mean, std = float(vol_raw.mean()), float(vol_raw.std() + 1e-8)
        vol_norm_arr = (vol_raw - mean) / std
        vol_norm_arr[~make_sphere_mask(vol_raw.shape[0])] = 0.0   # match training: exterior = 0 in normalized space
        vol_norm = torch.from_numpy(vol_norm_arr)

        logger.info("Inferring k=%d/%d rsm=(%.2f,%.2f,%.2f) shape=%s",
                    k, K, rsm_dir[0], rsm_dir[1], rsm_dir[2], tuple(vol_norm.shape))

        filled_norm = infer_volume(
            vol_norm=vol_norm, rsm_dir=rsm_dir, model=model, alpha_deg=alpha_deg,
            goniometer_axis=goniometer_axis, cube_size=cube_size, stride=stride,
            conditioning_dim=conditioning_dim, device=device,
        )

        # Denormalize, then enforce measured frequencies from the input (raw space).
        filled_raw = filled_norm * std + mean
        mask_vol = torch.from_numpy(
            missing_wedge_mask_3d(rsm_dir, alpha_deg, tuple(vol_raw.shape), goniometer_axis)
        )
        corrected = enforce_fourier_consistency(
            filled_raw, torch.from_numpy(vol_raw), mask_vol)

        # Re-apply sphere mask: Fourier operations spread energy to the exterior,
        # so we zero it back out to keep volumes consistent with training.
        sphere_mask = torch.from_numpy(make_sphere_mask(cube_size))
        corrected = corrected * sphere_mask

        out_p = out_dir / f'vol_{k:04d}.npy'
        np.save(out_p, corrected.numpy().astype(np.float32))
        out_paths.append(out_p)
        logger.info("  → saved %s", out_p)

    return out_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SAXS-TT missing-wedge inference')
    p.add_argument('--volume_dir',       required=True)
    p.add_argument('--rsm_dirs_path',    required=True)
    p.add_argument('--checkpoint',       required=True)
    p.add_argument('--alpha_deg',        type=float, required=True)
    p.add_argument('--half_space',       default='y', choices=['y', 'z'])
    p.add_argument('--output_dir',       required=True)
    p.add_argument('--norm_stats_path',  default=None)
    p.add_argument('--cube_size',        type=int, default=32)
    p.add_argument('--stride',           type=int, default=None)
    p.add_argument('--conditioning_dim', type=int, default=128)
    p.add_argument('--log_level',        default='INFO')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(message)s',
    )
    rsm_dirs = np.load(args.rsm_dirs_path)
    norm_stats = np.load(args.norm_stats_path) if args.norm_stats_path else None
    infer(
        volume_dir=args.volume_dir,
        rsm_dirs=rsm_dirs,
        checkpoint=args.checkpoint,
        alpha_deg=args.alpha_deg,
        half_space=args.half_space,
        output_dir=args.output_dir,
        cube_size=args.cube_size,
        stride=args.stride,
        conditioning_dim=args.conditioning_dim,
        norm_stats=norm_stats,
    )
