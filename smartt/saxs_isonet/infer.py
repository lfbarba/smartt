"""Inference script for the SAXS-TT missing-wedge correction pipeline.

Called as a subprocess by pipeline.py each round:

    python -m smartt.saxs_isonet.infer \\
        --volume_dir    /path/to/round_N/ \\
        --y_dirs_path   /path/to/y_dirs.npy \\
        --checkpoint    /path/to/checkpoint.pt \\
        --alpha_deg     45.0 \\
        --half_space    y \\
        --output_dir    /path/to/round_N+1/

For each of the K volumes:
  1. Run the model patch-by-patch (overlapping patches, averaged at boundaries).
  2. Enforce Fourier consistency: replace measured frequencies of the model output
     with those from the *input* volume (cascaded enforcement, not the original FBP).
  3. Save the corrected volume as a .npy file in output_dir.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from smartt.saxs_isonet.train import build_model
from smartt.saxs_isonet.wedge import (
    goniometer_axis_for_half_space,
    missing_wedge_mask_3d,
    sinusoidal_wedge_embedding,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch aggregation helpers
# ---------------------------------------------------------------------------

def _extract_patches_with_positions(
    vol: torch.Tensor,
    patch_size: int,
    stride: int,
) -> tuple[list[torch.Tensor], list[tuple]]:
    """Extract overlapping cubic patches and their top-left corner positions.

    Parameters
    ----------
    vol        : (X, Y, Z) float32
    patch_size : P
    stride     : step between patch origins (< P gives overlap)

    Returns
    -------
    patches   : list of (1, 1, P, P, P) tensors
    positions : list of (x0, y0, z0) int tuples
    """
    X, Y, Z = vol.shape
    P, S = patch_size, stride
    patches, positions = [], []

    xs = list(range(0, max(1, X - P + 1), S))
    ys = list(range(0, max(1, Y - P + 1), S))
    zs = list(range(0, max(1, Z - P + 1), S))

    # Ensure the last patch reaches the volume boundary
    if xs[-1] + P < X:
        xs.append(X - P)
    if ys[-1] + P < Y:
        ys.append(Y - P)
    if zs[-1] + P < Z:
        zs.append(Z - P)

    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                p = vol[x0:x0 + P, y0:y0 + P, z0:z0 + P]
                if p.shape != (P, P, P):
                    padded = torch.zeros(P, P, P, dtype=vol.dtype)
                    padded[:p.shape[0], :p.shape[1], :p.shape[2]] = p
                    p = padded
                patches.append(p.unsqueeze(0).unsqueeze(0))   # (1, 1, P, P, P)
                positions.append((x0, y0, z0))

    return patches, positions


def _aggregate_patches(
    patch_preds: list[torch.Tensor],
    positions: list[tuple],
    vol_shape: tuple,
    patch_size: int,
) -> torch.Tensor:
    """Average overlapping patch predictions into a full volume.

    Parameters
    ----------
    patch_preds : list of (P, P, P) tensors
    positions   : matching list of (x0, y0, z0) tuples
    vol_shape   : (X, Y, Z) output shape
    patch_size  : P

    Returns
    -------
    aggregated : (X, Y, Z) float32 tensor
    """
    X, Y, Z = vol_shape
    P = patch_size
    acc   = torch.zeros(X, Y, Z, dtype=torch.float32)
    count = torch.zeros(X, Y, Z, dtype=torch.float32)

    for pred, (x0, y0, z0) in zip(patch_preds, positions):
        x1 = min(x0 + P, X)
        y1 = min(y0 + P, Y)
        z1 = min(z0 + P, Z)
        acc  [x0:x1, y0:y1, z0:z1] += pred[:x1-x0, :y1-y0, :z1-z0]
        count[x0:x1, y0:y1, z0:z1] += 1.0

    return acc / count.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Fourier consistency enforcement
# ---------------------------------------------------------------------------

def enforce_fourier_consistency(
    model_output: torch.Tensor,
    measured_input: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Replace the measured frequencies of model_output with those from measured_input.

    Parameters
    ----------
    model_output   : (X, Y, Z) float32 — model-predicted complete volume.
    measured_input : (X, Y, Z) float32 — volume used as input this round
                     (previous round's output, or original FBP for round 0).
    mask           : (X, Y, Z) bool — True = measured frequency.

    Returns
    -------
    enforced : (X, Y, Z) float32 — model output with known Fourier data restored.
    """
    fft_pred    = torch.fft.fftn(model_output)
    fft_measured = torch.fft.fftn(measured_input)
    fft_pred[mask] = fft_measured[mask]
    return torch.fft.ifftn(fft_pred).real


# ---------------------------------------------------------------------------
# Per-volume inference
# ---------------------------------------------------------------------------

def infer_volume(
    vol: torch.Tensor,
    y_dir: np.ndarray,
    model: torch.nn.Module,
    alpha_deg: float,
    goniometer_axis: np.ndarray,
    patch_size: int,
    stride: int,
    conditioning_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Run model inference on a single volume and enforce Fourier consistency.

    Parameters
    ----------
    vol            : (X, Y, Z) float32 — normalized input volume.
    y_dir          : (3,) unit vector — q-direction for this sub-CT.
    model          : trained UNet3DConditionModel.
    alpha_deg      : polar-cap half-angle (degrees).
    goniometer_axis: tilt-axis unit vector.
    patch_size     : P — must match what was used during training.
    stride         : step between overlapping patches (P//2 gives 50% overlap).
    conditioning_dim : cross_attention_dim.
    device         : compute device.

    Returns
    -------
    corrected : (X, Y, Z) float32 — wedge-corrected volume with Fourier consistency.
    """
    vol_shape = tuple(vol.shape)

    # Conditioning embedding for this volume's wedge
    from smartt.saxs_isonet.wedge import missing_arc_length
    arc = missing_arc_length(y_dir, alpha_deg, goniometer_axis)
    cond = sinusoidal_wedge_embedding(arc, dim=conditioning_dim, device=device)
    cond = cond.expand(1, -1, -1)  # (1, 1, dim)

    timesteps = torch.zeros(1, device=device, dtype=torch.long)

    # Extract patches
    patches, positions = _extract_patches_with_positions(vol.cpu(), patch_size, stride)
    logger.debug("Volume %s → %d patches", vol_shape, len(patches))

    # Run model patch-by-patch
    model.eval()
    patch_preds = []
    with torch.no_grad():
        for patch in patches:
            patch = patch.to(device)  # (1, 1, P, P, P)
            pred = model(
                patch,
                timestep=timesteps,
                encoder_hidden_states=cond,
                return_dict=False,
            )[0]  # (1, 1, P, P, P)
            patch_preds.append(pred.squeeze().cpu())  # (P, P, P)

    # Aggregate overlapping predictions
    pred_vol = _aggregate_patches(patch_preds, positions, vol_shape, patch_size)
    # pred_vol: (X, Y, Z)

    # Fourier consistency: restore measured frequencies from the input volume
    mask_vol = torch.from_numpy(
        missing_wedge_mask_3d(y_dir, alpha_deg, vol_shape, goniometer_axis)
    )  # (X, Y, Z) bool
    corrected = enforce_fourier_consistency(pred_vol, vol.cpu(), mask_vol)

    return corrected  # (X, Y, Z)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def infer(
    volume_dir: str,
    y_dirs: np.ndarray,
    checkpoint: str,
    alpha_deg: float,
    half_space: str,
    output_dir: str,
    patch_size: int = 64,
    stride: int | None = None,
    conditioning_dim: int = 128,
    device: torch.device | None = None,
) -> list[Path]:
    """Run inference on all K volumes and save corrected .npy files.

    Parameters
    ----------
    volume_dir  : directory with K .npy input volumes.
    y_dirs      : (K, 3) q-directions.
    checkpoint  : path to trained model checkpoint (.pt).
    alpha_deg   : polar-cap half-angle (degrees).
    half_space  : 'z' or 'y'.
    output_dir  : where to save K corrected .npy files.
    patch_size  : must match the patch_size used during training.
    stride      : patch stride (defaults to patch_size // 2 = 50% overlap).
    conditioning_dim : cross_attention_dim.
    device      : compute device (defaults to CUDA if available).

    Returns
    -------
    out_paths : list of K Path objects pointing to the corrected volumes.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if stride is None:
        stride = patch_size // 2

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    goniometer_axis = goniometer_axis_for_half_space(half_space)

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model(cross_attention_dim=conditioning_dim).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint)

    # ── Load input volumes ────────────────────────────────────────────────
    paths = sorted(Path(volume_dir).glob('*.npy'))
    K = len(y_dirs)
    if len(paths) != K:
        raise ValueError(f"Found {len(paths)} .npy files, expected K={K}")

    out_paths = []
    for k, (p, y_dir) in enumerate(zip(paths, y_dirs)):
        vol_np = np.load(p).astype(np.float32)
        # Per-volume normalisation (match training convention)
        mean, std = vol_np.mean(), vol_np.std() + 1e-8
        vol_norm = (vol_np - mean) / std
        vol = torch.from_numpy(vol_norm)

        logger.info(
            "Inferring volume k=%d/%d  y=(%.2f,%.2f,%.2f)  shape=%s",
            k, K, y_dir[0], y_dir[1], y_dir[2], tuple(vol.shape),
        )

        corrected = infer_volume(
            vol=vol,
            y_dir=y_dir,
            model=model,
            alpha_deg=alpha_deg,
            goniometer_axis=goniometer_axis,
            patch_size=patch_size,
            stride=stride,
            conditioning_dim=conditioning_dim,
            device=device,
        )

        # Denormalise back to original scale
        corrected_np = (corrected.numpy() * std + mean).astype(np.float32)

        out_p = out_dir / f'vol_{k:04d}.npy'
        np.save(out_p, corrected_np)
        out_paths.append(out_p)
        logger.info("  → saved %s", out_p)

    return out_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SAXS-TT missing-wedge inference')
    p.add_argument('--volume_dir',       required=True)
    p.add_argument('--y_dirs_path',      required=True)
    p.add_argument('--checkpoint',       required=True)
    p.add_argument('--alpha_deg',        type=float, required=True)
    p.add_argument('--half_space',       default='y', choices=['y', 'z'])
    p.add_argument('--output_dir',       required=True)
    p.add_argument('--patch_size',       type=int, default=64)
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
    y_dirs = np.load(args.y_dirs_path)
    infer(
        volume_dir=args.volume_dir,
        y_dirs=y_dirs,
        checkpoint=args.checkpoint,
        alpha_deg=args.alpha_deg,
        half_space=args.half_space,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        conditioning_dim=args.conditioning_dim,
    )
