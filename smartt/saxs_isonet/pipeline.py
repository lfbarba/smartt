"""Top-level iterative pipeline for SAXS-TT missing-wedge correction.

Usage (Python API)
------------------
    from smartt.saxs_isonet.pipeline import run_pipeline

    run_pipeline(
        reconstruction=recon_tensor,   # (K, X, Y, Z) from saxs_fbp/gd_reconstruction
        y_dirs=y_directions,           # (K, 3) from the same call
        alpha_deg=45.0,
        half_space='y',
        output_dir='/path/to/results/',
        num_rounds=3,
        epochs_per_round=100,
    )

Usage (CLI)
-----------
    python -m smartt.saxs_isonet.pipeline \\
        --reconstruction_path /path/to/recon.pt \\
        --y_dirs_path         /path/to/y_dirs.npy \\
        --alpha_deg           45.0 \\
        --half_space          y \\
        --output_dir          /path/to/results/ \\
        --num_rounds          3 \\
        --epochs_per_round    100

Directory layout produced
--------------------------
    output_dir/
      rsm_dirs.npy             — saved once, shared across all rounds
      round_00/                — original FBP volumes (K .npy files)
      round_01/                — checkpoint.pt after round 1 training
      round_01_volumes/        — round 1 inference output (K .npy files)
      round_02/                — checkpoint.pt (fine-tuned from round_01)
      round_02_volumes/        — round 2 inference output
      ...
      final/                   — fully-trained model applied to round_00 (output)

Each round:
  1. Train on volumes in round_N/ (subprocess → smartt.saxs_isonet.train).
  2. Infer on those same volumes → corrected volumes saved to round_N+1/ (subprocess → smartt.saxs_isonet.infer).
  3. Enforce Fourier consistency cascades through the subprocess automatically.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from smartt.saxs_isonet.dataset import compute_norm_stats, save_reconstruction_volumes
from smartt.saxs_isonet.preprocess import make_sphere_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subprocess launchers
# ---------------------------------------------------------------------------

def _run_training(
    volume_dir: Path,
    target_dir: Path,
    rsm_dirs_path: Path,
    norm_stats_path: Path,
    alpha_deg: float,
    half_space: str,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    cube_size: int,
    n_samples: int,
    conditioning_dim: int,
    min_wedge_deg: float,
    max_rsm_wedge_deg: Optional[float],
    resume: Optional[Path],
    num_workers: int,
) -> Path:
    """Launch training as a subprocess; return path to saved checkpoint."""
    cmd = [
        sys.executable, '-m', 'smartt.saxs_isonet.train',
        '--volume_dir',       str(volume_dir),
        '--target_dir',       str(target_dir),
        '--rsm_dirs_path',    str(rsm_dirs_path),
        '--norm_stats_path',  str(norm_stats_path),
        '--alpha_deg',        str(alpha_deg),
        '--half_space',       half_space,
        '--output_dir',       str(checkpoint_dir),
        '--epochs',           str(epochs),
        '--batch_size',       str(batch_size),
        '--lr',               str(lr),
        '--weight_decay',     str(weight_decay),
        '--cube_size',        str(cube_size),
        '--n_samples',        str(n_samples),
        '--conditioning_dim', str(conditioning_dim),
        '--min_wedge_deg',    str(min_wedge_deg),
        '--num_workers',      str(num_workers),
        '--log_level',        'INFO',
    ]
    if max_rsm_wedge_deg is not None:
        cmd += ['--max_rsm_wedge_deg', str(max_rsm_wedge_deg)]
    if resume is not None:
        cmd += ['--resume', str(resume)]

    logger.info("Launching training subprocess:\n  %s", ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return checkpoint_dir / 'checkpoint.pt'


def _run_inference(
    volume_dir: Path,
    rsm_dirs_path: Path,
    norm_stats_path: Path,
    checkpoint: Path,
    alpha_deg: float,
    half_space: str,
    output_dir: Path,
    cube_size: int,
    stride: Optional[int],
    conditioning_dim: int,
) -> Path:
    """Launch inference as a subprocess; return output directory."""
    cmd = [
        sys.executable, '-m', 'smartt.saxs_isonet.infer',
        '--volume_dir',       str(volume_dir),
        '--rsm_dirs_path',    str(rsm_dirs_path),
        '--norm_stats_path',  str(norm_stats_path),
        '--checkpoint',       str(checkpoint),
        '--alpha_deg',        str(alpha_deg),
        '--half_space',       half_space,
        '--output_dir',       str(output_dir),
        '--cube_size',        str(cube_size),
        '--conditioning_dim', str(conditioning_dim),
        '--log_level',        'INFO',
    ]
    if stride is not None:
        cmd += ['--stride', str(stride)]

    logger.info("Launching inference subprocess:\n  %s", ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return output_dir


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    reconstruction: Optional[torch.Tensor],
    rsm_dirs: np.ndarray,
    alpha_deg: float,
    half_space: str,
    output_dir: str | Path,
    num_rounds: int = 3,
    epochs_per_round: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    cube_size: int = 32,
    n_samples: int = 2000,
    conditioning_dim: int = 128,
    min_wedge_deg: float = 10.0,
    max_rsm_wedge_deg: Optional[float] = None,
    stride: Optional[int] = None,
    num_workers: int = 0,
    start_volume_dir: Optional[str | Path] = None,
) -> Path:
    """Run the iterative SAXS-TT missing-wedge correction pipeline.

    Parameters
    ----------
    reconstruction  : (K, X, Y, Z) float32 tensor — initial FBP/GD reconstruction.
        Pass ``None`` when ``start_volume_dir`` is set (reconstruction already on disk).
    rsm_dirs        : (K, 3) float64 ndarray — RSM directions from fibonacci_hemisphere.
    alpha_deg       : half-angle (°) of the goniometer's unmeasurable polar caps.
    half_space      : 'z' or 'y' — must match the half_space used during reconstruction.
    output_dir      : root directory for all pipeline outputs.
    num_rounds      : number of train→infer iterations.
    epochs_per_round: training epochs per round (fine-tuned from previous checkpoint).
    batch_size      : DataLoader batch size during training.
    lr              : AdamW learning rate.
    weight_decay    : AdamW weight decay.
    cube_size       : model input side-length (load size = ceil(√3·cube_size)).
    n_samples       : virtual dataset length per epoch.
    conditioning_dim: sinusoidal embedding / cross_attention_dim.
    min_wedge_deg   : minimum missing arc (°) for a direction to be carved in training.
    stride          : inference patch stride (defaults to cube_size // 2).
    num_workers     : DataLoader workers.
    start_volume_dir: if set, skip saving reconstruction and use this directory as
        round_00 (the K .npy files must already exist there).

    Returns
    -------
    final_volume_dir : Path to the ``final/`` directory containing the K corrected
                       .npy volumes produced by applying the fully-trained model to
                       the original round_00 data.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Save shared metadata ──────────────────────────────────────────────
    rsm_dirs_path = out / 'rsm_dirs.npy'
    if not rsm_dirs_path.exists():
        np.save(rsm_dirs_path, np.asarray(rsm_dirs, dtype=np.float64))
        logger.info("Saved rsm_dirs → %s", rsm_dirs_path)

    # ── Round 0: save initial FBP volumes (or use pre-existing dir) ───────
    if start_volume_dir is not None:
        round0_dir = Path(start_volume_dir)
        logger.info("Using pre-existing volume directory as round_00: %s", round0_dir)
    else:
        round0_dir = out / 'round_00'
        logger.info("Saving initial reconstruction to %s …", round0_dir)
        save_reconstruction_volumes(reconstruction, round0_dir)

    # ── Fixed per-volume normalization stats from round_00 (target) ───────
    # Computed once and reused by every round + inference, so input and target
    # always share identical, stable per-volume scales (the wedge-fill task is
    # scale-equivariant; freezing the stats keeps the objective stationary).
    norm_stats_path = out / 'norm_stats.npy'
    if not norm_stats_path.exists():
        sphere_mask = make_sphere_mask(cube_size)
        np.save(norm_stats_path, compute_norm_stats(round0_dir, mask=sphere_mask))
        logger.info("Saved norm_stats → %s", norm_stats_path)

    # ── Detect last fully-completed round (safe resume after crash) ───────
    # A round is complete when both its checkpoint.pt AND its volumes directory
    # (with K .npy files) exist.  Scan in reverse so we find the latest one.
    K = len(rsm_dirs)
    current_volume_dir = round0_dir
    prev_checkpoint: Optional[Path] = None
    start_round = 1

    for rnd in range(num_rounds, 0, -1):
        ckpt  = out / f'round_{rnd:02d}' / 'checkpoint.pt'
        vols  = out / f'round_{rnd:02d}_volumes'
        n_vols = len(list(vols.glob('vol_*.npy'))) if vols.exists() else 0
        if ckpt.exists() and n_vols == K:
            prev_checkpoint    = ckpt
            current_volume_dir = vols
            start_round        = rnd + 1
            logger.info(
                "Resuming: rounds 1–%d already complete. Starting from round %d.",
                rnd, start_round,
            )
            break

    if start_round == 1 and prev_checkpoint is None:
        logger.info("No completed rounds found. Starting from round 1.")

    for rnd in range(start_round, num_rounds + 1):
        round_tag = f'round_{rnd:02d}'
        logger.info("=" * 60)
        logger.info("ROUND %d / %d", rnd, num_rounds)
        logger.info("=" * 60)

        checkpoint_dir = out / round_tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── Train ─────────────────────────────────────────────────────
        checkpoint = _run_training(
            volume_dir=current_volume_dir,
            target_dir=round0_dir,
            rsm_dirs_path=rsm_dirs_path,
            norm_stats_path=norm_stats_path,
            alpha_deg=alpha_deg,
            half_space=half_space,
            checkpoint_dir=checkpoint_dir,
            epochs=epochs_per_round,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            cube_size=cube_size,
            n_samples=n_samples,
            conditioning_dim=conditioning_dim,
            min_wedge_deg=min_wedge_deg,
            max_rsm_wedge_deg=max_rsm_wedge_deg,
            resume=prev_checkpoint,
            num_workers=num_workers,
        )
        logger.info("Round %d training complete. Checkpoint: %s", rnd, checkpoint)

        # ── Infer ─────────────────────────────────────────────────────
        infer_output_dir = out / f'round_{rnd:02d}_volumes'
        _run_inference(
            volume_dir=current_volume_dir,
            rsm_dirs_path=rsm_dirs_path,
            norm_stats_path=norm_stats_path,
            checkpoint=checkpoint,
            alpha_deg=alpha_deg,
            half_space=half_space,
            output_dir=infer_output_dir,
            cube_size=cube_size,
            stride=stride,
            conditioning_dim=conditioning_dim,
        )
        logger.info("Round %d inference complete. Output: %s", rnd, infer_output_dir)

        # Cascade: next round trains on this round's corrected volumes
        current_volume_dir = infer_output_dir
        prev_checkpoint = checkpoint

    # ── Final inference on the original FBP volumes ───────────────────────
    # Apply the fully-trained model to the original round_00 data (not the
    # cascaded approximations) to produce the definitive pipeline output.
    # Fourier consistency is enforced against the original FBP frequencies.
    logger.info("=" * 60)
    logger.info("FINAL INFERENCE on original FBP volumes (round_00)")
    logger.info("=" * 60)
    final_dir = out / 'final'
    _run_inference(
        volume_dir=round0_dir,
        rsm_dirs_path=rsm_dirs_path,
        norm_stats_path=norm_stats_path,
        checkpoint=prev_checkpoint,
        alpha_deg=alpha_deg,
        half_space=half_space,
        output_dir=final_dir,
        cube_size=cube_size,
        stride=stride,
        conditioning_dim=conditioning_dim,
    )
    logger.info("Final output saved to: %s", final_dir)
    return final_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='SAXS-TT missing-wedge correction pipeline')
    p.add_argument('--reconstruction_path', required=True,
                   help='Path to reconstruction .pt file with keys "reconstruction" and "y_directions"')
    p.add_argument('--rsm_dirs_path',        default=None,
                   help='Optional separate (K,3) .npy file for rsm_dirs (overrides reconstruction_path key)')
    p.add_argument('--alpha_deg',            type=float, required=True)
    p.add_argument('--half_space',           default='y', choices=['y', 'z'])
    p.add_argument('--output_dir',           required=True)
    p.add_argument('--num_rounds',           type=int, default=3)
    p.add_argument('--epochs_per_round',     type=int, default=100)
    p.add_argument('--batch_size',           type=int, default=2)
    p.add_argument('--lr',                   type=float, default=1e-4)
    p.add_argument('--weight_decay',         type=float, default=0.0)
    p.add_argument('--cube_size',            type=int, default=32)
    p.add_argument('--n_samples',            type=int, default=2000)
    p.add_argument('--conditioning_dim',     type=int, default=128)
    p.add_argument('--min_wedge_deg',        type=float, default=10.0)
    p.add_argument('--stride',               type=int, default=None)
    p.add_argument('--num_workers',          type=int, default=0)
    p.add_argument('--log_level',            default='INFO')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(message)s',
    )

    saved = torch.load(args.reconstruction_path, map_location='cpu')
    reconstruction = saved['reconstruction']

    if args.rsm_dirs_path is not None:
        rsm_dirs = np.load(args.rsm_dirs_path)
    else:
        rsm_dirs = np.asarray(saved['y_directions'])

    run_pipeline(
        reconstruction=reconstruction,
        rsm_dirs=rsm_dirs,
        alpha_deg=args.alpha_deg,
        half_space=args.half_space,
        output_dir=args.output_dir,
        num_rounds=args.num_rounds,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cube_size=args.cube_size,
        n_samples=args.n_samples,
        conditioning_dim=args.conditioning_dim,
        min_wedge_deg=args.min_wedge_deg,
        stride=args.stride,
        num_workers=args.num_workers,
    )
