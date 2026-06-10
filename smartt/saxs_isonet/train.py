"""Training script for the SAXS-TT missing-wedge correction model.

Called as a subprocess by pipeline.py each round:

    python -m smartt.saxs_isonet.train \\
        --volume_dir     /path/to/round_N_volumes/ \\
        --target_dir     /path/to/round_00/ \\
        --rsm_dirs_path  /path/to/rsm_dirs.npy \\
        --norm_stats_path /path/to/norm_stats.npy \\
        --alpha_deg      45.0 \\
        --half_space     y \\
        --output_dir     /path/to/checkpoints/ \\
        --epochs         100 \\
        [--resume        /path/to/prev_checkpoint.pt]

A single shared UNet3DConditionModel is trained on all K volumes.  The dataset
returns raw patch_size windows (zero-padded so they can straddle the boundary);
the GPU VolumeAugmentor rotates @patch_size, crops @cube_size, and carves the
canonical wedge.  Loss = scale-normalised complex Fourier MSE inside the carved
wedge AND where the (rotated) source data is measured.  In dual-source mode the
target is the frozen round_00 volume (input changes each round, target does not).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from smartt.saxs_isonet.augment import VolumeAugmentor
from smartt.saxs_isonet.dataset import MissingWedgeSAXS
from smartt.saxs_isonet.wedge import goniometer_axis_for_half_space

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cross_attention_dim: int = 128) -> nn.Module:
    """Instantiate the UNet3DConditionModel from diffusers.

    Inputs:
      - sample:                (B, 1, P, P, P) — carved input
      - timestep:              (B,) long — zeros (direct regression, no diffusion)
      - encoder_hidden_states: (B, 1, cross_attention_dim) — wedge-size conditioning
    Output: predicted complete patch (B, 1, P, P, P).
    """
    try:
        from diffusers import UNet3DConditionModel
    except ImportError:
        raise ImportError(
            "diffusers is required for the SAXS isonet model. "
            "Install with: pip install diffusers"
        )

    model = UNet3DConditionModel(
        sample_size=None,
        in_channels=1,
        out_channels=1,
        down_block_types=(
            'DownBlock3D',
            'DownBlock3D',
            'CrossAttnDownBlock3D',
        ),
        up_block_types=(
            'CrossAttnUpBlock3D',
            'UpBlock3D',
            'UpBlock3D',
        ),
        block_out_channels=(32, 64, 128),
        layers_per_block=2,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=8,
        norm_num_groups=8,
    )
    return model


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def fourier_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_missing: torch.Tensor,
) -> torch.Tensor:
    """Scale-normalised complex Fourier MSE in the carved missing-wedge region.

    pred, target  : (B, 1, P, P, P) float32
    valid_missing : (B, P, P, P) bool — True = supervise (fftshifted layout).

    Uses the fftshifted layout to match the masks produced by VolumeAugmentor.
    """
    p = pred.squeeze(1).float()    # (B, P, P, P)
    t = target.squeeze(1).float()

    F_pred = torch.fft.fftshift(torch.fft.fftn(p, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    F_tgt  = torch.fft.fftshift(torch.fft.fftn(t, dim=(-3, -2, -1)), dim=(-3, -2, -1))

    vm = valid_missing.to(p.device).to(F_pred.real.dtype)   # (B, P, P, P)
    masked_pred = F_pred * vm
    masked_tgt  = F_tgt  * vm

    # Normalise by RMS magnitude of the target inside the missing region.
    n_valid = vm.sum(dim=(-3, -2, -1), keepdim=True).clamp(min=1)
    scale = (
        masked_tgt.abs().pow(2).sum(dim=(-3, -2, -1), keepdim=True) / n_valid
    ).sqrt() + 1e-8                                          # (B, 1, 1, 1)

    loss_fourier = nn.functional.mse_loss(
        torch.view_as_real(masked_pred) / scale.unsqueeze(-1),
        torch.view_as_real(masked_tgt)  / scale.unsqueeze(-1),
    )
    # Real-space MSE anchors the DC component: the Fourier loss has zero gradient
    # for any constant offset (DC is never in valid_missing), so without this term
    # the model's bias drifts freely to large values.
    loss_real = nn.functional.mse_loss(p, t)
    return loss_fourier + loss_real


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    volume_dir: str,
    rsm_dirs: np.ndarray,
    alpha_deg: float,
    half_space: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    cube_size: int = 32,
    n_samples: int = 2000,
    conditioning_dim: int = 128,
    min_wedge_deg: float = 10.0,
    max_rsm_wedge_deg: float | None = None,
    target_dir: str | None = None,
    norm_stats: np.ndarray | None = None,
    resume: str | None = None,
    device: torch.device | None = None,
    num_workers: int = 0,
) -> Path:
    """Train (or fine-tune) the missing-wedge correction model.

    Parameters
    ----------
    volume_dir    : directory with K .npy volumes (this round's input).
    rsm_dirs      : (K, 3) RSM directions.
    alpha_deg     : polar-cap half-angle (degrees).
    half_space    : 'z' or 'y' — determines goniometer_axis.
    output_dir    : where to save the checkpoint.
    epochs        : training epochs this round (always counts from 1).
    batch_size    : samples per gradient step.
    lr            : AdamW learning rate.
    weight_decay  : AdamW weight decay.
    cube_size     : model input side-length (load size = ceil(√3·cube_size)).
    n_samples     : virtual dataset length per epoch.
    conditioning_dim : must match cross_attention_dim of the model.
    min_wedge_deg : minimum missing arc (°) for a direction to be carved.
    target_dir    : frozen target volumes (round_00); None = single-source.
    norm_stats    : (K, 2) fixed per-volume (mean, std) from round_00.
    resume        : checkpoint .pt to fine-tune from (weights only).
    device        : compute device (defaults to CUDA if available).
    num_workers   : DataLoader workers.

    Returns
    -------
    checkpoint_path : Path to the saved checkpoint.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Training on device: %s", device)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    goniometer_axis = goniometer_axis_for_half_space(half_space)

    # ── Dataset / DataLoader (returns raw windows + indices) ──────────────
    dataset = MissingWedgeSAXS(
        volume_dir=volume_dir,
        rsm_dirs=rsm_dirs,
        alpha_deg=alpha_deg,
        goniometer_axis=goniometer_axis,
        cube_size=cube_size,
        n_samples=n_samples,
        min_wedge_deg=min_wedge_deg,
        max_rsm_wedge_deg=max_rsm_wedge_deg,
        target_dir=target_dir,
        norm_stats=norm_stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    # ── GPU augmentor (rotation @patch → crop @cube → canonical carving) ──
    augmentor = VolumeAugmentor(
        rsm_dirs=rsm_dirs,
        alpha_deg=alpha_deg,
        goniometer_axis=goniometer_axis,
        cube_size=cube_size,
        conditioning_dim=conditioning_dim,
        device=device,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cross_attention_dim=conditioning_dim).to(device)
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        logger.info("Fine-tuning from checkpoint %s", resume)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Per-epoch cosine: reaches eta_min at the last epoch, not zero.
    # Stepping per-batch with get_cosine_schedule_with_warmup caused lr to
    # reach 0 by the final batch, leaving most of the last epoch at near-zero
    # lr.  A per-epoch cosine keeps lr ≥ 50% of base through the first half
    # of training and never actually hits zero.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.1,
    )

    # ── Training ──────────────────────────────────────────────────────────
    from tqdm import tqdm

    model.train()
    epoch_bar = tqdm(range(epochs), desc='Training', unit='epoch')
    for epoch in epoch_bar:
        epoch_loss = 0.0
        n_batches = 0

        batch_bar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', leave=False)
        for v0, v1, k_src, k_wedge in batch_bar:
            carved, target, cond, valid_missing = augmentor(v0, v1, k_src, k_wedge)
            timesteps = torch.zeros(carved.shape[0], device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            pred = model(
                carved,
                timestep=timesteps,
                encoder_hidden_states=cond,
                return_dict=False,
            )[0]  # (B, 1, P, P, P)

            loss = fourier_mse_loss(pred, target, valid_missing)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            batch_bar.set_postfix(loss=f'{loss.item():.3e}')

        scheduler.step()   # per epoch — lr stays useful throughout
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_bar.set_postfix(loss=f'{avg_loss:.3e}', lr=f'{scheduler.get_last_lr()[0]:.2e}')
        logger.info("Epoch %d/%d  loss=%.4e  lr=%.2e",
                    epoch + 1, epochs, avg_loss, scheduler.get_last_lr()[0])

        ckpt_path = out_dir / 'checkpoint.pt'
        torch.save({
            'model': model.state_dict(),
            'alpha_deg': alpha_deg,
            'half_space': half_space,
            'cube_size': cube_size,
            'conditioning_dim': conditioning_dim,
        }, ckpt_path)
        logger.info("Checkpoint saved → %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train SAXS-TT missing-wedge model')
    p.add_argument('--volume_dir',      required=True)
    p.add_argument('--rsm_dirs_path',   required=True, help='Path to (K,3) .npy file')
    p.add_argument('--alpha_deg',       type=float, required=True)
    p.add_argument('--half_space',      default='y', choices=['y', 'z'])
    p.add_argument('--output_dir',      required=True)
    p.add_argument('--target_dir',      default=None, help='Frozen target volumes (round_00)')
    p.add_argument('--norm_stats_path', default=None, help='(K,2) fixed per-volume mean/std .npy')
    p.add_argument('--epochs',          type=int, default=100)
    p.add_argument('--batch_size',      type=int, default=2)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--weight_decay',    type=float, default=0.0)
    p.add_argument('--cube_size',       type=int, default=32)
    p.add_argument('--n_samples',       type=int, default=2000)
    p.add_argument('--conditioning_dim',type=int, default=128)
    p.add_argument('--min_wedge_deg',       type=float, default=1.0)
    p.add_argument('--max_rsm_wedge_deg',   type=float, default=None,
                   help='Max missing arc (°) for a volume to be used as training source')
    p.add_argument('--resume',              default=None)
    p.add_argument('--num_workers',     type=int, default=0)
    p.add_argument('--log_level',       default='INFO')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(message)s',
    )
    rsm_dirs = np.load(args.rsm_dirs_path)
    norm_stats = np.load(args.norm_stats_path) if args.norm_stats_path else None
    train(
        volume_dir=args.volume_dir,
        rsm_dirs=rsm_dirs,
        alpha_deg=args.alpha_deg,
        half_space=args.half_space,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        cube_size=args.cube_size,
        n_samples=args.n_samples,
        conditioning_dim=args.conditioning_dim,
        min_wedge_deg=args.min_wedge_deg,
        max_rsm_wedge_deg=args.max_rsm_wedge_deg,
        target_dir=args.target_dir,
        norm_stats=norm_stats,
        resume=args.resume,
        num_workers=args.num_workers,
    )
