"""Training script for the SAXS-TT missing-wedge correction model.

Called as a subprocess by pipeline.py each round:

    python -m smartt.saxs_isonet.train \\
        --volume_dir  /path/to/round_N/ \\
        --y_dirs_path /path/to/y_dirs.npy \\
        --alpha_deg   45.0 \\
        --half_space  y \\
        --output_dir  /path/to/checkpoints/ \\
        --epochs      100 \\
        [--resume     /path/to/prev_checkpoint.pt]

The script trains a single shared UNet3DConditionModel on all K volumes.
Missing-arc conditioning is passed via encoder_hidden_states.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from smartt.saxs_isonet.dataset import MissingWedgeSAXS
from smartt.saxs_isonet.wedge import goniometer_axis_for_half_space

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cross_attention_dim: int = 128) -> nn.Module:
    """Instantiate the UNet3DConditionModel from diffusers.

    The model takes:
      - sample:                (B, 1, P, P, P) — noisy/carved input
      - timestep:              (B,) long — zeros (direct regression, no diffusion)
      - encoder_hidden_states: (B, 1, cross_attention_dim) — wedge conditioning

    Returns the predicted complete patch (B, 1, P, P, P).
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
            'CrossAttnDownBlock3D',
            'CrossAttnDownBlock3D',
            'CrossAttnDownBlock3D',
            'DownBlock3D',
        ),
        up_block_types=(
            'UpBlock3D',
            'CrossAttnUpBlock3D',
            'CrossAttnUpBlock3D',
            'CrossAttnUpBlock3D',
        ),
        block_out_channels=(32, 64, 128, 128),
        layers_per_block=1,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=8,
        norm_num_groups=8,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    volume_dir: str,
    y_dirs: np.ndarray,
    alpha_deg: float,
    half_space: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    patch_size: int = 64,
    n_samples: int = 2000,
    conditioning_dim: int = 128,
    resume: str | None = None,
    device: torch.device | None = None,
    num_workers: int = 0,
) -> Path:
    """Train (or fine-tune) the missing-wedge correction model.

    Parameters
    ----------
    volume_dir    : directory with K .npy volumes from the previous round.
    y_dirs        : (K, 3) q-directions.
    alpha_deg     : polar-cap half-angle (degrees).
    half_space    : 'z' or 'y' — determines goniometer_axis.
    output_dir    : where to save the final checkpoint.
    epochs        : training epochs this round.
    batch_size    : samples per gradient step.
    lr            : Adam learning rate.
    patch_size    : spatial side-length of cubic patches.
    n_samples     : virtual dataset length per epoch.
    conditioning_dim : must match cross_attention_dim of the model.
    resume        : path to a checkpoint .pt to fine-tune from.
    device        : compute device (defaults to CUDA if available).
    num_workers   : DataLoader workers (0 = main process).

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

    # ── Dataset / DataLoader ──────────────────────────────────────────────
    dataset = MissingWedgeSAXS(
        volume_dir=volume_dir,
        y_dirs=y_dirs,
        alpha_deg=alpha_deg,
        goniometer_axis=goniometer_axis,
        patch_size=patch_size,
        n_samples=n_samples,
        conditioning_dim=conditioning_dim,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    logger.info("Dataset: K=%d volumes, patch_size=%d, %d samples/epoch, batch=%d",
                dataset.K, patch_size, n_samples, batch_size)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cross_attention_dim=conditioning_dim).to(device)

    start_epoch = 0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        logger.info("Resumed from %s (epoch %d)", resume, start_epoch)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1,
    )

    # ── Training ──────────────────────────────────────────────────────────
    from tqdm import tqdm

    def _fourier_mse_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_missing: torch.Tensor,
    ) -> torch.Tensor:
        """Scale-normalised complex MSE in the carved missing-wedge region.

        pred, target   : (B, 1, P, P, P) float32
        valid_missing  : (B, P, P, P) bool — True = carved frequency to supervise
        """
        p = pred.squeeze(1).float()    # (B, P, P, P)
        t = target.squeeze(1).float()

        F_pred = torch.fft.fftn(p, dim=(-3, -2, -1))
        F_tgt  = torch.fft.fftn(t, dim=(-3, -2, -1))

        vm = valid_missing.to(p.device)           # (B, P, P, P)
        masked_pred = F_pred * vm
        masked_tgt  = F_tgt  * vm

        # Scale by RMS magnitude of target in the missing region to normalise
        n_valid = vm.sum(dim=(-3, -2, -1), keepdim=True).clamp(min=1)
        scale = (
            masked_tgt.abs().pow(2).sum(dim=(-3, -2, -1), keepdim=True) / n_valid
        ).sqrt() + 1e-8                           # (B, 1, 1, 1)

        return nn.functional.mse_loss(
            torch.view_as_real(masked_pred) / scale.unsqueeze(-1),
            torch.view_as_real(masked_tgt)  / scale.unsqueeze(-1),
        )

    model.train()
    epoch_bar = tqdm(
        range(start_epoch, start_epoch + epochs),
        desc='Training',
        unit='epoch',
    )
    for epoch in epoch_bar:
        epoch_loss = 0.0
        n_batches = 0

        batch_bar = tqdm(loader, desc=f'Epoch {epoch + 1}', unit='batch', leave=False)
        for carved, target, cond, valid_missing in batch_bar:
            # carved        : (B, 1, P, P, P)
            # target        : (B, 1, P, P, P)
            # cond          : (B, 1, 1, dim) — squeeze extra dim from stacking
            # valid_missing : (B, P, P, P) bool
            carved = carved.to(device)
            target = target.to(device)
            cond   = cond.squeeze(2).to(device)   # (B, 1, dim)

            timesteps = torch.zeros(carved.shape[0], device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            pred = model(
                carved,
                timestep=timesteps,
                encoder_hidden_states=cond,
                return_dict=False,
            )[0]  # (B, 1, P, P, P)

            loss = _fourier_mse_loss(pred, target, valid_missing)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            batch_bar.set_postfix(loss=f'{loss.item():.3e}')

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_bar.set_postfix(loss=f'{avg_loss:.3e}', lr=f'{scheduler.get_last_lr()[0]:.2e}')
        logger.info("Epoch %d/%d  loss=%.4e  lr=%.2e",
                    epoch + 1, start_epoch + epochs, avg_loss,
                    scheduler.get_last_lr()[0])

    # ── Save checkpoint ───────────────────────────────────────────────────
    ckpt_path = out_dir / 'checkpoint.pt'
    torch.save({
        'model': model.state_dict(),
        'epoch': start_epoch + epochs,
        'alpha_deg': alpha_deg,
        'half_space': half_space,
        'patch_size': patch_size,
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
    p.add_argument('--y_dirs_path',     required=True, help='Path to (K,3) .npy file')
    p.add_argument('--alpha_deg',       type=float, required=True)
    p.add_argument('--half_space',      default='y', choices=['y', 'z'])
    p.add_argument('--output_dir',      required=True)
    p.add_argument('--epochs',          type=int, default=100)
    p.add_argument('--batch_size',      type=int, default=2)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--patch_size',      type=int, default=64)
    p.add_argument('--n_samples',       type=int, default=2000)
    p.add_argument('--conditioning_dim',type=int, default=128)
    p.add_argument('--resume',          default=None)
    p.add_argument('--num_workers',     type=int, default=0)
    p.add_argument('--log_level',       default='INFO')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(message)s',
    )
    y_dirs = np.load(args.y_dirs_path)
    train(
        volume_dir=args.volume_dir,
        y_dirs=y_dirs,
        alpha_deg=args.alpha_deg,
        half_space=args.half_space,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        n_samples=args.n_samples,
        conditioning_dim=args.conditioning_dim,
        resume=args.resume,
        num_workers=args.num_workers,
    )
