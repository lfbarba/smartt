"""GPU-batched augmentation for SAXS-TT missing-wedge training.

The CPU dataset only slices raw patches; the expensive SO(3) rotation and
Fourier carving run here on the training device, mirroring the VolumeAugmentor
design in isodiffusion (rotating volumes on CPU is the training bottleneck).

Fourier convention
-------------------
All Fourier ops use the **fftshifted** layout (DC at the volume centre).  This
lets a frequency-domain mask be rotated with ``grid_sample`` exactly like a
real-space volume: rotation about the array centre == rotation about DC.

What the augmentor produces per sample (matching the agreed design)
-------------------------------------------------------------------
- ``carved``        : rotated patch with the **canonical** wedge of ``k_wedge``
                      zeroed (model input).
- ``target``        : the rotated patch (supervision target).
- ``cond``          : scalar missing-arc embedding for ``k_wedge``.
- ``valid_missing`` : (carved-missing) AND (source ``k_src`` measured-after-rotation)
                      — supervise only where we carved AND have trustworthy data.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from smartt.saxs_isonet.wedge import (
    _unit,
    canonical_rsm_dir,
    missing_arc_length,
    missing_wedge_mask_3d,
    sinusoidal_wedge_embedding,
)


# ---------------------------------------------------------------------------
# Rotation primitives
# ---------------------------------------------------------------------------

def random_so3(batch_size: int, device: torch.device,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(B, 3, 3) uniform SO(3) rotation matrices (Shoemake's quaternion method)."""
    u = torch.rand(batch_size, 3, device=device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    q1 = torch.sqrt(1.0 - u1) * torch.sin(2.0 * math.pi * u2)
    q2 = torch.sqrt(1.0 - u1) * torch.cos(2.0 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2.0 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2.0 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    rows = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ]
    return torch.stack(rows, dim=1).reshape(batch_size, 3, 3).to(dtype)


def rotate_batch(vols: torch.Tensor, R: torch.Tensor,
                 mode: str = 'bilinear') -> torch.Tensor:
    """Rotate a (B, D, H, W) batch by per-sample (B, 3, 3) rotations.

    ``grid_sample`` performs inverse sampling, so ``theta`` uses ``Rᵀ`` to rotate
    the *content* by ``R``.  ``mode='nearest'`` is used for boolean masks.
    """
    B, D, H, W = vols.shape
    theta = torch.zeros(B, 3, 4, device=vols.device, dtype=vols.dtype)
    theta[:, :, :3] = R.to(device=vols.device, dtype=vols.dtype).transpose(1, 2)
    grid = F.affine_grid(theta, (B, 1, D, H, W), align_corners=False)
    out = F.grid_sample(vols.unsqueeze(1), grid, mode=mode,
                        padding_mode='zeros', align_corners=False)
    return out.squeeze(1)


# ---------------------------------------------------------------------------
# Fourier carving (shifted layout)
# ---------------------------------------------------------------------------

def carve_shifted(vols: torch.Tensor, keep_shifted: torch.Tensor) -> torch.Tensor:
    """Zero the missing wedge of (B, P, P, P) volumes given a shifted keep-mask.

    keep_shifted : (B, P, P, P) bool — True = measured (fftshifted layout).
    """
    Fs = torch.fft.fftshift(torch.fft.fftn(vols, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    Fs = Fs * keep_shifted.to(Fs.real.dtype)
    return torch.fft.ifftn(torch.fft.ifftshift(Fs, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real


# ---------------------------------------------------------------------------
# Augmentor
# ---------------------------------------------------------------------------

class VolumeAugmentor:
    """Rotate + carve a batch of raw patches on the GPU.

    Precomputes, per RSM direction k (all in fftshifted layout):
      - ``canonical_keep[k]`` : canonical (orientation-normalised) wedge keep-mask
      - ``measured_keep[k]``  : natural-orientation measured-frequency mask
      - ``cond[k]``           : scalar missing-arc conditioning embedding
    """

    def __init__(
        self,
        rsm_dirs: np.ndarray,
        alpha_deg: float,
        goniometer_axis: Optional[np.ndarray] = None,
        patch_size: int = 64,
        conditioning_dim: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device('cpu')
        P = patch_size
        shape = (P, P, P)
        g = (_unit(goniometer_axis) if goniometer_axis is not None
             else np.array([0., 0., 1.]))

        # Canonical carved wedges (orientation fixed by canonical frame), shifted.
        canonical = [
            np.fft.fftshift(missing_wedge_mask_3d(
                canonical_rsm_dir(r, g), alpha_deg, shape, np.array([0., 0., 1.])))
            for r in rsm_dirs
        ]
        # Original measured masks in natural orientation, shifted.
        measured = [
            np.fft.fftshift(missing_wedge_mask_3d(r, alpha_deg, shape, g))
            for r in rsm_dirs
        ]
        self.canonical_keep = torch.from_numpy(np.stack(canonical)).to(self.device)  # (K,P,P,P) bool
        self.measured_keep = torch.from_numpy(np.stack(measured)).to(self.device)    # (K,P,P,P) bool

        conds = [
            sinusoidal_wedge_embedding(missing_arc_length(r, alpha_deg, g),
                                       dim=conditioning_dim)
            for r in rsm_dirs
        ]
        self.cond = torch.cat(conds, dim=0).to(self.device)   # (K, 1, dim)

    def to(self, device: torch.device) -> 'VolumeAugmentor':
        self.device = device
        self.canonical_keep = self.canonical_keep.to(device)
        self.measured_keep = self.measured_keep.to(device)
        self.cond = self.cond.to(device)
        return self

    def __call__(self, patches: torch.Tensor, k_src: torch.Tensor,
                 k_wedge: torch.Tensor):
        """Rotate + carve a batch.

        Parameters
        ----------
        patches : (B, 1, P, P, P) or (B, P, P, P) raw patches.
        k_src   : (B,) long — source RSM index per sample (for the measured mask).
        k_wedge : (B,) long — RSM index whose canonical wedge is carved.

        Returns
        -------
        carved        : (B, 1, P, P, P)
        target        : (B, 1, P, P, P)
        cond          : (B, 1, dim)
        valid_missing : (B, P, P, P) bool — supervise the Fourier loss here
                        (fftshifted layout, matching the training loss).
        """
        if patches.dim() == 5:
            patches = patches.squeeze(1)
        patches = patches.to(self.device).float()
        k_src = k_src.to(self.device)
        k_wedge = k_wedge.to(self.device)
        B = patches.shape[0]

        R = random_so3(B, self.device, patches.dtype)
        rotated = rotate_batch(patches, R, mode='bilinear')          # target (B,P,P,P)

        keep_canon = self.canonical_keep[k_wedge]                    # (B,P,P,P) bool (shifted)
        carved = carve_shifted(rotated, keep_canon)                  # (B,P,P,P)

        measured_rot = rotate_batch(self.measured_keep[k_src].float(), R,
                                    mode='nearest') > 0.5            # (B,P,P,P) bool
        valid_missing = (~keep_canon) & measured_rot                 # carved AND ground-truth

        cond = self.cond[k_wedge]                                    # (B,1,dim)
        return carved.unsqueeze(1), rotated.unsqueeze(1), cond, valid_missing
