"""GPU-batched augmentation for SAXS-TT missing-wedge training.

The CPU dataset loads full spherically-masked (cube_size, cube_size, cube_size)
volumes; the SO(3) rotation and Fourier carving run here on the training device.

Pipeline per batch:
  1. Rotate the full cube volumes by a random SO(3) `R` (same R for v0 & v1).
     The spherical mask makes corners safely zero after rotation — no oversampled
     load buffer or center-crop is needed.
  2. Carve the **canonical** wedge of `k_wedge` from v0 → model input.
  3. target = rotated v1 (the fixed round_00 volume in dual-source mode).
  4. valid_missing = (carved-missing) AND (source k_src measured-after-rotation).

Fourier convention: fftshifted (DC centred), so a frequency mask rotates with
`grid_sample` exactly like a real volume (rotation about centre == about DC).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from smartt.saxs_isonet.wedge import (
    _unit,
    canonical_goniometer_axis,
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

    ``grid_sample`` does inverse sampling, so ``theta`` uses ``Rᵀ`` to rotate the
    *content* by ``R``.  ``mode='nearest'`` is used for boolean masks.
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
    """Rotate → carve a batch on the GPU.

    Masks are precomputed at ``cube_size`` (fftshifted):
      - ``canonical_keep[k]`` : canonical (orientation-normalised) wedge keep-mask
      - ``measured_keep[k]``  : natural-orientation measured-frequency mask
      - ``cond[k]``           : scalar missing-arc conditioning embedding
    """

    def __init__(
        self,
        rsm_dirs: np.ndarray,
        alpha_deg: float,
        goniometer_axis: Optional[np.ndarray] = None,
        cube_size: int = 32,
        conditioning_dim: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device('cpu')
        self.cube_size = cube_size
        self.alpha_deg = float(alpha_deg)
        self.rsm_dirs = np.asarray(rsm_dirs, dtype=np.float64)
        g = (_unit(goniometer_axis) if goniometer_axis is not None
             else np.array([0., 0., 1.]))
        self.goniometer_axis = g.copy()
        shape = (cube_size, cube_size, cube_size)

        canonical = [
            np.fft.fftshift(missing_wedge_mask_3d(
                canonical_rsm_dir(r, g),          # always y_hat
                alpha_deg, shape,
                canonical_goniometer_axis(r, g),  # (0, cosθ, sinθ) → kz-aligned wedge
            ))
            for r in rsm_dirs
        ]
        measured = [
            np.fft.fftshift(missing_wedge_mask_3d(r, alpha_deg, shape, g))
            for r in rsm_dirs
        ]
        self.canonical_keep = torch.from_numpy(np.stack(canonical)).to(self.device)  # (K,C,C,C) bool
        self.measured_keep = torch.from_numpy(np.stack(measured)).to(self.device)    # (K,C,C,C) bool

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

    def _measured_mask_rotated(self, R: torch.Tensor,
                               k_src: torch.Tensor) -> torch.Tensor:
        """Compute the measured-frequency mask after rotation R analytically.

        After rotating a volume by R, frequency f (in the rotated frame) was
        measured by k_src iff  R@rsm_dirs[k_src] and R@g  satisfy the same
        closed-form condition as the original mask.  This avoids the aliasing
        introduced by rotating a discrete boolean mask with nearest neighbours.

        Returns (B, C, C, C) bool in **fftshifted** layout, matching canonical_keep.
        """
        B = R.shape[0]
        device = R.device
        C = self.cube_size
        cos_alpha = float(np.cos(np.radians(self.alpha_deg)))

        rsm_k = torch.from_numpy(
            self.rsm_dirs[k_src.cpu().numpy()]
        ).float().to(device)                                          # (B, 3)
        g_t = torch.from_numpy(self.goniometer_axis).float().to(device)  # (3,)

        # Rotate RSM dir and goniometer axis into the augmented frame.
        rsm_rot = torch.einsum('bij,bj->bi', R, rsm_k)               # (B, 3)
        g_rot   = (R @ g_t.unsqueeze(-1)).squeeze(-1)                 # (B, 3)

        # Normalise (R is orthogonal; clamp guards against numerical drift).
        rsm_rot = rsm_rot / rsm_rot.norm(dim=1, keepdim=True).clamp(min=1e-10)
        g_rot   = g_rot   / g_rot.norm(  dim=1, keepdim=True).clamp(min=1e-10)

        # fftshifted frequency grid: centre = DC, coords in (-0.5, 0.5].
        idx = torch.arange(C, device=device)
        freq = (idx - C // 2).float() / C                            # (C,)
        FX, FY, FZ = torch.meshgrid(freq, freq, freq, indexing='ij') # (C,C,C)

        # Broadcast batch dim → (B,1,1,1).
        def b(t): return t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y0, y1, y2 = b(rsm_rot[:, 0]), b(rsm_rot[:, 1]), b(rsm_rot[:, 2])
        g0, g1, g2 = b(g_rot[:, 0]),   b(g_rot[:, 1]),   b(g_rot[:, 2])
        FX, FY, FZ = FX.unsqueeze(0), FY.unsqueeze(0), FZ.unsqueeze(0)  # (1,C,C,C)

        # Scalar triple product (y × f) · g  — the same formula as missing_wedge_mask_3d.
        cross_dot_g = (
            g0 * (y1 * FZ - y2 * FY) +
            g1 * (y2 * FX - y0 * FZ) +
            g2 * (y0 * FY - y1 * FX)
        )                                                              # (B,C,C,C)

        f_dot_y = y0 * FX + y1 * FY + y2 * FZ
        f_perp  = (FX**2 + FY**2 + FZ**2 - f_dot_y**2).clamp(min=0.).sqrt()

        mask = cross_dot_g.abs() <= cos_alpha * f_perp                # (B,C,C,C)

        # Friedel symmetry: keep(f) == keep(-f).
        neg = (-torch.arange(C, device=device)) % C
        mask_neg = mask[:, neg][:, :, neg][:, :, :, neg]
        mask = mask & mask_neg

        mask[:, C // 2, C // 2, C // 2] = True                       # DC always measured
        return mask

    def __call__(self, v0: torch.Tensor, v1: Optional[torch.Tensor],
                 k_src: torch.Tensor, k_wedge: torch.Tensor):
        """Augment a batch.

        Parameters
        ----------
        v0 : (B, 1, P, P, P) or (B, P, P, P) — raw patch_size windows (input source).
        v1 : same shape — frozen target windows (round_00), or None for single-source
             (then v1 = v0).
        k_src   : (B,) long — source RSM index (for the measured mask).
        k_wedge : (B,) long — RSM index whose canonical wedge is carved.

        Returns
        -------
        carved        : (B, 1, C, C, C)
        target        : (B, 1, C, C, C)
        cond          : (B, 1, dim)
        valid_missing : (B, C, C, C) bool (fftshifted layout)
        """
        if v0.dim() == 5:
            v0 = v0.squeeze(1)
        v0 = v0.to(self.device).float()
        if v1 is None:
            v1 = v0
        else:
            if v1.dim() == 5:
                v1 = v1.squeeze(1)
            v1 = v1.to(self.device).float()
        k_src = k_src.to(self.device)
        k_wedge = k_wedge.to(self.device)
        B = v0.shape[0]
        C = self.cube_size

        # Same rotation for v0 and v1. The spherical mask keeps corners near-zero
        # after rotation, so no center-crop buffer is needed.
        R = random_so3(B, self.device, v0.dtype)
        v0c = rotate_batch(v0, R, mode='bilinear')
        v1c = rotate_batch(v1, R, mode='bilinear')

        keep_canon = self.canonical_keep[k_wedge]                    # (B,C,C,C) bool (shifted)
        carved = carve_shifted(v0c, keep_canon)                      # (B,C,C,C)
        target = v1c                                                 # (B,C,C,C)

        # Analytical measured mask — avoids nearest-neighbour aliasing by
        # rotating the RSM direction and goniometer axis then re-evaluating
        # the closed-form condition, giving smooth wedge boundaries.
        measured_rot = self._measured_mask_rotated(R, k_src)         # (B,C,C,C) bool (shifted)
        valid_missing = (~keep_canon) & measured_rot                 # carved AND ground-truth

        cond = self.cond[k_wedge]                                    # (B,1,dim)
        return carved.unsqueeze(1), target.unsqueeze(1), cond, valid_missing
