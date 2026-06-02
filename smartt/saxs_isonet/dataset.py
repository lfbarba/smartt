"""Training dataset for the SAXS-TT missing-wedge correction pipeline.

Each __getitem__ call produces one self-supervised training triplet:

  (carved_patch, rotated_patch, conditioning_emb)

where

  rotated_patch = SO(3)(patch from random volume k_data)
  carved_patch  = IFFT( FFT(rotated_patch) * mask[k_wedge] )
  k_wedge       ~ Uniform{0, …, K-1}  (independent of k_data)

The random decoupling of k_data and k_wedge ensures that high-quality near-pole
volumes (small missing wedge) are used with all wedge sizes, maximising the
learning signal from the best-conditioned reconstructions.

Public API
----------
save_reconstruction_volumes(reconstruction, output_dir)
    Save the (K, X, Y, Z) tensor as K numbered .npy files.

MissingWedgeSAXS(Dataset)
    PyTorch Dataset.  Instantiate once; use with DataLoader as usual.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from smartt.saxs_isonet.wedge import (
    _unit,
    missing_wedge_mask_3d,
    y_dir_embedding,
)


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def save_reconstruction_volumes(
    reconstruction: torch.Tensor,
    output_dir: str | Path,
) -> list[Path]:
    """Save the (K, X, Y, Z) reconstruction tensor as K numbered .npy files.

    Parameters
    ----------
    reconstruction : (K, X, Y, Z) float32 tensor — output of saxs_fbp/gd_reconstruction.
    output_dir : target directory (created if absent).

    Returns
    -------
    paths : list of K Path objects, one per saved file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = reconstruction.detach().cpu().numpy() if isinstance(reconstruction, torch.Tensor) \
          else np.asarray(reconstruction)
    K = rec.shape[0]
    paths = []
    for k in range(K):
        p = out / f'vol_{k:04d}.npy'
        np.save(p, rec[k].astype(np.float32))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# SO(3) rotation utilities
# ---------------------------------------------------------------------------

def _random_rotation_matrix(device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Uniformly random SO(3) rotation matrix (3×3) via QR decomposition."""
    M = torch.randn(3, 3, device=device)
    Q, R_mat = torch.linalg.qr(M)
    # QR decomposition is unique up to sign — fix det to +1
    Q = Q * torch.sign(torch.linalg.det(Q))
    return Q


def _rotate_volume(patch: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate a (1, P, P, P) patch by a (3×3) rotation matrix.

    Uses trilinear interpolation via F.grid_sample.  Regions outside the
    original volume are filled with zero (padding_mode='zeros').

    Parameters
    ----------
    patch : (1, P, P, P) float32 — channel-first 3D patch.
    R     : (3, 3) float32 — SO(3) rotation matrix (maps output → input via R.T).

    Returns
    -------
    rotated : (1, P, P, P) float32
    """
    vol_b = patch.unsqueeze(0)             # (1, 1, P, P, P)

    # affine_grid expects theta = (B, 3, 4): [R | t] with t=0 for pure rotation.
    # grid_sample does inverse mapping: grid point = theta @ [x, y, z, 1]^T.
    # To rotate the content by R, the inverse mapping uses R^T.
    theta = torch.zeros(1, 3, 4, dtype=patch.dtype, device=patch.device)
    theta[0, :3, :3] = R.T.to(patch.device)

    grid = F.affine_grid(theta, vol_b.shape, align_corners=False)
    rotated = F.grid_sample(
        vol_b, grid, mode='bilinear', padding_mode='zeros', align_corners=False,
    )
    return rotated.squeeze(0)              # (1, P, P, P)


# ---------------------------------------------------------------------------
# Fourier masking
# ---------------------------------------------------------------------------

def _rotate_mask(mask: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate a (P, P, P) boolean mask by (3×3) rotation R (nearest-neighbour).

    Used to find which Fourier frequencies of the *rotated* source volume are
    actually measured, so the Fourier loss is only supervised on valid targets.
    """
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, P, P, P)
    theta = torch.zeros(1, 3, 4, dtype=torch.float32)
    theta[0, :3, :3] = R.T
    grid = F.affine_grid(theta, mask_f.shape, align_corners=False)
    rotated = F.grid_sample(
        mask_f, grid, mode='nearest', padding_mode='zeros', align_corners=False,
    )
    return rotated.squeeze() > 0.5              # (P, P, P) bool


def _apply_fourier_mask(patch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out the missing-wedge region of a patch in Fourier space.

    Parameters
    ----------
    patch : (1, P, P, P) float32 — rotated patch (target).
    mask  : (P, P, P) bool — True = measured, False = missing wedge.

    Returns
    -------
    carved : (1, P, P, P) float32 — patch with missing frequencies zeroed.
    """
    x = patch.squeeze(0)                   # (P, P, P)
    fft = torch.fft.fftn(x)
    fft[~mask] = 0.0
    carved = torch.fft.ifftn(fft).real     # (P, P, P)
    return carved.unsqueeze(0)             # (1, P, P, P)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MissingWedgeSAXS(Dataset):
    """Self-supervised dataset for SAXS-TT missing-wedge correction.

    Loads K .npy volumes from ``volume_dir`` (sorted by name, matching the
    order of ``y_dirs``).  Each call to ``__getitem__`` draws a random patch
    from a random volume, rotates it by a random SO(3) rotation, and carves
    a missing wedge sampled randomly from the K-distribution.

    The tuple returned is::

        carved_patch  — input to the model (missing frequencies zeroed).
        rotated_patch — supervision target (complete rotated patch).
        cond_emb      — (1, 1, dim) sinusoidal embedding of the wedge size.
        valid_missing — (P, P, P) bool, True where the wedge was carved.
                        Used by the Fourier loss to restrict supervision to
                        the carved region.

    Parameters
    ----------
    volume_dir : directory containing K .npy volume files (float32, shape (X,Y,Z)).
    y_dirs     : (K, 3) float64 ndarray — q-directions from fibonacci_hemisphere.
    alpha_deg  : half-angle (°) of the unmeasurable goniometer polar caps.
    goniometer_axis : tilt-axis unit vector.
        Use ``goniometer_axis_for_half_space(half_space)`` to match the
        convention used in saxs_fbp/gd_reconstruction. Defaults to [0,0,1].
    patch_size : spatial side-length (voxels) of cubic training patches.
    n_samples  : virtual dataset length (each epoch draws this many patches).
    normalize  : ``'volume'`` — per-volume zero-mean unit-variance before
        loading (recommended).  ``'none'`` — load raw values.
    conditioning_dim : dimensionality of the sinusoidal conditioning embedding.
        Must match the ``cross_attention_dim`` of the UNet3DConditionModel.
    """

    def __init__(
        self,
        volume_dir: str | Path,
        y_dirs: np.ndarray,
        alpha_deg: float,
        goniometer_axis: Optional[np.ndarray] = None,
        patch_size: int = 64,
        n_samples: int = 2000,
        normalize: str = 'volume',
        conditioning_dim: int = 128,
    ) -> None:
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.y_dirs = np.asarray(y_dirs, dtype=float)
        self.alpha_deg = float(alpha_deg)

        if goniometer_axis is None:
            goniometer_axis = np.array([0., 0., 1.])
        self.goniometer_axis = _unit(np.asarray(goniometer_axis, dtype=float))

        # ── Load volumes ──────────────────────────────────────────────────
        volume_dir = Path(volume_dir)
        paths = sorted(volume_dir.glob('*.npy'))
        if not paths:
            raise FileNotFoundError(f"No .npy files found in {volume_dir}")
        K = len(y_dirs)
        if len(paths) != K:
            raise ValueError(
                f"Found {len(paths)} .npy files in {volume_dir} "
                f"but y_dirs has K={K} directions."
            )

        self.volumes: list[torch.Tensor] = []
        for p in paths:
            vol = np.load(p).astype(np.float32)
            if normalize == 'volume':
                vol = (vol - vol.mean()) / (vol.std() + 1e-8)
            self.volumes.append(torch.from_numpy(vol))  # (X, Y, Z)

        self.K = K
        P = patch_size
        patch_shape = (P, P, P)

        # ── Precompute Fourier masks for patch_size ───────────────────────
        self.masks: list[torch.Tensor] = [
            torch.from_numpy(
                missing_wedge_mask_3d(y, alpha_deg, patch_shape, goniometer_axis)
            )  # (P, P, P) bool
            for y in y_dirs
        ]

        # ── Precompute conditioning embeddings ────────────────────────────
        # Embed the full y_dir unit vector (3 components) so that each wedge
        # orientation gets a unique conditioning — unlike the scalar missing_arc
        # which collapses directions with the same arc length.
        self.cond_embs: list[torch.Tensor] = [
            y_dir_embedding(y, dim=conditioning_dim)  # (1, 1, dim)
            for y in y_dirs
        ]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def _random_patch(self, vol: torch.Tensor) -> torch.Tensor:
        """Extract a random cubic patch from (X, Y, Z) volume → (1, P, P, P)."""
        P = self.patch_size
        X, Y, Z = vol.shape
        # Random top-left corner, clamped so patch stays inside volume
        x0 = int(torch.randint(0, max(1, X - P + 1), (1,)))
        y0 = int(torch.randint(0, max(1, Y - P + 1), (1,)))
        z0 = int(torch.randint(0, max(1, Z - P + 1), (1,)))
        patch = vol[x0:x0 + P, y0:y0 + P, z0:z0 + P]
        # Pad to (P, P, P) if volume is smaller than patch_size along any axis
        if patch.shape != (P, P, P):
            padded = torch.zeros(P, P, P, dtype=vol.dtype)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded
        return patch.unsqueeze(0)  # (1, P, P, P)

    def __getitem__(self, idx: int):
        # 1. Random source volume k_data → random patch
        k_data = int(torch.randint(0, self.K, (1,)))
        patch = self._random_patch(self.volumes[k_data])   # (1, P, P, P)

        # 2. Random SO(3) rotation
        R = _random_rotation_matrix()
        rotated = _rotate_volume(patch, R)                 # (1, P, P, P)

        # 3. Random wedge from K-distribution (independent of k_data)
        k_wedge = int(torch.randint(0, self.K, (1,)))
        mask = self.masks[k_wedge]                         # (P, P, P) bool

        # 4. Carve wedge in Fourier space
        carved = _apply_fourier_mask(rotated, mask)        # (1, P, P, P)

        # 5. Conditioning embedding
        cond = self.cond_embs[k_wedge]                    # (1, 1, dim)

        # 6. Valid missing region for Fourier loss: the carved frequencies
        valid_missing = ~mask                              # (P, P, P) bool

        return carved, rotated, cond, valid_missing
