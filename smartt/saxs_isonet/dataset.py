"""Training dataset for the SAXS-TT missing-wedge correction pipeline.

The dataset is deliberately lightweight: it only loads the K reconstructed
volumes and, on each call, returns a **raw** cubic patch plus two indices.
The expensive SO(3) rotation and Fourier carving happen on the GPU in
:class:`smartt.saxs_isonet.augment.VolumeAugmentor` (CPU rotation is the
training bottleneck).

``__getitem__`` returns ``(patch, k_src, k_wedge)``:
  - ``patch``   : (1, P, P, P) raw patch drawn from RSM volume ``k_src``.
  - ``k_src``   : index of the source RSM volume (its measured-frequency mask
                  is excluded from the loss after rotation).
  - ``k_wedge`` : index of the RSM direction whose canonical wedge is carved.

``k_src`` and ``k_wedge`` are sampled independently.  ``k_wedge`` is restricted
to directions with a missing arc ≥ ``min_wedge_deg`` — tiny near-pole wedges
carry almost no learning signal and would waste compute.  ``k_src`` is
unrestricted: near-pole volumes (almost complete ground truth) are the most
valuable sources to carve from.

Public API
----------
save_reconstruction_volumes(reconstruction, output_dir)
MissingWedgeSAXS(Dataset)
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from smartt.saxs_isonet.wedge import _unit, all_missing_arcs

logger = logging.getLogger(__name__)


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
# Dataset
# ---------------------------------------------------------------------------

class MissingWedgeSAXS(Dataset):
    """Self-supervised dataset for SAXS-TT missing-wedge correction.

    Loads K .npy volumes from ``volume_dir`` (sorted by name, matching the order
    of ``rsm_dirs``) and returns raw patches + indices.  All augmentation is
    delegated to :class:`VolumeAugmentor` on the training device.

    Parameters
    ----------
    volume_dir : directory containing K .npy volume files (float32, (X,Y,Z)).
    rsm_dirs   : (K, 3) RSM directions (from fibonacci_hemisphere).
    alpha_deg  : half-angle (°) of the unmeasurable goniometer polar caps.
    goniometer_axis : tilt-axis unit vector (defaults to [0,0,1]).
    patch_size : spatial side-length (voxels) of cubic training patches.
    n_samples  : virtual dataset length (patches drawn per epoch).
    normalize  : ``'volume'`` (per-volume zero-mean unit-variance) or ``'none'``.
    min_wedge_deg : minimum missing arc (degrees) for a direction to be eligible
        as ``k_wedge``.  Directions below this carve almost nothing and are
        skipped.  Set 0 to allow all directions.
    """

    def __init__(
        self,
        volume_dir: str | Path,
        rsm_dirs: np.ndarray,
        alpha_deg: float,
        goniometer_axis: Optional[np.ndarray] = None,
        patch_size: int = 64,
        n_samples: int = 2000,
        normalize: str = 'volume',
        min_wedge_deg: float = 10.0,
    ) -> None:
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.rsm_dirs = np.asarray(rsm_dirs, dtype=float)
        self.alpha_deg = float(alpha_deg)

        if goniometer_axis is None:
            goniometer_axis = np.array([0., 0., 1.])
        self.goniometer_axis = _unit(np.asarray(goniometer_axis, dtype=float))

        # ── Load volumes ──────────────────────────────────────────────────
        volume_dir = Path(volume_dir)
        paths = sorted(volume_dir.glob('vol_*.npy'))
        if not paths:
            raise FileNotFoundError(f"No vol_*.npy files found in {volume_dir}")
        K = len(self.rsm_dirs)
        if len(paths) != K:
            raise ValueError(
                f"Found {len(paths)} .npy files in {volume_dir} "
                f"but rsm_dirs has K={K} directions."
            )

        self.volumes: list[torch.Tensor] = []
        for p in paths:
            vol = np.load(p).astype(np.float32)
            if normalize == 'volume':
                vol = (vol - vol.mean()) / (vol.std() + 1e-8)
            self.volumes.append(torch.from_numpy(vol))  # (X, Y, Z)

        self.K = K

        # ── Eligible carve directions (skip near-zero wedges) ─────────────
        arcs_deg = np.degrees(all_missing_arcs(self.rsm_dirs, alpha_deg, self.goniometer_axis))
        candidates = np.where(arcs_deg >= min_wedge_deg)[0]
        if candidates.size == 0:
            logger.warning(
                "No RSM direction has a missing arc ≥ %.1f°; using all K directions "
                "as carve candidates.", min_wedge_deg,
            )
            candidates = np.arange(K)
        self.wedge_candidates = candidates.astype(np.int64)
        logger.info(
            "Dataset: K=%d volumes, %d eligible carve directions (min_wedge_deg=%.1f°).",
            K, len(self.wedge_candidates), min_wedge_deg,
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def _random_patch(self, vol: torch.Tensor) -> torch.Tensor:
        """Extract a random cubic patch from (X, Y, Z) volume → (1, P, P, P)."""
        P = self.patch_size
        X, Y, Z = vol.shape
        x0 = int(torch.randint(0, max(1, X - P + 1), (1,)))
        y0 = int(torch.randint(0, max(1, Y - P + 1), (1,)))
        z0 = int(torch.randint(0, max(1, Z - P + 1), (1,)))
        patch = vol[x0:x0 + P, y0:y0 + P, z0:z0 + P]
        if patch.shape != (P, P, P):     # pad if a volume axis is smaller than P
            padded = torch.zeros(P, P, P, dtype=vol.dtype)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded
        return patch.unsqueeze(0)        # (1, P, P, P)

    def __getitem__(self, idx: int):
        k_src = int(torch.randint(0, self.K, (1,)))
        patch = self._random_patch(self.volumes[k_src])      # (1, P, P, P)
        k_wedge = int(self.wedge_candidates[
            torch.randint(0, len(self.wedge_candidates), (1,))
        ])
        return patch, k_src, k_wedge
