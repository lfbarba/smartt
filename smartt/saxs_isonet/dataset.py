"""Training dataset for the SAXS-TT missing-wedge correction pipeline.

Lightweight: loads K spherically-masked volumes (normalized) into memory, and
on each call returns a full (cube_size, cube_size, cube_size) volume plus two
indices. Rotation and Fourier carving happen on the GPU in VolumeAugmentor.

``__getitem__`` → ``(v0, v1, k_src, k_wedge)``:
  - ``v0``     : (1, C, C, C) normalized input volume for k_src (this round).
  - ``v1``     : (1, C, C, C) frozen target volume at k_src (round_00 in
                  dual-source mode; equals v0 when ``target_dir`` is None).
  - ``k_src``  : source RSM index (its measured mask is excluded from the loss).
  - ``k_wedge``: RSM index whose canonical wedge is carved. Drawn only from
                  directions with missing arc ≥ ``min_wedge_deg``.

Design choices baked in here:
  - **Spherical masking**: volumes are pre-cropped to C×C×C and masked to the
    inscribed sphere (exterior = 0 in normalised space). Rotation corner
    artefacts are avoided without a √3-oversampled load buffer.
  - **Full-volume training**: each dataset item is the complete volume for a
    randomly chosen eligible k_src; no subvolume windows are drawn.
  - **Fixed target** (dual-source): input changes each round, target = round_00.
  - **Per-volume normalization fixed to round_00 stats**: freezing the scale
    keeps input and target on identical, stable scales across rounds.
  - **Interior-only norm stats**: mean/std computed from sphere-interior voxels
    only, so the exterior zeros do not bias normalization.

Public API
----------
save_reconstruction_volumes(reconstruction, output_dir)
compute_norm_stats(volume_dir, mask=None)
MissingWedgeSAXS(Dataset)
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from smartt.saxs_isonet.preprocess import make_sphere_mask
from smartt.saxs_isonet.wedge import _unit, all_missing_arcs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_reconstruction_volumes(
    reconstruction: torch.Tensor,
    output_dir: str | Path,
) -> list[Path]:
    """Save the (K, X, Y, Z) reconstruction tensor as K numbered .npy files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = reconstruction.detach().cpu().numpy() if isinstance(reconstruction, torch.Tensor) \
          else np.asarray(reconstruction)
    paths = []
    for k in range(rec.shape[0]):
        p = out / f'vol_{k:04d}.npy'
        np.save(p, rec[k].astype(np.float32))
        paths.append(p)
    return paths


def compute_norm_stats(
    volume_dir: str | Path,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-volume (mean, std) for all vol_*.npy in a directory → (K, 2) float64.

    mask : (X, Y, Z) bool — if provided, statistics are computed from the True
           voxels only. Pass the sphere mask to exclude exterior zeros from the
           normalization statistics.
    """
    paths = sorted(Path(volume_dir).glob('vol_*.npy'))
    if not paths:
        raise FileNotFoundError(f"No vol_*.npy files found in {volume_dir}")
    stats = []
    for p in paths:
        v = np.load(p).astype(np.float64)
        interior = v[mask] if mask is not None else v.ravel()
        stats.append((interior.mean(), interior.std() + 1e-8))
    return np.array(stats, dtype=np.float64)   # (K, 2)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MissingWedgeSAXS(Dataset):
    """Self-supervised dataset for SAXS-TT missing-wedge correction.

    Parameters
    ----------
    volume_dir        : directory with K input vol_*.npy (this round's volumes).
    rsm_dirs          : (K, 3) RSM directions.
    alpha_deg         : half-angle (°) of the unmeasurable goniometer polar caps.
    goniometer_axis   : tilt-axis unit vector (defaults to [0,0,1]).
    cube_size         : volume side length C; volumes must be (C, C, C) cubes.
    n_samples         : virtual dataset length (items drawn per epoch).
    min_wedge_deg     : minimum missing arc (°) for a direction to be eligible
                        as k_wedge.
    max_rsm_wedge_deg : maximum missing arc (°) for an RSM location to be used
                        as k_src. None = all K volumes eligible. Raises
                        ValueError if the filter leaves no eligible volumes.
    target_dir        : directory with K frozen target vol_*.npy (round_00).
                        None = single-source (target == input).
    norm_stats        : (K, 2) per-volume (mean, std). Computed from target
                        volumes if None.
    """

    def __init__(
        self,
        volume_dir: str | Path,
        rsm_dirs: np.ndarray,
        alpha_deg: float,
        goniometer_axis: Optional[np.ndarray] = None,
        cube_size: int = 64,
        n_samples: int = 2000,
        min_wedge_deg: float = 10.0,
        max_rsm_wedge_deg: Optional[float] = None,
        target_dir: Optional[str | Path] = None,
        norm_stats: Optional[np.ndarray] = None,
    ) -> None:
        self.cube_size = cube_size
        self.n_samples = n_samples
        self.rsm_dirs  = np.asarray(rsm_dirs, dtype=float)
        self.alpha_deg = float(alpha_deg)

        if goniometer_axis is None:
            goniometer_axis = np.array([0., 0., 1.])
        self.goniometer_axis = _unit(np.asarray(goniometer_axis, dtype=float))

        K = len(self.rsm_dirs)
        self.K = K

        self.sphere_mask = make_sphere_mask(cube_size)   # (C, C, C) bool

        v0_paths = self._resolve(volume_dir, K, 'volume_dir')
        v1_paths = (self._resolve(target_dir, K, 'target_dir')
                    if target_dir is not None else v0_paths)
        self.dual_source = target_dir is not None

        if norm_stats is None:
            norm_stats = compute_norm_stats(
                Path(v1_paths[0]).parent, mask=self.sphere_mask,
            )
        self.norm_stats = np.asarray(norm_stats, dtype=np.float64)
        if self.norm_stats.shape != (K, 2):
            raise ValueError(f"norm_stats must be (K,2)=({K},2), got {self.norm_stats.shape}")

        self.v0 = [self._load_norm(v0_paths[k], k) for k in range(K)]
        self.v1 = (self.v0 if not self.dual_source
                   else [self._load_norm(v1_paths[k], k) for k in range(K)])

        arcs_deg = np.degrees(all_missing_arcs(self.rsm_dirs, alpha_deg, self.goniometer_axis))

        # k_wedge pool: directions with a large-enough missing wedge to carve.
        carve_pool = np.where(arcs_deg >= min_wedge_deg)[0]
        if carve_pool.size == 0:
            logger.warning("No direction with missing arc ≥ %.1f°; using all K.", min_wedge_deg)
            carve_pool = np.arange(K)
        self.wedge_candidates = carve_pool.astype(np.int64)

        # k_src pool: RSM locations with a small-enough wedge to learn from.
        if max_rsm_wedge_deg is not None:
            src_pool = np.where(arcs_deg <= max_rsm_wedge_deg)[0]
            if src_pool.size == 0:
                raise ValueError(
                    f"No RSM location has missing arc ≤ {max_rsm_wedge_deg}°. "
                    "Loosen max_rsm_wedge_deg or leave it unset to use all K volumes."
                )
            self.src_candidates = src_pool.astype(np.int64)
        else:
            self.src_candidates = np.arange(K, dtype=np.int64)

        logger.info(
            "Dataset: K=%d, cube=%d, dual_source=%s, %d carve dirs, %d src dirs.",
            K, cube_size, self.dual_source,
            len(self.wedge_candidates), len(self.src_candidates),
        )

    @staticmethod
    def _resolve(d: str | Path, K: int, name: str) -> list[Path]:
        paths = sorted(Path(d).glob('vol_*.npy'))
        if not paths:
            raise FileNotFoundError(f"No vol_*.npy files found in {name}={d}")
        if len(paths) != K:
            raise ValueError(f"{name}={d}: found {len(paths)} files, expected K={K}")
        return paths

    def _load_norm(self, path: Path, k: int) -> torch.Tensor:
        mean, std = self.norm_stats[k]
        vol = (np.load(path).astype(np.float32) - mean) / std
        vol[~self.sphere_mask] = 0.0
        return torch.from_numpy(np.ascontiguousarray(vol))   # (C, C, C)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        k_src = int(self.src_candidates[
            torch.randint(0, len(self.src_candidates), (1,))
        ])
        v0 = self.v0[k_src].unsqueeze(0)   # (1, C, C, C)
        v1 = self.v1[k_src].unsqueeze(0)   # (1, C, C, C)
        k_wedge = int(self.wedge_candidates[
            torch.randint(0, len(self.wedge_candidates), (1,))
        ])
        return v0, v1, k_src, k_wedge
