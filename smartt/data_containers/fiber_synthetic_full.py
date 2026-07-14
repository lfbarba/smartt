"""Fiber-synthetic (full-sample) dataset — no missing wedge.

Identical in construction to ``fiber-synthetic`` (see
``smartt/data_containers/fiber_synthetic.py`` and
``notebooks/SyntheticDataContainers.ipynb``): SH coefficients are assigned
voxel-wise from a composite fiber volume and forward-projected to produce the
projections/geometry saved here, with the ground truth known exactly by
construction and cached alongside the HDF5 file as an ``.npy`` + JSON sidecar.

The *only* difference is angular coverage: this dataset is full-sample — its
projection directions span the whole orientation space, with no missing-wedge
gap — so it serves as the no-missing-wedge counterpart to ``fiber-synthetic``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SmarttDataContainer


class FiberSyntheticFullDataContainer(SmarttDataContainer):
    """Single-mount synthetic dataset (full sample, no missing wedge) with a known SH ground truth.

    No remount or combined DC — one HDF5 file, load and go.  As with
    ``fiber-synthetic``, the saved geometry has ``full_circle_covered=True``
    baked in even though the detector segments span only 180°, so it must be
    overridden on load.  (That flag concerns the detector's azimuthal segment
    coverage, not the sample-tilt coverage, so the override is unrelated to this
    dataset being full-sample.)
    """

    name = "fiber-synthetic-full"
    has_remount = False
    has_combined = False

    _PATH_DATA = Path("/myhome/data/smartt/shared/synthetic_no_missing/dataset_composite_synthetic.h5")
    _CACHE_DIR = Path("/myhome/data/smartt/shared/synthetic_no_missing/recon_cache")

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        dc = DataContainer(str(self._PATH_DATA))
        dc.geometry.full_circle_covered = False
        return dc

    def get_ground_truth(self) -> np.ndarray:
        """Load the exact ``(X, Y, Z, 45)`` SH tensor used to generate this dataset.

        Reads the ``ground_truth_*.npy`` sidecar written by
        ``notebooks/SyntheticDataContainers.ipynb`` via ``save_recon`` — no
        reconstruction needed, this dataset's ground truth is exact by
        construction. If multiple sidecars exist (e.g. from re-runs with
        different generation params), the most recently modified one is used.
        """
        matches = sorted(self._CACHE_DIR.glob("ground_truth_*.npy"), key=lambda p: p.stat().st_mtime)
        if not matches:
            raise FileNotFoundError(
                f"No ground-truth sidecar found in {self._CACHE_DIR}. "
                "Run notebooks/SyntheticDataContainers.ipynb to generate it."
            )
        return np.load(matches[-1])
