"""Zenodo missing-wedge dataset (two-mount SAXS-TT experiment)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .base import SmarttDataContainer


class ZenodoDataContainer(SmarttDataContainer):
    """Missing-wedge dataset from https://zenodo.org/records/10995088.

    Two datasets (data_set_1.h5, data_set_2.h5) are padded to a common
    detector shape (65 × 66) and combined via full_geometry.geo.

    Padding convention: pad **at the start** ``((j_pad, 0), (k_pad, 0))``,
    matching the original authors' pre-processing.  No intensity scaling is
    applied (unlike the b411 remount which uses 0.75).
    """

    name = "zenodo"
    has_remount = True
    has_combined = True

    _DATA_DIR    = Path("/myhome/data/smartt/shared/missing_wedge/full_data_zenodo/data")
    _PATH_DS1    = _DATA_DIR / "data_set_1.h5"
    _PATH_DS2    = _DATA_DIR / "data_set_2.h5"
    _PATH_GEO    = _DATA_DIR / "full_geometry.geo"
    _CACHE_DIR   = Path("/myhome/data/smartt/shared/results/zenodo_benchmark")

    _TARGET_J = 65
    _TARGET_K = 66

    # ------------------------------------------------------------------

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def _pad_dc_projections(self, dc) -> None:
        """Pad projections in-place to (_TARGET_J, _TARGET_K), at the start."""
        for frame in dc.projections:
            j_pad = self._TARGET_J - frame.data.shape[0]
            k_pad = self._TARGET_K - frame.data.shape[1]
            frame.diode   = np.pad(frame.diode,   ((j_pad, 0), (k_pad, 0)))
            frame.data    = np.pad(frame.data,    ((j_pad, 0), (k_pad, 0), (0, 0)))
            frame.weights = np.pad(frame.weights, ((j_pad, 0), (k_pad, 0), (0, 0)))

    def _n_ds1(self) -> int:
        """Number of projections in data_set_1 (fast header-only load)."""
        from mumott.data_handling import DataContainer
        return len(DataContainer(str(self._PATH_DS1)).projections)

    def get_combined_dc(self):
        from mumott.data_handling import DataContainer

        combined = DataContainer(str(self._PATH_DS1))
        self._pad_dc_projections(combined)

        dc2 = DataContainer(str(self._PATH_DS2))
        self._pad_dc_projections(dc2)

        n2 = len(dc2.projections)
        for _ in range(n2):
            frame = dc2.projections[0]
            del dc2.projections[0]
            combined.projections.append(frame)

        combined.geometry.read(str(self._PATH_GEO))
        return combined

    def get_main_dc(self):
        n = self._n_ds1()
        combined = self.get_combined_dc()
        sub = copy.deepcopy(combined)
        for i in sorted(range(n, len(combined.projections)), reverse=True):
            del sub.projections[i]
        return sub

    def get_remount_dc(self):
        n = self._n_ds1()
        combined = self.get_combined_dc()
        sub = copy.deepcopy(combined)
        for i in sorted(range(n), reverse=True):
            del sub.projections[i]
        return sub
