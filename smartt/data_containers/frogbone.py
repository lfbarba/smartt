"""Frogbone dataset — single-mount SAXS-TT scan."""
from __future__ import annotations

from pathlib import Path

from .base import SmarttDataContainer


class FrogboneDataContainer(SmarttDataContainer):
    """Single-mount frogbone dataset.

    No remount or combined DC — this is the simplest case: one HDF5 file,
    load and go.
    """

    name = "frogbone"
    has_remount = False
    has_combined = False

    _PATH_DATA = Path("/myhome/data/smartt/shared/frogbone/dataset_qbin_0009.h5")
    _CACHE_DIR = Path("/myhome/data/smartt/shared/results/frogbone_benchmark")

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        dc = DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)
        dc.geometry.full_circle_covered = False
        return dc
