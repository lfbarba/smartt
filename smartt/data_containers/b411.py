"""B411 remounting experiment — bone sample, two mounts."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .base import SmarttDataContainer


class B411DataContainer(SmarttDataContainer):
    """Two-mount remounting experiment on a bone sample (b411R).

    The main and remount DCs are derived from the combined DC so that all
    reconstructions share the combined (141 × 111 × 141) coordinate frame.
    This makes voxel-level comparison with the ground truth meaningful.

    Combined DC recipe:
      1. Load b411R projections and pad detector width by (27, 28) pixels.
      2. Append remount projections (scaled by 0.75 as in the original notebook).
      3. Load combined_geometry.h5 to set the shared volume grid.

    Main / remount are derived by index-splitting the combined DC.
    """

    name = "b411"
    has_remount = True
    has_combined = True

    _DATA_DIR    = Path("/myhome/data/smartt/shared/b411")
    _PATH_MAIN   = _DATA_DIR / "dataset_b411R_inf_1_0.220_1.900.h5"
    _PATH_REMOUNT = _DATA_DIR / "dataset_b411R_inf1_remount_0.220_1.900.h5"
    _PATH_GEO    = _DATA_DIR / "combined_geometry.h5"
    _CACHE_DIR   = Path("/myhome/data/smartt/shared/results/b411_benchmark")

    # Number of b411R (first-mount) projections — fixed for this dataset.
    _N_MAIN = 266

    # ------------------------------------------------------------------

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_combined_dc(self):
        from mumott.data_handling import DataContainer

        combined = DataContainer(str(self._PATH_MAIN))

        for proj in combined.projections:
            proj.diode   = np.pad(proj.diode,   ((0, 0), (27, 28)),         constant_values=1)
            proj.data    = np.pad(proj.data,    ((0, 0), (27, 28), (0, 0)), constant_values=0)
            proj.weights = np.pad(proj.weights, ((0, 0), (27, 28), (0, 0)), constant_values=0)

        dc_rem = DataContainer(str(self._PATH_REMOUNT))
        n_rem = len(dc_rem.projections)
        for _ in range(n_rem):
            dc_rem.projections[0].data *= 0.75
            frame = dc_rem.projections[0]
            del dc_rem.projections[0]
            combined.projections.append(frame)

        combined.geometry.read(str(self._PATH_GEO))
        
        combined.geometry.full_circle_covered = False
        return combined

    def get_main_dc(self):
        combined = self.get_combined_dc()
        sub = copy.deepcopy(combined)
        for i in sorted(range(self._N_MAIN, len(combined.projections)), reverse=True):
            del sub.projections[i]
        sub.geometry.full_circle_covered = False
        return sub

    def get_remount_dc(self):
        combined = self.get_combined_dc()
        sub = copy.deepcopy(combined)
        for i in sorted(range(self._N_MAIN), reverse=True):
            del sub.projections[i]
        sub.geometry.full_circle_covered = False
        return sub
