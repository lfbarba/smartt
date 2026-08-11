"""PX chameleon dataset — single-mount, q-resolved SAXS-TT scan.

Same multi-file pattern as :class:`~smartt.data_containers.cf_carolina.CfCarolinaDataContainer`
/ :class:`~smartt.data_containers.c4.C4DataContainer`: the sample is
azimuthally integrated into many q-bins, each saved as its own file
``px_chameleon_q_{idx}.h5`` under ``_DATA_DIR`` (no zero-padding, indices
almost contiguous 0-154, only 98 and 99 missing).

``full_circle_covered`` is correctly auto-detected as ``False`` on load, no
geometry patching required.
"""
from __future__ import annotations

from pathlib import Path

from .base import SmarttDataContainer


class PxChameleonDataContainer(SmarttDataContainer):
    """Single-mount PX chameleon dataset, parametrized by q index.

    No remount or combined DC.

    >>> ds = PxChameleonDataContainer()          # default q index (15)
    >>> ds = PxChameleonDataContainer(q=100)      # px_chameleon_q_100.h5
    """

    has_remount = False
    has_combined = False

    _DATA_DIR       = Path("/myhome/data/smartt/shared/px")
    _CACHE_DIR_ROOT = Path("/myhome/data/smartt/shared/results/px_chameleon_benchmark")

    _DEFAULT_Q = 0

    def __init__(self, q: int = _DEFAULT_Q):
        q = int(q)
        path = self._DATA_DIR / f"px_chameleon_q_{q}.h5"
        if not path.exists():
            available = sorted(
                int(p.stem.rsplit("_", 1)[-1]) for p in self._DATA_DIR.glob("px_chameleon_q_*.h5")
            )
            raise FileNotFoundError(
                f"No file for q={q} in {self._DATA_DIR}. Available q indices: {available}"
            )
        self.q          = q
        self._PATH_DATA = path
        self._CACHE_DIR = self._CACHE_DIR_ROOT / f"q_{q}"
        self.name        = f"px-chameleon-q{q}"

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        return DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)
