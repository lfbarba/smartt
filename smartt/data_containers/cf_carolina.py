"""CF Carolina dataset — single-mount, q-resolved SAXS-TT scan.

The sample was azimuthally integrated into many q-bins, each saved as its
own file: ``dataset_qbin_{idx:03d}_{q_low}_{q_high}.h5`` under
``_DATA_DIR``. All q-bins share identical acquisition geometry (235
projections, ``volume_shape=[85,110,85]``) and differ only in scattering
intensity — each file is self-describing via its ``q_bin``/``q_bin_index``
fields. ``full_circle_covered`` is correctly auto-detected as ``False`` on
load (like cf-peek/auditory-ossicle), no geometry patching required.

Pass ``qbin`` to select which file to load; ``_PATH_DATA``/``_CACHE_DIR``
are resolved per-instance so different q-bins never share a cache.
"""
from __future__ import annotations

from pathlib import Path

from .base import SmarttDataContainer


class CfCarolinaDataContainer(SmarttDataContainer):
    """Single-mount CF Carolina dataset, parametrized by q-bin.

    No remount or combined DC.

    >>> ds = CfCarolinaDataContainer(qbin=70)   # dataset_qbin_070_*.h5
    >>> ds = CfCarolinaDataContainer()          # default qbin
    """

    has_remount = False
    has_combined = False

    _DATA_DIR       = Path("/myhome/data/smartt/shared/cf_carolina")
    _CACHE_DIR_ROOT = Path("/myhome/data/smartt/shared/results/cf_carolina_benchmark")

    _DEFAULT_QBIN = 71

    def __init__(self, qbin: int = _DEFAULT_QBIN):
        qbin = int(qbin)
        matches = sorted(self._DATA_DIR.glob(f"dataset_qbin_{qbin:03d}_*.h5"))
        if not matches:
            available = sorted(
                int(p.name.split("_")[2]) for p in self._DATA_DIR.glob("dataset_qbin_*.h5")
            )
            raise FileNotFoundError(
                f"No file for qbin={qbin} in {self._DATA_DIR}. Available qbins: {available}"
            )
        self.qbin       = qbin
        self._PATH_DATA = matches[0]
        self._CACHE_DIR = self._CACHE_DIR_ROOT / f"qbin_{qbin:03d}"
        self.name        = f"cf-carolina-qbin{qbin:03d}"

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        return DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)
