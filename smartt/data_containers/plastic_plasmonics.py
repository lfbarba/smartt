"""Plastic plasmonics dataset — single-mount, q-resolved SAXS-TT scan.

Same multi-file pattern as :class:`~smartt.data_containers.cf_carolina.CfCarolinaDataContainer`:
the sample is azimuthally integrated into many q-windows, each saved as its
own file under ``_DATA_DIR``. Unlike cf-carolina, filenames encode the
q-window directly as floats (``dataset_q_{q_low}_{q_high}.h5``) rather than
a zero-padded bin index, and there is no internal ``q_bin``/``q_bin_index``
metadata to cross-check against — so the file is selected by matching
``q_low`` against the filename (with a small numeric tolerance, since future
files may format the float with different precision).

``full_circle_covered`` is correctly auto-detected as ``False`` on load
(like cf-peek/auditory-ossicle/cf-carolina), no geometry patching required.

Only one file exists at the time of writing (``dataset_q_0.362_0.381.h5``);
more q-windows are expected later.
"""
from __future__ import annotations

import re
from pathlib import Path

from .base import SmarttDataContainer

_FILENAME_RE = re.compile(r"dataset_q_([\d.]+)_([\d.]+)\.h5$")


class PlasticPlasmonicsDataContainer(SmarttDataContainer):
    """Single-mount plastic plasmonics dataset, parametrized by q-window.

    No remount or combined DC.

    >>> ds = PlasticPlasmonicsDataContainer()             # default q-window
    >>> ds = PlasticPlasmonicsDataContainer(q=0.362)       # dataset_q_0.362_0.381.h5
    """

    has_remount = False
    has_combined = False

    _DATA_DIR       = Path("/myhome/data/smartt/shared/plastic_plasmonics")
    _CACHE_DIR_ROOT = Path("/myhome/data/smartt/shared/results/plastic_plasmonics_benchmark")

    _DEFAULT_Q = 0.362
    _TOL       = 1e-6

    def __init__(self, q: float = _DEFAULT_Q):
        q = float(q)
        match = None
        available = []
        for p in sorted(self._DATA_DIR.glob("dataset_q_*.h5")):
            m = _FILENAME_RE.search(p.name)
            if not m:
                continue
            q_low = float(m.group(1))
            available.append(q_low)
            if abs(q_low - q) < self._TOL:
                match = (p, q_low)
        if match is None:
            raise FileNotFoundError(
                f"No file for q={q} in {self._DATA_DIR}. Available q_low values: {sorted(available)}"
            )
        path, q_low = match
        self.q          = q_low
        self._PATH_DATA = path
        self._CACHE_DIR = self._CACHE_DIR_ROOT / f"q_{q_low:.3f}"
        self.name        = f"plastic-plasmonics-q{q_low:.3f}"

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        return DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)
