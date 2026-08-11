"""Frogbone dataset — single-mount, q-resolved SAXS-TT scan.

Like ``cf-carolina``/``plastic-plasmonics``, the sample was azimuthally
integrated into many q-bins, each saved as its own file
(``dataset_qbin_{idx:04d}.h5``) under ``_DATA_DIR``. All 79 q-bins
(``0000``-``0078``) share identical acquisition geometry (240 projections,
``volume_shape=[65, 82, 65]``, SAXS/flat-detector) and differ only in
scattering intensity; each file also carries its own scalar ``q`` value
(monotonically increasing, log-spaced, ``~2.7e-4`` to ``~4.9e-2``) under the
top-level ``q`` key — not exposed via mumott's ``DataContainer``/``Geometry``,
so read directly with h5py (:meth:`get_q_value`).

Pass ``qbin`` to select which file; ``_PATH_DATA``/``_CACHE_DIR`` are resolved
per-instance so different q-bins never share a cache. The registry default
(``qbin=9``) keeps the original flat, non-namespaced cache directory used by
every reconstruction cached before q-shell support was added; every other
q-bin gets its own ``qbin_XXX`` subdirectory.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import SmarttDataContainer

_N_QSHELLS = 79
_DEFAULT_QBIN = 9


class FrogboneDataContainer(SmarttDataContainer):
    """Single-mount frogbone dataset, parametrized by q-bin.

    No remount or combined DC.

    >>> ds = FrogboneDataContainer(qbin=40)   # dataset_qbin_0040.h5
    >>> ds = FrogboneDataContainer()          # default qbin (9, back-compat)
    """

    has_remount = False
    has_combined = False

    _DATA_DIR       = Path("/myhome/data/smartt/shared/frogbone")
    _CACHE_DIR_ROOT = Path("/myhome/data/smartt/shared/results/frogbone_benchmark")

    def __init__(self, qbin: int = _DEFAULT_QBIN):
        qbin = int(qbin)
        path = self._DATA_DIR / f"dataset_qbin_{qbin:04d}.h5"
        if not path.exists():
            raise FileNotFoundError(
                f"No file for qbin={qbin} in {self._DATA_DIR}. "
                f"Available qbins: 0..{_N_QSHELLS - 1}"
            )
        self.qbin       = qbin
        self._PATH_DATA = path
        # Back-compat: the historical default (qbin=9) keeps using the
        # original flat cache dir every prior frogbone reconstruction was
        # saved under; every other q-bin gets its own subdirectory so the 79
        # shells' caches never collide.
        if qbin == _DEFAULT_QBIN:
            self._CACHE_DIR = self._CACHE_DIR_ROOT
        else:
            self._CACHE_DIR = self._CACHE_DIR_ROOT / f"qbin_{qbin:04d}"
        self.name = f"frogbone-qbin{qbin:04d}" if qbin != _DEFAULT_QBIN else "frogbone"

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        dc = DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)
        dc.geometry.full_circle_covered = False
        return dc

    def get_q_value(self) -> float:
        """The scalar ``q`` value for this instance's q-bin (Å⁻¹).

        Not exposed via mumott's ``DataContainer``/``Geometry`` — read
        directly from the raw h5 file's top-level ``q`` dataset.
        """
        import h5py
        with h5py.File(self._PATH_DATA, "r") as f:
            return float(f["q"][()])

    @classmethod
    def list_qshells(cls) -> List[int]:
        """All available q-bin indices, sorted ascending."""
        return sorted(
            int(p.name.split("_")[2].split(".")[0])
            for p in cls._DATA_DIR.glob("dataset_qbin_*.h5")
        )

    @classmethod
    def q_values(cls) -> dict:
        """``{qbin: q}`` for every available q-bin (reads each file's header only)."""
        import h5py
        out = {}
        for qbin in cls.list_qshells():
            path = cls._DATA_DIR / f"dataset_qbin_{qbin:04d}.h5"
            with h5py.File(path, "r") as f:
                out[qbin] = float(f["q"][()])
        return out
