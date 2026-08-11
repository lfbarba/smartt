"""Shared base for single-mount datasets stored as one h5 file per integer
q index (``{_FILE_PREFIX}{idx}.h5``, no zero-padding, sparse indices), each
file carrying its own ``q`` value under a top-level ``q`` key.

Used by :class:`~smartt.data_containers.c4.C4DataContainer` and the
near-identical ``c5`` dataset — factored out here (rather than copy-pasted)
so a new subclass only needs to set a handful of class attributes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import SmarttDataContainer


class QIndexedDataContainer(SmarttDataContainer):
    """Single-mount dataset parametrized by an integer q index.

    Subclasses set: ``_DATA_DIR``, ``_CACHE_DIR_ROOT``, ``_FILE_PREFIX``
    (files are ``{_FILE_PREFIX}{idx}.h5``), ``_DEFAULT_Q``, ``_NAME_PREFIX``
    (used to build ``self.name``).
    """

    has_remount = False
    has_combined = False

    _DATA_DIR: Path
    _CACHE_DIR_ROOT: Path
    _FILE_PREFIX: str
    _DEFAULT_Q: int
    _NAME_PREFIX: str

    def __init__(self, q: Optional[int] = None):
        q = int(q if q is not None else self._DEFAULT_Q)
        path = self._DATA_DIR / f"{self._FILE_PREFIX}{q}.h5"
        if not path.exists():
            raise FileNotFoundError(
                f"No file for q={q} in {self._DATA_DIR}. Available q indices: {self.list_qshells()}"
            )
        self.q = q
        self._PATH_DATA = path
        self._CACHE_DIR = self._CACHE_DIR_ROOT / f"q_{q}"
        self.name = f"{self._NAME_PREFIX}-q{q}"

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        return DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)

    def get_q_value(self) -> float:
        """Scalar q (Å⁻¹) for this instance.

        The raw file stores a ``(low, center, high)`` triplet with all 3
        entries identical for these datasets, so any element works.
        """
        import h5py
        with h5py.File(self._PATH_DATA, "r") as f:
            return float(np.asarray(f["q"]).ravel()[0])

    @classmethod
    def list_qshells(cls) -> List[int]:
        """All available q indices, sorted ascending."""
        n = len(cls._FILE_PREFIX)
        return sorted(
            int(p.stem[n:]) for p in cls._DATA_DIR.glob(f"{cls._FILE_PREFIX}*.h5")
        )

    @classmethod
    def q_values(cls) -> Dict[int, float]:
        """``{q_index: q}`` for every available file (reads each header only)."""
        import h5py
        out = {}
        for q in cls.list_qshells():
            path = cls._DATA_DIR / f"{cls._FILE_PREFIX}{q}.h5"
            with h5py.File(path, "r") as f:
                out[q] = float(np.asarray(f["q"]).ravel()[0])
        return out
