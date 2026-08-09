"""C5 dataset — single-mount, q-resolved SAXS-TT scan.

Same multi-file pattern as :class:`~smartt.data_containers.c4.C4DataContainer`
(near-identical acquisition: 16 detector segments, ``q`` triplet + raw
per-projection groups in the same layout) -- each q-bin saved as its own file
``c5_q_{idx}.h5`` under ``_DATA_DIR``, sparse indices, no zero-padding.

71 of the originally acquired q-indices (4-9, 35-99) failed to download from
the source and are absent from the archive -- ``list_qshells()`` reflects
only what's actually on disk (123 of the nominal 194).
"""
from __future__ import annotations

from pathlib import Path

from .qindexed_base import QIndexedDataContainer


class C5DataContainer(QIndexedDataContainer):
    """Single-mount C5 dataset, parametrized by q index.

    No remount or combined DC.

    >>> ds = C5DataContainer()          # default q index (15)
    >>> ds = C5DataContainer(q=100)     # c5_q_100.h5
    >>> C5DataContainer.list_qshells()  # all available q indices (123)
    """

    _DATA_DIR       = Path("/myhome/data/smartt/shared/c5")
    _CACHE_DIR_ROOT = Path("/myhome/data/smartt/shared/results/c5_benchmark")
    _FILE_PREFIX    = "c5_q_"
    _DEFAULT_Q      = 15
    _NAME_PREFIX    = "c5"
