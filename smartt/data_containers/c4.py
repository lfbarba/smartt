"""C4 dataset — single-mount, q-resolved SAXS-TT scan.

Same multi-file pattern as :class:`~smartt.data_containers.cf_carolina.CfCarolinaDataContainer`:
the sample is azimuthally integrated into many q-bins, each saved as its own
file ``c4_q_{idx}.h5`` under ``_DATA_DIR`` (no zero-padding, and indices are
sparse — not every integer in ``[1, 194]`` has a file). Each file stores its
own ``q`` triplet internally, but the filename index is sufficient to select
the file directly.

Unlike cf-carolina/plastic-plasmonics, loading requires
``nonfinite_replacement_value=0`` — the raw data contains NaNs that mumott
rejects by default. ``full_circle_covered`` is correctly auto-detected as
``False`` on load, no geometry patching required.
"""
from __future__ import annotations

from pathlib import Path

from .qindexed_base import QIndexedDataContainer


class C4DataContainer(QIndexedDataContainer):
    """Single-mount C4 dataset, parametrized by q index.

    No remount or combined DC.

    >>> ds = C4DataContainer()          # default q index (15)
    >>> ds = C4DataContainer(q=100)     # c4_q_100.h5
    >>> C4DataContainer.list_qshells()  # all 129 available q indices
    """

    _DATA_DIR       = Path("/myhome/data/smartt/shared/c4")
    _CACHE_DIR_ROOT = Path("/myhome/data/smartt/shared/results/c4_benchmark")
    _FILE_PREFIX    = "c4_q_"
    _DEFAULT_Q      = 15
    _NAME_PREFIX    = "c4"
