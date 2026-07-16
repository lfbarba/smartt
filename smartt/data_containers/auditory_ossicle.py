"""Auditory ossicle dataset — single-mount SAXS-TT scan, migrated from mumott 0.2.

Original file (``dataset_q_0.053_0.056.h5``) was recorded and pre-processed
with mumott 0.2's raw ``DataContainer`` + ``DataContainer.transform(...)``
workflow, which no longer exists in current mumott. It was migrated to the
current per-projection ``rotation_matrix`` schema with
``scripts/migrate_legacy_mumott_h5.py`` using the ``TransformParameters``
recovered from a collaborator's old notebook screenshot::

    data_sorting=(0, 1, 2), data_index_origin=(0, 0),
    principal_rotation_right_handed=True, secondary_rotation_right_handed=True,
    detector_angle_0=(0, 1), detector_angle_right_handed=False,
    offset_positive=(True, True)

Geometry: rotation (principal axis) spans the full circle; tilt (secondary
axis) is one-sided, sampled over ``[0, 45]`` degrees in 7 levels — the
missing wedge covers the remaining ``(45, 180)`` degree tilt range that a
second mount would normally fill. 8 detector segments, 306 projections
total. No embedded ground truth (real experimental data).
"""
from __future__ import annotations

from pathlib import Path

from .base import SmarttDataContainer


class AuditoryOssicleDataContainer(SmarttDataContainer):
    """Single-mount auditory ossicle dataset.

    No remount or combined DC. ``full_circle_covered`` is correctly
    auto-detected as ``False`` on load (like cf-peek), so no geometry
    patching is required here.
    """

    name = "auditory-ossicle"
    has_remount = False
    has_combined = False

    _PATH_DATA = Path(
        "/myhome/data/smartt/shared/auditory_ossicle/dataset_q_0.053_0.056_migrated.h5"
    )
    _CACHE_DIR = Path("/myhome/data/smartt/shared/results/auditory_ossicle_benchmark")

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        dc = DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)
        return dc
