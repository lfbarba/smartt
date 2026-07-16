"""CF/PEEK dataset — single-mount experimental SAXS-TT scan.

From https://zenodo.org/records/17713339 (Liebi, Kristiansen, Grob),
repackaged for MUMOTT. Carbon-fiber-reinforced PEEK sample, no remount.

Missing-wedge geometry: rotation (``inner_angle``) about the ``-x`` axis
spans the full circle at every tilt except the 0-degree tilt level (which
only spans 180 degrees, presumably dropped by Friedel symmetry at that
tilt). Tilt (``outer_angle``) about the ``-y`` axis is one-sided, sampled
only over ``[-50, 0]`` degrees (8 levels: 0, -10, -20, -25, -30, -35, -40,
-50) — the missing wedge covers the remaining ``(-90, -50)`` and
``(0, 90)`` degree tilt range that a second mount would normally fill.
430 projections total. No embedded ground truth (real experimental data).
"""
from __future__ import annotations

from pathlib import Path

from .base import SmarttDataContainer


class CfPeekDataContainer(SmarttDataContainer):
    """Single-mount CF/PEEK dataset.

    No remount or combined DC. The file already sets
    ``full_circle_covered=False`` correctly on load (unlike b411/frogbone/
    fiber-synthetic, which need it overridden), so no geometry patching is
    required here.
    """

    name = "cf-peek"
    has_remount = False
    has_combined = False

    _PATH_DATA = Path("/myhome/data/smartt/shared/cf_peek/sample2015_new.h5")
    _CACHE_DIR = Path("/myhome/data/smartt/shared/results/cf_peek_benchmark")

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        dc = DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)
        dc.geometry.full_circle_covered = False
        return dc
