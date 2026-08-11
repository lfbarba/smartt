"""Steel-wire WAXS dataset — Ewald-sphere-curvature tensor tomography.

Wide-angle X-ray scattering tensor-tomography scan of a tangled knot of
hardened steel wire (cSAXS beamline, PSI), from Carlsen et al., *J. Appl.
Cryst.* 57, 986-1000 (2024). Detector images were azimuthally regrouped into
48 bins around three austenite/martensite Bragg peaks, each saved as its own
mumott-format file: ``gamma_111.h5``, ``gamma_200.h5``, ``gamma_220.h5``
under ``_DATA_DIR``. Each file carries its own nonzero ``geometry.two_theta``
(the peak's scattering angle), so ``is_waxs = True`` and the flat-plane SH
forward model (see ``smartt.shutils.evaulate_sh.precompute_Y_int``) is not
curvature-correct for this dataset — use ``mumott_projection_matrix`` /
``smartt.saxs_naf.reconstruct``'s auto-detection instead.

Source: Zenodo record 10.5281/zenodo.10889439 (MPL-2.0). Pass ``peak`` to
select which file; ``_PATH_DATA``/``_CACHE_DIR`` are resolved per-instance so
different peaks never share a cache.
"""
from __future__ import annotations

from pathlib import Path

from .base import SmarttDataContainer

_PEAKS = ("111", "200", "220")


class SteelWireWaxsDataContainer(SmarttDataContainer):
    """Single-mount steel-wire WAXS dataset, parametrized by scattering peak.

    No remount or combined DC.

    >>> ds = SteelWireWaxsDataContainer(peak="220")   # gamma_111.h5
    >>> ds = SteelWireWaxsDataContainer()             # default peak
    """

    is_waxs = True
    has_remount = False
    has_combined = False

    _DATA_DIR       = Path("/myhome/data/smartt/shared/steel_wire_waxs")
    _CACHE_DIR_ROOT = Path("/myhome/data/smartt/shared/results/steel_wire_waxs_benchmark")

    _DEFAULT_PEAK = "220"

    def __init__(self, peak = _DEFAULT_PEAK):
        if type(peak) == int:
            peak = str(peak)
        if peak not in _PEAKS:
            raise ValueError(f"peak must be one of {_PEAKS}, got {peak!r}")
        self.peak       = peak
        self._PATH_DATA = self._DATA_DIR / f"gamma_{peak}.h5"
        self._CACHE_DIR = self._CACHE_DIR_ROOT / f"peak_{peak}"
        self.name        = f"steel-wire-waxs-{peak}"

    def get_cache_dir(self) -> Path:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return self._CACHE_DIR

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        return DataContainer(str(self._PATH_DATA), nonfinite_replacement_value=0)
