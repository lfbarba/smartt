"""Nielsen 2023 synthetic SAXS-TT datasets (``M``, ``T``, ``mammoth``).

Simulated missing-wedge datasets published alongside

    Nielsen, Erhart, Guizar-Sicairos & Liebi (2023), *Small-angle scattering
    tensor tomography algorithm for robust reconstruction of complex textures*,
    Acta Cryst. A79, 515-526.  https://doi.org/10.1107/S205327332300863X

and archived at https://zenodo.org/records/7673985 (downloaded into
``/myhome/data/smartt/shared/nielsen_synthetic/``).

Three phantoms, each simulated at five shot-noise levels::

    {sample}_sim_{poisson_rate}.h5      sample in {M, T, mammoth}
                                        poisson_rate in
                                        {1.0e+00, 3.2e+00, 1.0e+01, 3.2e+01, 1.0e+02}

A **larger** ``poisson_rate`` means **less** noise; ``1.0e+02`` (the default here)
is the cleanest.  The phantoms differ in the symmetry of their reciprocal-space
maps: ``M`` is zonal (rotationally symmetric, zonal harmonics up to l=12), ``T``
is fibre-like (pure symmetric rank-2 tensors), and ``mammoth`` has unrestricted
symmetries (spherical harmonics up to l=8 with a weak l=2 component).

Missing-wedge geometry
----------------------
The sample is rotated fully about the beam-orthogonal ``z``-axis (0-360 deg,
``rotations`` / ``inner_angle``) but tilted about ``x`` (``tilts`` /
``outer_angle``) only up to ``max_tilt = pi/4 = 45 deg``, sampled at nine tilt
levels (0, 5.6, 11.2, ..., 45 deg).  The unreachable tilt beyond 45 deg is the
missing wedge.  417 projections in total (one of the 418 simulated frames is
dropped on load) and 8 azimuthal detector segments (Friedel-reduced, so l<=6 is
recoverable from the data even though the ground truth may carry higher orders).
The reconstruction volume is 50x50x50 for ``M`` and ``T`` and 60x60x80 for
``mammoth``.

Ground truth
------------
Unlike the mumott-reconstructed benchmarks, the exact tensor field is stored in
the file's ``model`` group, so :meth:`get_ground_truth` reads it directly — no
reconstruction or sidecar needed.  Coefficients are ``(nx, ny, nz, n_coeff)``
spherical-harmonic values, 4pi-normalized, defined with respect to ``z``, in
mumott order ``{a_0^0, a_-2^2, a_-1^2, a_0^2, a_1^2, a_2^2, a_-4^4, ...}``.  The
band limit — and hence ``n_coeff`` — differs per phantom: ``M`` uses even orders
up to l=12 (91 coefficients), ``T`` only l<=2 (6 coefficients), and ``mammoth``
l<=8 (45 coefficients).  Quality metrics should be restricted to the sample
support via :meth:`get_main_volume_mask`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import SmarttDataContainer


class NielsenSyntheticDataContainer(SmarttDataContainer):
    """Base for the Nielsen 2023 synthetic phantoms — single mount, load and go.

    Subclasses set :attr:`sample` (``"M"``, ``"T"`` or ``"mammoth"``).  The
    shot-noise level is selectable via the ``poisson_rate`` constructor argument
    (default: the cleanest, ``"1.0e+02"``); it must be one of
    :attr:`POISSON_RATES`.
    """

    has_remount = False
    has_combined = False

    #: phantom name, set by subclasses; also the filename stem.
    sample: str = ""

    #: reconstruction volume ``(nx, ny, nz)`` — fixed per phantom by subclasses.
    VOLUME_SHAPE: tuple = ()
    #: number of SH coefficients in the ground truth — fixed per phantom.
    N_COEFF: int = 0
    #: SH band limit (max even order) of the ground truth — fixed per phantom.
    ELL_MAX: int = 0

    #: available shot-noise tokens (filename suffix); larger == less noise.
    POISSON_RATES = ("1.0e+00", "3.2e+00", "1.0e+01", "3.2e+01", "1.0e+02")
    DEFAULT_RATE = "1.0e+02"

    #: tilt-limited missing-wedge extent (about x), read from ``max_tilt``.
    MISSING_WEDGE_MAX_TILT_DEG = 45.0

    _DATA_ROOT = Path("/myhome/data/smartt/shared/nielsen_synthetic")

    def __init__(self, poisson_rate: Optional[str] = None):
        rate = poisson_rate or self.DEFAULT_RATE
        if rate not in self.POISSON_RATES:
            raise ValueError(
                f"Unknown poisson_rate {rate!r}. Choose from {self.POISSON_RATES}."
            )
        self.poisson_rate = rate

    # ------------------------------------------------------------------

    @property
    def _path(self) -> Path:
        return self._DATA_ROOT / f"{self.sample}_sim_{self.poisson_rate}.h5"

    def get_cache_dir(self) -> Path:
        cache = self._DATA_ROOT / "recon_cache" / f"{self.sample}_{self.poisson_rate}"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def get_main_dc(self):
        from mumott.data_handling import DataContainer
        if not self._path.exists():
            raise FileNotFoundError(
                f"{self._path} not found. Download the Nielsen 2023 datasets from "
                "https://zenodo.org/records/7673985 into "
                f"{self._DATA_ROOT}."
            )
        # Rotation covers the full circle; mumott loads full_circle_covered=False
        # by default, matching the file's use in the mumott tutorials.  The
        # missing wedge lives in the tilt axis, not the rotation, so we leave
        # the flag at its loaded default.
        return DataContainer(str(self._path))

    # ------------------------------------------------------------------
    # Exact ground truth (embedded in the file's ``model`` group)
    # ------------------------------------------------------------------

    def get_ground_truth(self) -> np.ndarray:
        """Return the exact SH tensor field ``model/coefficients`` of shape
        ``(*VOLUME_SHAPE, N_COEFF)``.

        4pi-normalized, defined w.r.t. ``z``, in mumott ordering.  The shape is
        fixed per phantom by the subclass (see :attr:`VOLUME_SHAPE`,
        :attr:`N_COEFF`, :attr:`ELL_MAX`) and is asserted against the file on
        load, so callers can rely on it without re-inspecting the data.
        """
        import h5py
        with h5py.File(self._path, "r") as f:
            gt = f["model"]["coefficients"][()]
        expected = (*self.VOLUME_SHAPE, self.N_COEFF)
        if gt.shape != expected:
            raise ValueError(
                f"{self.name}: ground-truth shape {gt.shape} != expected {expected}."
            )
        return gt

    def get_main_volume_mask(self) -> np.ndarray:
        """Return the ``(nx, ny, nz)`` boolean sample-support mask (``model/main_volume``).

        Reconstruction-quality metrics should be evaluated on
        ``coefficients[mask]`` only — voxels outside the support are background.
        """
        import h5py
        with h5py.File(self._path, "r") as f:
            return np.asarray(f["model"]["main_volume"][()]) > 0

    def get_ground_truth_eigen(self):
        """Return ``(eigenvectors, eigenvalues)`` of the rank-2 tensor component.

        ``eigenvectors``: ``(*VOLUME_SHAPE, 3, 3)``; ``eigenvalues``:
        ``(*VOLUME_SHAPE, 3)``, both sorted by eigenvalue in ascending order
        (convenience for orientation plots).
        """
        import h5py
        with h5py.File(self._path, "r") as f:
            return f["model"]["eigenvectors"][()], f["model"]["eigenvalues"][()]


class NielsenMDataContainer(NielsenSyntheticDataContainer):
    """Zonal phantom ``M`` — rotationally symmetric reciprocal-space maps.

    50x50x50 volume, zonal harmonics up to l=12 (91 coefficients).
    """
    name = "nielsen-m"
    sample = "M"
    VOLUME_SHAPE = (50, 50, 50)
    N_COEFF = 91
    ELL_MAX = 12


class NielsenTDataContainer(NielsenSyntheticDataContainer):
    """Fibre-like phantom ``T`` — pure symmetric rank-2 tensors.

    50x50x50 volume, l<=2 only (6 coefficients).
    """
    name = "nielsen-t"
    sample = "T"
    VOLUME_SHAPE = (50, 50, 50)
    N_COEFF = 6
    ELL_MAX = 2


class NielsenMammothDataContainer(NielsenSyntheticDataContainer):
    """Complex phantom ``mammoth`` — unrestricted symmetries (l<=8, weak l=2).

    60x60x80 volume, spherical harmonics up to l=8 (45 coefficients).
    """
    name = "nielsen-mammoth"
    sample = "mammoth"
    VOLUME_SHAPE = (60, 60, 80)
    N_COEFF = 45
    ELL_MAX = 8
