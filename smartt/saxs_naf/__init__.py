"""SAXS NAF — implicit neural representation for SAXS tensor tomography.

A coordinate network (multiresolution hash encoding + MLP) emits the per-voxel
spherical-harmonic coefficient field, which is fed through the existing linear
forward model (``build_mumott_projector`` + ``forward_quadrature``).  Per-scan,
self-supervised, with linear coarse-to-fine spatial+angular annealing.

See the design memory ``project_saxs_naf_design`` for the full rationale.
"""

from .hash_encoding import MultiResolutionHashEncoding, GridLevel
from .model import SaxsNafField
from .schedule import Annealer
from .reconstruct import saxs_naf_reconstruction
from .cache import save_recon, load_recon, list_cache
from .metrics import (
    split_holdout,
    to_sh_coefficients,
    compute_ground_truth,
    compute_metrics,
    metrics_table,
)
from .eval import (
    evaluate_real_sh,
    coeffs_to_rsm_volumes,
    relative_anisotropy,
    evaluate_models,
    plot_rsm_direction,
)

__all__ = [
    "MultiResolutionHashEncoding",
    "GridLevel",
    "SaxsNafField",
    "Annealer",
    "saxs_naf_reconstruction",
    "save_recon",
    "load_recon",
    "list_cache",
    "split_holdout",
    "to_sh_coefficients",
    "compute_ground_truth",
    "compute_metrics",
    "metrics_table",
    "evaluate_real_sh",
    "coeffs_to_rsm_volumes",
    "relative_anisotropy",
    "evaluate_models",
    "plot_rsm_direction",
]
