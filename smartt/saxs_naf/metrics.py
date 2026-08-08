"""Quantitative benchmark metrics for SAXS NAF reconstructions.

Public API
----------
split_holdout(dc, fraction, seed)
    Partition a DataContainer into train / held-out subsets.

to_sh_coefficients(basis_or_array, ell_max, n_fit_dirs)
    Convert any mumott basis-set reconstruction to ``(X, Y, Z, C)`` SH
    coefficients in the ``forward_quadrature`` convention.

compute_ground_truth(combined_dc, gt_method, ell_max, n_iterations, ...)
    Run a mumott reconstruction on the full combined DataContainer.

compute_metrics(reconstructions, ground_truth, dc, held_out_dc, ...)
    Core metric computation. All inputs must be ``(X, Y, Z, C)`` SH arrays
    (use :func:`to_sh_coefficients` to convert mumott basis-set outputs first).

Metrics returned per method
---------------------------
rsm_corr_map          (X, Y, Z)  — per-voxel Pearson r with GT RSM over K dirs
rsm_corr_mean         float      — mean over masked voxels
rsm_corr_by_arc       dict       — mean rsm_corr bucketed by missing-arc severity
orientation_error_map (X, Y, Z)  — per-voxel angular error vs GT (degrees), RA-weighted
orientation_error_mean float     — RA-weighted mean angular error (degrees)
ra_map                (X, Y, Z)  — relative anisotropy
ra_mae                float      — mean absolute error of RA vs GT, over mask
fiber_symmetry_map    (X, Y, Z)  — per-voxel degree of cylindrical symmetry
fiber_symmetry_mae    float      — MAE of fiber symmetry vs GT, over mask
psnr                  float      — PSNR of c00 volume vs GT (dB), masked
ssim                  float      — SSIM of c00 volume vs GT, masked
nrmse                 float      — NRMSE of c00 volume vs GT, masked
holdout_nrmse         float      — NRMSE of re-projected held-out measurements
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

def split_holdout(dc, fraction: float = 0.1, seed: int = 42):
    """Split a DataContainer into (train_dc, held_out_dc).

    Both returned containers share no projections.  The split is random and
    reproducible via ``seed``.

    Parameters
    ----------
    dc : mumott DataContainer
    fraction : Fraction of projections to hold out (default 0.10).
    seed : RNG seed.

    Returns
    -------
    train_dc, held_out_dc : two deep-copied DataContainers.
    """
    import copy as _copy
    n = len(dc.projections)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    n_held = max(1, int(round(n * fraction)))
    held_idx = set(all_idx[:n_held].tolist())
    train_idx = set(all_idx[n_held:].tolist())

    train_dc = _copy.deepcopy(dc)
    held_dc = _copy.deepcopy(dc)

    for i in sorted(held_idx, reverse=True):
        del train_dc.projections[i]
    for i in sorted(train_idx, reverse=True):
        del held_dc.projections[i]

    return train_dc, held_dc


# ---------------------------------------------------------------------------
# Basis conversion
# ---------------------------------------------------------------------------

def to_sh_coefficients(
    basis_or_array,
    ell_max: int = 8,
    coefficients: "np.ndarray | None" = None,
) -> np.ndarray:
    """Return ``(X, Y, Z, C)`` SH coefficients for any mumott basis or array.

    Parameters
    ----------
    basis_or_array : ``np.ndarray`` or mumott basis-set instance.
        If a numpy array of shape ``(X, Y, Z, C)`` it is returned as-is
        (assumed already in the ``forward_quadrature`` SH convention).
        If a mumott ``SphericalHarmonics`` basis: ``basis.get_spherical_harmonic_coefficients``
        is called with ``coefficients`` (or ``basis.coefficients`` if omitted).
        If a ``GaussianKernels`` or ``NearestNeighbor`` basis: ``coefficients`` must
        be provided (pass ``result['x']`` from ``LBFGS.optimize()``); mumott's
        native ``basis.get_spherical_harmonic_coefficients`` is used for the
        conversion.
    ell_max : int
        Target SH band-limit.
    coefficients : ``(X, Y, Z, K)`` array, optional
        Optimized basis coefficients.  Required for ``GaussianKernels`` and
        ``NearestNeighbor``.  For ``SphericalHarmonics`` it defaults to
        ``basis.coefficients``.

    Returns
    -------
    ``(X, Y, Z, C)`` float32 numpy array.
    """
    if isinstance(basis_or_array, np.ndarray):
        return basis_or_array.astype(np.float32)

    basis = basis_or_array

    # SphericalHarmonics: delegate to mumott's own converter (handles truncation /
    # zero-padding) using either the supplied coefficients or the live basis state.
    try:
        from mumott.methods.basis_sets import SphericalHarmonics
        if isinstance(basis, SphericalHarmonics):
            coeffs = coefficients if coefficients is not None else basis.coefficients
            return basis.get_spherical_harmonic_coefficients(coeffs, ell_max=ell_max).astype(np.float32)
    except ImportError:
        pass

    # GaussianKernels / NearestNeighbor: coefficients must be supplied explicitly
    # (the basis has no .coefficients attribute; they live in the ResidualCalculator).
    if coefficients is None:
        raise ValueError(
            "GaussianKernels / NearestNeighbor: pass `coefficients=result['x']` "
            "from LBFGS.optimize() to to_sh_coefficients()."
        )
    return basis.get_spherical_harmonic_coefficients(coefficients, ell_max=ell_max).astype(np.float32)


# ---------------------------------------------------------------------------
# Ground-truth reconstruction
# ---------------------------------------------------------------------------

def compute_ground_truth(
    combined_dc,
    gt_method: str = "sh",
    ell_max: int = 8,
    n_iterations: int = 20,
    laplacian_weight: float = 1e-1,
    maxcor: int = 5,
) -> np.ndarray:
    """Run a mumott reconstruction on the full combined DataContainer.

    Parameters
    ----------
    combined_dc : DataContainer with both mounts merged (near-full angular coverage).
    gt_method : ``'sh'`` (SphericalHarmonics + LBFGS) or ``'gk'``
        (GaussianKernels + LBFGS).
    ell_max : SH band-limit (used for ``'sh'`` method).
    n_iterations : LBFGS iterations.  LBFGS converges fast — 20 is typically
        sufficient and matches the existing comparison notebook.
    laplacian_weight : Weight for the Laplacian spatial regulariser (default
        1e-1, same as the comparison notebook).
    maxcor : Number of L-BFGS-B correction vectors (scipy ``maxcor``).

        **Critical for large volumes**: scipy's L-BFGS-B Fortran code uses
        32-bit integers for internal array size calculations.  For
        ``n = X*Y*Z*C`` parameters, the work-array size is approximately
        ``(2*maxcor + 5)*n``.  This overflows INT32 when
        ``maxcor ≥ 9`` at n ≈ 99 M (the 141³ combined volume with C=45),
        causing a **segmentation fault**.  The default ``maxcor=5`` keeps
        the work array at ~11.9 GB, well below the overflow boundary.
        Raise to 7 at most if you need faster convergence on smaller volumes.

    Returns
    -------
    ``(X, Y, Z, C)`` float32 SH coefficients.
    """
    import math
    from mumott.methods.basis_sets import SphericalHarmonics, GaussianKernels
    from mumott.methods.projectors import SAXSProjector
    from mumott.methods.residual_calculators import GradientResidualCalculator
    from mumott.optimization.loss_functions import SquaredLoss
    from mumott.optimization.optimizers import LBFGS
    from mumott.optimization.regularizers import Laplacian

    # Guard: verify maxcor is safe before launching the expensive computation.
    vol = combined_dc.geometry.volume_shape
    n_params = int(vol[0]) * int(vol[1]) * int(vol[2]) * (ell_max + 1) ** 2 // 4 * 2 + (ell_max + 1)
    # Simpler safe upper bound: use actual C derived from ell_max
    n_coeffs = sum(2 * ell + 1 for ell in range(0, ell_max + 1, 2))
    n_params = int(vol[0]) * int(vol[1]) * int(vol[2]) * n_coeffs
    wa_size = (2 * maxcor + 5) * n_params
    int32_max = 2 ** 31 - 1
    if wa_size > int32_max:
        safe_maxcor = int((int32_max / n_params - 5) / 2)
        raise ValueError(
            f"maxcor={maxcor} causes a 32-bit integer overflow in scipy L-BFGS-B "
            f"(wa_size={wa_size:,} > INT32_MAX={int32_max:,}).  "
            f"Use maxcor≤{safe_maxcor} for this volume/ell_max."
        )

    projector = SAXSProjector(combined_dc.geometry)

    if gt_method == "sh":
        basis = SphericalHarmonics(
            ell_max=ell_max,
            probed_coordinates=combined_dc.geometry.probed_coordinates,
        )
    elif gt_method == "gk":
        basis = GaussianKernels(
            probed_coordinates=combined_dc.geometry.probed_coordinates,
        )
    else:
        raise ValueError(f"gt_method must be 'sh' or 'gk', got {gt_method!r}")

    from smartt.shutils.evaulate_sh import patch_nonconvergent_basis_set
    patch_nonconvergent_basis_set(basis)

    rc = GradientResidualCalculator(
        data_container=combined_dc,
        basis_set=basis,
        projector=projector,
    )
    loss = SquaredLoss(residual_calculator=rc, use_weights=True)
    loss.add_regularizer("laplacian", Laplacian(), regularization_weight=laplacian_weight)
    result = LBFGS(loss, maxiter=n_iterations, maxcor=maxcor).optimize()

    # Use the return value (authoritative flat array) and reshape to (X,Y,Z,C).
    # GaussianKernels stores K=50 kernel weights; convert to SH before returning
    # so the output is always (X,Y,Z,C) SH coefficients regardless of gt_method.
    if gt_method == "gk":
        return to_sh_coefficients(basis, ell_max=ell_max, coefficients=result["x"])
    return result["x"].astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample_coeffs(coeffs: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Trilinearly resample ``(X2,Y2,Z2,C)`` to ``(X1,Y1,Z1,C)``."""
    from scipy.ndimage import zoom
    X2, Y2, Z2 = coeffs.shape[:3]
    X1, Y1, Z1 = target_shape
    if (X2, Y2, Z2) == (X1, Y1, Z1):
        return coeffs
    factors = (X1 / X2, Y1 / Y2, Z1 / Z2, 1.0)
    return zoom(coeffs.astype(np.float64), factors, order=1).astype(np.float32)


def fbp_to_sh(
    reconstruction: np.ndarray,
    y_directions: np.ndarray,
    ell_max: int = 8,
) -> np.ndarray:
    """Convert FBP directional output ``(K,X,Y,Z)`` to SH coefficients ``(X,Y,Z,C)``.

    Parameters
    ----------
    reconstruction : ``(K, X, Y, Z)`` float32 array — output of
        :func:`~smartt.saxs_fbp.reconstruction.saxs_fbp_reconstruction`.
    y_directions : ``(K, 3)`` float64 array — unit q-directions returned by
        :func:`~smartt.saxs_fbp.reconstruction.saxs_fbp_reconstruction`.
    ell_max : SH band-limit.

    Returns
    -------
    ``(X, Y, Z, C)`` float32 SH coefficient array.
    """
    import torch
    from .eval import evaluate_real_sh

    K, X, Y, Z = reconstruction.shape
    B = evaluate_real_sh(
        torch.tensor(y_directions, dtype=torch.float32), ell_max
    ).numpy()                               # (K, C)
    B_pinv = np.linalg.pinv(B)             # (C, K)
    recon_flat = reconstruction.reshape(K, -1).astype(np.float32)
    coeffs = (B_pinv @ recon_flat).reshape(B_pinv.shape[0], X, Y, Z).transpose(1, 2, 3, 0)
    return coeffs.astype(np.float32)


def _rsm_volumes(coeffs: np.ndarray, directions: np.ndarray, ell_max: int) -> np.ndarray:
    """``(X,Y,Z,C)`` → ``(K,X,Y,Z)`` directional RSM volumes (numpy)."""
    from .eval import evaluate_real_sh
    import torch
    dirs_t = torch.tensor(directions, dtype=torch.float32)
    B = evaluate_real_sh(dirs_t, ell_max).numpy()              # (K, C)
    return np.einsum("xyzc,kc->kxyz", coeffs, B)


def _pearson_r_voxelwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-voxel Pearson r between ``(K,X,Y,Z)`` arrays over the K axis."""
    a_c = a - a.mean(0, keepdims=True)
    b_c = b - b.mean(0, keepdims=True)
    num = (a_c * b_c).sum(0)
    denom = np.sqrt((a_c ** 2).sum(0) * (b_c ** 2).sum(0)).clip(1e-8)
    return num / denom   # (X, Y, Z)


def _auto_mask(ground_truth: np.ndarray) -> np.ndarray:
    """Auto-Otsu mask on the c00 volume of the ground-truth reconstruction."""
    from skimage.filters import threshold_otsu
    c00 = ground_truth[..., 0]
    thresh = threshold_otsu(c00)
    return c00 > thresh


def _angles_to_unit(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Polar/azimuthal angles → ``(*shape, 3)`` unit vectors."""
    return np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=-1)


def _holdout_nrmse(
    coeffs: np.ndarray,
    held_out_dc,
    ell_max: int,
    device,
) -> float:
    """Re-project ``coeffs`` through ``held_out_dc`` and compute NRMSE.

    Uses mumott's own curvature-correct basis set for WAXS geometries
    (``two_theta`` nonzero) instead of the flat-plane quadrature, matching
    whichever Y_int source the training loop used (see
    ``smartt.saxs_naf.reconstruct._full_dataset_Y_int``) — the absolute scale
    of Y_int is arbitrary (self-calibrated via ``calibrate_c00``), but it must
    be the *same* convention at train and eval time for NRMSE to be meaningful.
    """
    import torch
    from smartt.projectors import build_mumott_projector
    from smartt.shutils.evaulate_sh import (
        forward_quadrature, precompute_Y_int, mumott_projection_matrix,
    )

    geometry = held_out_dc.geometry
    is_waxs = bool(np.any(np.asarray(geometry.two_theta) != 0))
    if is_waxs:
        Y_int = mumott_projection_matrix(
            geometry.probed_coordinates, ell_max=ell_max, device=device,
        )
    else:
        Y_int = precompute_Y_int(geometry.probed_coordinates, ell_max=ell_max, device=device)

    projector = build_mumott_projector(geometry, device=device)
    coeffs_t = torch.tensor(coeffs, dtype=torch.float32, device=device)
    with torch.no_grad():
        spatial = projector(coeffs_t)
        pred = forward_quadrature(
            geometry.probed_coordinates, spatial, ell_max=ell_max,
            projection_matrix=Y_int, mumott_compat=False,
        ).cpu().numpy()

    target = held_out_dc.projections.data
    weights = held_out_dc.projections.weights.astype(bool)
    diff = pred[weights] - target[weights]
    nrmse = float(np.sqrt((diff ** 2).mean()) / (target[weights].std() + 1e-8))
    return nrmse


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    reconstructions: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    dc,
    held_out_dc,
    ell_max: int = 8,
    K: int = 30,
    mask: Optional[np.ndarray] = None,
    half_space: str = "y",
    compute_orientation: bool = True,
    device=None,
) -> Dict[str, Dict]:
    """Compute all benchmark metrics for each reconstruction.

    Parameters
    ----------
    reconstructions : ``name → (X,Y,Z,C)`` float32 SH coefficients.
        Use :func:`to_sh_coefficients` to convert non-SH mumott outputs first.
    ground_truth : ``(X,Y,Z,C)`` float32 SH coefficients.  Typically from
        :func:`compute_ground_truth`.
    dc : DataContainer used for training (provides geometry / missing-arc info).
    held_out_dc : DataContainer containing only the held-out projections.
    ell_max : SH band-limit.
    K : Number of Fibonacci RSM evaluation directions.
    mask : Optional boolean ``(X,Y,Z)`` array.  If ``None``, auto-Otsu on the
        GT c00 volume is used.  Pass ``False`` to disable masking entirely.
    half_space : Hemisphere for Fibonacci sampling (``'y'`` for SAXS).
    compute_orientation : If ``False``, skip the (slow) orientation metrics.
    device : torch.device for held-out re-projection.  Defaults to CUDA if
        available.

    Returns
    -------
    dict : ``method_name → {metric_name → scalar_or_array}``.
        Scalar metrics are plain floats; ``*_map`` entries are ``(X,Y,Z)``
        numpy arrays for visualisation in the notebook.
    """
    import torch
    from skimage.metrics import (
        peak_signal_noise_ratio as psnr_fn,
        structural_similarity as ssim_fn,
    )
    from smartt.saxs_fbp import fibonacci_hemisphere
    from smartt.saxs_isonet.wedge import all_missing_arcs

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- mask ---
    if mask is False:
        flat_mask = np.ones(ground_truth.shape[:3], dtype=bool)
    elif mask is None:
        flat_mask = _auto_mask(ground_truth)
    else:
        flat_mask = mask.astype(bool)

    # --- evaluation directions ---
    dirs = fibonacci_hemisphere(K, half_space=half_space)      # (K, 3)

    # --- ground truth RSM and derived quantities ---
    gt_rsm = _rsm_volumes(ground_truth, dirs, ell_max)         # (K, X, Y, Z)
    gt_ra = gt_rsm.std(0) / gt_rsm.mean(0).clip(1e-8)         # (X, Y, Z)

    # --- missing-arc bucketing ---
    goniometer_axis = getattr(dc.geometry, "rotation_axis", None)
    alpha_deg = getattr(dc.geometry, "tilt_angle", None)
    arc_buckets = None
    if alpha_deg is not None and goniometer_axis is not None:
        arcs = np.degrees(all_missing_arcs(dirs, alpha_deg, goniometer_axis))
        arc_buckets = {
            "low (<60°)":    arcs < 60,
            "medium (60-120°)": (arcs >= 60) & (arcs < 120),
            "high (>120°)":  arcs >= 120,
        }

    # --- orientation of ground truth ---
    gt_theta, gt_phi = None, None
    if compute_orientation:
        from mumott.methods.utilities.fiber_fit import find_approximate_symmetry_axis
        _, gt_theta, gt_phi = find_approximate_symmetry_axis(
            ground_truth.astype(np.float64), ell_max=ell_max
        )
        gt_orient = _angles_to_unit(gt_theta, gt_phi)          # (X, Y, Z, 3)

    # --- per-method metrics ---
    results: Dict[str, Dict] = {}

    gt_spatial = ground_truth.shape[:3]

    for name, coeffs in reconstructions.items():
        coeffs = coeffs.astype(np.float32)
        # Keep a copy in the native geometry for held-out re-projection, which
        # must match the projector built from held_out_dc.geometry.
        coeffs_native = coeffs
        if coeffs.shape[:3] != gt_spatial:
            coeffs = _resample_coeffs(coeffs, gt_spatial)
        m: Dict = {}

        # RSM volumes
        pred_rsm = _rsm_volumes(coeffs, dirs, ell_max)         # (K, X, Y, Z)

        # RSM correlation (per voxel)
        corr_map = _pearson_r_voxelwise(gt_rsm, pred_rsm)      # (X, Y, Z)
        m["rsm_corr_map"] = corr_map
        m["rsm_corr_mean"] = float(corr_map[flat_mask].mean())

        if arc_buckets is not None:
            rsm_by_arc = {}
            for label, bucket in arc_buckets.items():
                if bucket.any():
                    bucket_corr = _pearson_r_voxelwise(
                        gt_rsm[bucket], pred_rsm[bucket]
                    )
                    rsm_by_arc[label] = float(bucket_corr[flat_mask].mean())
            m["rsm_corr_by_arc"] = rsm_by_arc

        # Relative anisotropy
        pred_ra = pred_rsm.std(0) / pred_rsm.mean(0).clip(1e-8)
        m["ra_map"] = pred_ra
        m["ra_mae"] = float(np.abs(pred_ra[flat_mask] - gt_ra[flat_mask]).mean())

        # Fiber symmetry factor: power in zonal components / total power per voxel
        # (derived from find_approximate_symmetry_axis → optimal_zonal_coeffs)
        if compute_orientation:
            from mumott.methods.utilities.fiber_fit import find_approximate_symmetry_axis
            opt_zonal, theta, phi = find_approximate_symmetry_axis(
                coeffs.astype(np.float64), ell_max=ell_max
            )
            # Degree of cylindrical symmetry: zonal energy / total energy
            total_power = (coeffs[..., 1:] ** 2).sum(-1).clip(1e-8)  # exclude c00
            zonal_power = (opt_zonal[..., 1:] ** 2).sum(-1)
            fiber_sym = zonal_power / total_power
            m["fiber_symmetry_map"] = fiber_sym

            # GT fiber symmetry (computed once, reused across methods via closure)
            if "gt_fiber_sym" not in results:
                opt_zonal_gt, _, _ = find_approximate_symmetry_axis(
                    ground_truth.astype(np.float64), ell_max=ell_max
                )
                gt_total = (ground_truth[..., 1:] ** 2).sum(-1).clip(1e-8)
                gt_fiber_sym = (opt_zonal_gt[..., 1:] ** 2).sum(-1) / gt_total
                results["__gt_fiber_sym__"] = gt_fiber_sym

            gt_fs = results.get("__gt_fiber_sym__", fiber_sym * 0)
            m["fiber_symmetry_mae"] = float(
                np.abs(fiber_sym[flat_mask] - gt_fs[flat_mask]).mean()
            )

            # Orientation angular error (RA-weighted)
            pred_orient = _angles_to_unit(theta, phi)
            dot = np.abs((pred_orient * gt_orient).sum(-1)).clip(0, 1)
            ang_err = np.degrees(np.arccos(dot))               # (X, Y, Z)
            weight = gt_ra * flat_mask
            weight_sum = weight.sum() + 1e-8
            m["orientation_error_map"] = ang_err
            m["orientation_error_mean"] = float((ang_err * weight).sum() / weight_sum)

        # PSNR / SSIM / NRMSE on c00 (masked)
        gt_c00 = ground_truth[..., 0]
        pred_c00 = coeffs[..., 0]
        gt_vals = gt_c00[flat_mask]
        pred_vals = pred_c00[flat_mask]
        data_range = float(gt_vals.max() - gt_vals.min()) + 1e-8
        diff = pred_vals - gt_vals
        mse = float((diff ** 2).mean())
        m["psnr"] = float(10 * np.log10(data_range ** 2 / (mse + 1e-12)))
        m["nrmse"] = float(np.sqrt(mse) / (gt_vals.std() + 1e-8))
        # SSIM requires 2D/3D arrays; compute on the masked cube volume
        try:
            m["ssim"] = float(ssim_fn(
                gt_c00, pred_c00,
                data_range=data_range,
            ))
        except Exception:
            m["ssim"] = float("nan")

        # Held-out projection consistency (native geometry — not resampled)
        m["holdout_nrmse"] = _holdout_nrmse(coeffs_native, held_out_dc, ell_max, device)

        results[name] = m

    # Remove internal GT fiber symmetry stash from output
    results.pop("__gt_fiber_sym__", None)
    return results


# ---------------------------------------------------------------------------
# Summary table helper
# ---------------------------------------------------------------------------

def metrics_table(results: Dict[str, Dict]) -> "pandas.DataFrame":
    """Convert scalar metrics from :func:`compute_metrics` to a DataFrame.

    Returns a DataFrame with methods as rows and scalar metrics as columns,
    suitable for display in a notebook or export to LaTeX.
    """
    import pandas as pd
    SCALAR_KEYS = [
        "rsm_corr_mean",
        "ra_mae",
        "fiber_symmetry_mae",
        "orientation_error_mean",
        "psnr",
        "ssim",
        "nrmse",
        "holdout_nrmse",
    ]
    rows = {}
    for name, m in results.items():
        rows[name] = {k: m[k] for k in SCALAR_KEYS if k in m}
    return pd.DataFrame(rows).T
