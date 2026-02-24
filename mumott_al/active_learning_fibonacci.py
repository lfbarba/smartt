"""
Active Learning Benchmark for Sparse Tensor Tomography (Fibonacci Search Space).

This script implements an active learning loop that operates entirely on
**synthetic** projections drawn from a Fibonacci-sampled search space.

Algorithm
---------
1.  Load a real DataContainer from *data_path* and compute a ground-truth (GT)
    reconstruction from all measured projections.

2.  Generate a **search space** of ``n_initial + n_search_extra`` Fibonacci
    directions (default 20 + 980 = 1000) and forward-project the GT
    reconstruction through them to obtain noiseless synthetic measurements
    for every candidate direction.

3.  **Initialise** the active set with the first ``n_initial`` Fibonacci
    directions (indices 0 … n_initial-1 in the search space).

4.  **Active learning loop** (repeated for ``num_iterations``):

    a.  Build the current DataContainer from the active projection indices.

    b.  Compute ``num_experiments`` independent reconstructions:

        * If ``subsample_fraction == 1.0`` (or
          ``num_subsamples == len(active_indices)``), all experiments reuse
          the full active DataContainer (saves time).
        * Otherwise, each experiment randomly draws
          ``round(subsample_fraction * len(active_indices))`` projections from
          the active set and reconstructs from that smaller DataContainer.

    c.  Forward-project every reconstruction through the **full** search-space
        geometry (all 1 000 directions).

    d.  Compute the **total variance** of each candidate direction — the sum
        of the variance (across experiments) over all pixels and detector
        channels.

    e.  Select the top ``b`` candidate directions with the highest variance
        that are **not** already in the active set.  Add them to the active set.

    f.  After updating the active set, reconstruct once more (mean of
        ``num_experiments`` runs) and evaluate the full metric suite from
        ``mumott_al.metrics.compare_reconstructions`` against the GT.

5.  Log every iteration's metrics and metadata to Weights & Biases.

Usage
-----
    python -m mumott_al.active_learning_fibonacci \\
        --data-path /path/to/dataset.h5 \\
        [--n-initial 20] \\
        [--n-search-extra 980] \\
        [--num-iterations 10] \\
        [--num-experiments 5] \\
        [--subsample-fraction 0.8] \\
        [--b 5] \\
        [options]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Local imports — add workspace root so the script can be run from any cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mumott.data_handling import DataContainer
from mumott import Geometry
from mumott.core.geometry import GeometryTuple
from mumott.core.projection_stack import ProjectionStack, Projection
from mumott.methods.basis_sets import SphericalHarmonics
from mumott.methods.projectors import SAXSProjector, SAXSProjectorCUDA

from mumott_al.geometry import (
    fibonacci_hemisphere,
    create_geometry_from_directions,
    generate_geometry_and_projections,
    create_synthetic_data_container,
    create_synthetic_projections,
)
from mumott_al.metrics import compare_reconstructions, print_comparison_results
from mumott_al.synthetic_data_processing import _perform_reconstruction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WANDB_PROJECT_DEFAULT = "active-learning-fibonacci"

# ---------------------------------------------------------------------------
# Ground-truth caching (disk)
# ---------------------------------------------------------------------------


def _gt_cache_path(data_path: str) -> Path:
    """Return a deterministic temp-file path for caching the GT reconstruction."""
    abs_path = str(Path(data_path).resolve())
    try:
        mtime = str(os.path.getmtime(abs_path))
    except OSError:
        mtime = "0"
    key = f"{abs_path}|{mtime}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_dir = Path(os.getenv("TMPDIR", "/tmp")) / "mumott_al_gt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"gt_{key_hash}.npz"


def _load_gt_reconstruction(
    data_path: str,
    ell_max: int,
    maxiter: int,
    regularization_weight: float,
    use_cuda: bool,
    verbose: bool,
) -> Tuple[np.ndarray, DataContainer]:
    """Load (or compute and cache) the GT reconstruction and DataContainer.

    Returns
    -------
    gt_reconstruction : np.ndarray  shape (*volume_shape, n_coeffs)
    dc               : DataContainer  (full measured data)
    """
    cache_path = _gt_cache_path(data_path)
    dc = DataContainer(str(data_path), nonfinite_replacement_value=0.0)

    if cache_path.exists():
        if verbose:
            print(f"[GT] Loading cached reconstruction from {cache_path}")
        data = np.load(str(cache_path))
        return data["gt_reconstruction"].astype(np.float32), dc

    if verbose:
        print(f"[GT] Computing reconstruction from {len(dc.geometry.inner_angles)} projections …")

    t0 = time.perf_counter()
    gt_reconstruction = _perform_reconstruction(
        dc=dc,
        ell_max=ell_max,
        maxiter=maxiter,
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"[GT] Reconstruction done in {elapsed:.1f} s.  Saving cache to {cache_path}")

    np.savez_compressed(str(cache_path), gt_reconstruction=gt_reconstruction)
    return gt_reconstruction, dc


# ---------------------------------------------------------------------------
# Search-space helpers
# ---------------------------------------------------------------------------


def _build_search_space(
    gt_reconstruction: np.ndarray,
    reference_dc: DataContainer,
    n_initial: int,
    n_search_extra: int,
    ell_max: int,
    verbose: bool,
) -> Tuple[DataContainer, np.ndarray, Geometry]:
    """Generate the full Fibonacci search-space DataContainer.

    All ``n_initial + n_search_extra`` Fibonacci directions are generated and
    the GT reconstruction is forward-projected through them.  The first
    ``n_initial`` directions are treated as the initial active set.

    Parameters
    ----------
    gt_reconstruction:
        GT volume array shape ``(*vol, n_coeffs)``.
    reference_dc:
        Real DataContainer used to copy detector / volume metadata.
    n_initial:
        Number of initial projection directions (first entries in the search space).
    n_search_extra:
        Number of additional candidate directions (the pool to draw from).
    ell_max:
        Maximum spherical harmonic degree.
    verbose:
        Print progress.

    Returns
    -------
    search_dc : DataContainer
        DataContainer containing all ``n_initial + n_search_extra`` synthetic
        projections.
    search_directions : np.ndarray  shape (n_total, 3)
        Unit-vector directions for every projection in *search_dc*.
    search_geometry : Geometry
        Geometry object for the full search space.
    """
    n_total = n_initial + n_search_extra
    if verbose:
        print(f"[Search space] Generating {n_total} Fibonacci directions "
              f"({n_initial} initial + {n_search_extra} extra) …")

    search_directions = fibonacci_hemisphere(n_total, upper=True)

    t0 = time.perf_counter()
    search_geometry, projection_stack = generate_geometry_and_projections(
        reconstruction=gt_reconstruction,
        directions=search_directions,
        reference_geometry=reference_dc.geometry,
        ell_max=ell_max,
        return_data_container=True,
        copy_from_reference=False,
    )
    elapsed = time.perf_counter() - t0

    search_dc = create_synthetic_data_container(
        geometry=search_geometry,
        projection_stack=projection_stack,
        reference_dc=reference_dc,
    )

    if verbose:
        print(f"[Search space] Done in {elapsed:.1f} s.  "
              f"Data shape: {search_dc.projections.data.shape}")

    return search_dc, search_directions, search_geometry


def _subset_dc_from_search_space(
    search_dc: DataContainer,
    search_geometry: Geometry,
    indices: np.ndarray,
) -> DataContainer:
    """Build a DataContainer containing only the *indices* from *search_dc*.

    Uses the projection stack and geometry from *search_dc* to avoid
    recomputing any forward projections.

    Parameters
    ----------
    search_dc:
        Full search-space DataContainer (n_total projections).
    search_geometry:
        Geometry for the full search space.
    indices:
        1-D integer array of projection indices (0-based) to include.

    Returns
    -------
    DataContainer with ``len(indices)`` projections.
    """
    indices = np.asarray(indices, dtype=int)
    subset_stack = ProjectionStack()

    for i in indices:
        old_proj = search_dc.projections[i]
        new_proj = Projection(
            data=old_proj.data.copy(),
            weights=old_proj.weights.copy(),
            rotation=np.array(search_geometry.rotations[i]),
            j_offset=search_geometry.j_offsets[i],
            k_offset=search_geometry.k_offsets[i],
            inner_angle=search_geometry.inner_angles[i],
            outer_angle=search_geometry.outer_angles[i],
            inner_axis=np.array(search_geometry.inner_axes[i]),
            outer_axis=np.array(search_geometry.outer_axes[i]),
        )
        subset_stack.append(new_proj)

    # Build a subset geometry for the DataContainer
    subset_geom = Geometry()
    for attr in (
        "_projection_shape",
        "_volume_shape",
        "_detector_angles",
        "_two_theta",
        "_full_circle_covered",
        "_p_direction_0",
        "_j_direction_0",
        "_k_direction_0",
        "_detector_direction_origin",
        "_detector_direction_positive_90",
    ):
        if hasattr(search_geometry, attr):
            val = getattr(search_geometry, attr)
            setattr(
                subset_geom,
                attr,
                val.copy() if hasattr(val, "copy") else val,
            )

    for i in indices:
        subset_geom.append(
            GeometryTuple(
                rotation=np.array(search_geometry.rotations[i]),
                j_offset=search_geometry.j_offsets[i],
                k_offset=search_geometry.k_offsets[i],
                inner_angle=search_geometry.inner_angles[i],
                outer_angle=search_geometry.outer_angles[i],
                inner_axis=np.array(search_geometry.inner_axes[i]),
                outer_axis=np.array(search_geometry.outer_axes[i]),
            )
        )

    return create_synthetic_data_container(
        geometry=subset_geom,
        projection_stack=subset_stack,
        reference_dc=search_dc,
    )


# ---------------------------------------------------------------------------
# Reconstruction ensemble
# ---------------------------------------------------------------------------


def _compute_reconstruction_ensemble(
    current_dc: DataContainer,
    search_dc: DataContainer,
    search_geometry: Geometry,
    active_indices: np.ndarray,
    num_experiments: int,
    subsample_fraction: float,
    ell_max: int,
    maxiter: int,
    regularization_weight: float,
    use_cuda: bool,
    rng: np.random.Generator,
    verbose: bool,
) -> np.ndarray:
    """Compute an ensemble of reconstructions with optional sub-sampling.

    Parameters
    ----------
    current_dc:
        DataContainer containing the current active projections (used directly
        when ``subsample_fraction == 1.0``).
    search_dc:
        Full search-space DataContainer (used to draw sub-samples).
    search_geometry:
        Full search-space Geometry.
    active_indices:
        Current active projection indices into the search space.
    num_experiments:
        Number of independent reconstructions to compute.
    subsample_fraction:
        Fraction of active projections to use per experiment.  ``1.0`` disables
        sub-sampling (all experiments use the same *current_dc*).
    ell_max, maxiter, regularization_weight, use_cuda:
        Passed to ``_perform_reconstruction``.
    rng:
        NumPy random-number generator for sub-sampling.
    verbose:
        Print progress.

    Returns
    -------
    np.ndarray  shape (num_experiments, *vol_shape, n_coeffs)
    """
    n_active = len(active_indices)
    num_subsamples = round(subsample_fraction * n_active)

    # Decide whether sub-sampling is needed
    no_subsampling = (subsample_fraction >= 1.0) or (num_subsamples >= n_active)

    vol_shape = current_dc.geometry.volume_shape
    n_coeffs = gt_reconstruction_shape_to_n_coeffs(ell_max)
    ensemble = np.zeros((num_experiments, *vol_shape, n_coeffs), dtype=np.float32)

    for exp_idx in range(num_experiments):
        if no_subsampling:
            dc_exp = current_dc
        else:
            sub_indices = rng.choice(active_indices, size=num_subsamples, replace=False)
            sub_indices = np.sort(sub_indices)
            dc_exp = _subset_dc_from_search_space(search_dc, search_geometry, sub_indices)

        if verbose:
            n_used = len(dc_exp.geometry.inner_angles)
            print(f"    Experiment {exp_idx + 1}/{num_experiments}: "
                  f"{n_used} projections …", end=" ", flush=True)

        t0 = time.perf_counter()
        recon = _perform_reconstruction(
            dc=dc_exp,
            ell_max=ell_max,
            maxiter=maxiter,
            regularization_weight=regularization_weight,
            use_cuda=use_cuda,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        ensemble[exp_idx] = recon

        if verbose:
            print(f"done in {elapsed:.1f} s")

        # Free DC only if it was freshly created (i.e. sub-sampled)
        if not no_subsampling:
            del dc_exp, recon

    return ensemble


def gt_reconstruction_shape_to_n_coeffs(ell_max: int) -> int:
    """Number of SH coefficients for even-ℓ-only mumott convention."""
    return sum(2 * ell + 1 for ell in range(0, ell_max + 1, 2))


# ---------------------------------------------------------------------------
# Forward projection and variance
# ---------------------------------------------------------------------------


def _forward_project_ensemble(
    ensemble: np.ndarray,
    search_geometry: Geometry,
    ell_max: int,
    use_cuda: bool,
) -> np.ndarray:
    """Forward-project every reconstruction in *ensemble* to all search-space directions.

    Parameters
    ----------
    ensemble:
        Array of shape ``(num_experiments, *vol_shape, n_coeffs)``.
    search_geometry:
        Full search-space Geometry (n_total directions).
    ell_max:
        Maximum spherical harmonic degree.
    use_cuda:
        Whether to use the CUDA projector.

    Returns
    -------
    np.ndarray  shape (num_experiments, n_total, J, K, n_channels)
    """
    if use_cuda:
        projector = SAXSProjectorCUDA(search_geometry)
    else:
        projector = SAXSProjector(search_geometry)

    basis_set = SphericalHarmonics(
        ell_max=ell_max,
        probed_coordinates=search_geometry.probed_coordinates,
    )

    num_experiments = ensemble.shape[0]
    n_total = len(search_geometry.inner_angles)
    proj_shape = search_geometry.projection_shape
    n_channels = len(search_geometry.detector_angles)

    all_fp = np.zeros(
        (num_experiments, n_total, proj_shape[0], proj_shape[1], n_channels),
        dtype=np.float32,
    )

    for exp_idx in range(num_experiments):
        recon = ensemble[exp_idx].astype(np.float64)
        spatial = projector.forward(recon)
        fp = basis_set.forward(spatial)
        all_fp[exp_idx] = fp.astype(np.float32)
        del recon, spatial, fp

    return all_fp


def _compute_variance_per_direction(forward_projections: np.ndarray) -> np.ndarray:
    """Sum-of-variance score for each candidate direction.

    Parameters
    ----------
    forward_projections:
        Array of shape ``(num_experiments, n_total, J, K, n_channels)``.

    Returns
    -------
    np.ndarray  shape (n_total,)  — higher = more uncertain
    """
    # Variance across experiments for every pixel × channel
    var = np.var(forward_projections, axis=0)   # (n_total, J, K, n_channels)
    # Sum over spatial and channel dims
    return var.sum(axis=(1, 2, 3))              # (n_total,)


def _select_top_b(
    variance_scores: np.ndarray,
    active_indices: np.ndarray,
    b: int,
) -> np.ndarray:
    """Return the *b* candidate indices with highest variance not already active.

    Parameters
    ----------
    variance_scores:
        Shape ``(n_total,)``.
    active_indices:
        Currently active projection indices (excluded from selection).
    b:
        Number of new projections to select.

    Returns
    -------
    np.ndarray  shape (b,)
    """
    scores = variance_scores.copy()
    scores[active_indices] = -np.inf   # mask out already active
    return np.argsort(scores)[-b:][::-1]


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------


def _make_run_name(data_path: str, n_initial: int, n_search_extra: int) -> str:
    stem = Path(data_path).stem
    return f"AL_fib_{stem}_init{n_initial}_pool{n_search_extra}"


def _check_wandb_run_exists(
    project: str,
    data_path: str,
    n_initial: int,
    n_search_extra: int,
    entity: Optional[str] = None,
) -> bool:
    """Return True if a finished wandb run with matching config already exists."""
    try:
        import wandb
    except ImportError:
        return False
    try:
        api = wandb.Api()
        filters = {
            "config.data_path": data_path,
            "config.n_initial": n_initial,
            "config.n_search_extra": n_search_extra,
            "state": "finished",
        }
        entity_str = entity or api.default_entity
        runs = api.runs(f"{entity_str}/{project}", filters=filters)
        for _ in runs:
            return True
        return False
    except Exception:
        return False


def _log_iteration_to_wandb(
    run,
    iteration: int,
    n_active: int,
    metrics,
    selected_indices: np.ndarray,
    variance_scores: np.ndarray,
) -> None:
    """Log per-iteration data to wandb."""
    log_dict: dict = {
        "iteration": iteration,
        "n_active_projections": n_active,
        "mse_coefficients": metrics.mse_coefficients,
        "normalized_mse_coefficients": metrics.normalized_mse_coefficients,
        # Mean variance of newly selected directions
        "mean_selected_variance": float(variance_scores[selected_indices].mean())
        if len(selected_indices) > 0
        else float("nan"),
    }

    if metrics.projection_metrics is not None:
        pm = metrics.projection_metrics
        log_dict.update(
            {
                "proj/mse_global": pm.mse_global,
                "proj/psnr_global": pm.psnr_global,
                "proj/ssim_global": pm.ssim_global,
            }
        )

    if metrics.orientation_metrics is not None:
        om = metrics.orientation_metrics
        log_dict.update(
            {
                "orient/cosine_similarity_mean": om.cosine_similarity_mean,
                "orient/angular_error_mean_deg": om.angular_error_mean_degrees,
            }
        )

    if metrics.real_space_metrics is not None:
        rs = metrics.real_space_metrics
        log_dict.update(
            {
                "real_space/mse": rs.mse,
                "real_space/psnr": rs.psnr,
                "real_space/ssim": rs.ssim,
            }
        )

    run.log(log_dict)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def run_active_learning_fibonacci(
    data_path: str,
    n_initial: int = 20,
    n_search_extra: int = 980,
    num_iterations: int = 10,
    num_experiments: int = 5,
    subsample_fraction: float = 0.8,
    b: int = 5,
    ell_max: int = 8,
    maxiter: int = 20,
    regularization_weight: float = 1.0,
    wandb_project: str = _WANDB_PROJECT_DEFAULT,
    wandb_entity: Optional[str] = None,
    force_cpu: bool = False,
    real_space_resolution_deg: int = 10,
    seed: Optional[int] = None,
    verbose: bool = True,
    skip_wandb_check: bool = False,
) -> None:
    """Run the Fibonacci active-learning benchmark and log results to wandb.

    Parameters
    ----------
    data_path:
        Path to the input HDF5 DataContainer (used to compute the GT).
    n_initial:
        Number of initial Fibonacci directions in the active set.
    n_search_extra:
        Number of additional Fibonacci directions forming the candidate pool.
        Total search-space size = *n_initial* + *n_search_extra*.
    num_iterations:
        Number of active-learning iterations.
    num_experiments:
        Number of reconstructions computed per iteration (ensemble size).
    subsample_fraction:
        Fraction of the active set used in each ensemble member.
        ``1.0`` disables sub-sampling (all members use the full active set).
    b:
        Number of new projection directions added per iteration.
    ell_max:
        Maximum spherical harmonic degree.
    maxiter:
        Maximum LBFGS iterations per reconstruction.
    regularization_weight:
        Laplacian regularisation weight.
    wandb_project:
        Weights & Biases project name.
    wandb_entity:
        Weights & Biases entity / team name (optional).
    force_cpu:
        Disable CUDA even when a GPU is available.
    real_space_resolution_deg:
        Angular resolution for the real-space SH evaluation grid.
    seed:
        Random seed for reproducibility.
    verbose:
        Print progress information.
    skip_wandb_check:
        If True, skip the "already exists" check and always run.
    """
    rng = np.random.default_rng(seed)
    use_cuda = torch.cuda.is_available() and not force_cpu
    n_total_search = n_initial + n_search_extra

    if verbose:
        print("=" * 72)
        print("FIBONACCI ACTIVE LEARNING BENCHMARK")
        print("=" * 72)
        print(f"  data_path          : {data_path}")
        print(f"  n_initial          : {n_initial}")
        print(f"  n_search_extra     : {n_search_extra}")
        print(f"  total search space : {n_total_search}")
        print(f"  num_iterations     : {num_iterations}")
        print(f"  num_experiments    : {num_experiments}")
        print(f"  subsample_fraction : {subsample_fraction}")
        print(f"  b (new per iter)   : {b}")
        print(f"  ell_max            : {ell_max}")
        print(f"  maxiter            : {maxiter}")
        print(f"  regularization_wt  : {regularization_weight}")
        print(f"  CUDA               : {use_cuda}")
        print("=" * 72)

    # ------------------------------------------------------------------  [0]
    # wandb duplicate check
    # ------------------------------------------------------------------
    if not skip_wandb_check and _check_wandb_run_exists(
        project=wandb_project,
        data_path=data_path,
        n_initial=n_initial,
        n_search_extra=n_search_extra,
        entity=wandb_entity,
    ):
        print(
            "[wandb] A finished run with the same config already exists.  "
            "Skipping (use --skip-wandb-check to force re-run)."
        )
        return

    # ------------------------------------------------------------------  [1]
    # Ground-truth reconstruction
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1] Computing / loading GT reconstruction …")

    gt_reconstruction, reference_dc = _load_gt_reconstruction(
        data_path=data_path,
        ell_max=ell_max,
        maxiter=maxiter,
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=verbose,
    )

    if verbose:
        print(f"    GT shape: {gt_reconstruction.shape}")

    # ------------------------------------------------------------------  [2]
    # Build the full Fibonacci search space (synthetic projections from GT)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[2] Building Fibonacci search-space …")

    search_dc, search_directions, search_geometry = _build_search_space(
        gt_reconstruction=gt_reconstruction,
        reference_dc=reference_dc,
        n_initial=n_initial,
        n_search_extra=n_search_extra,
        ell_max=ell_max,
        verbose=verbose,
    )

    # ------------------------------------------------------------------  [3]
    # Initialise active set with the first n_initial directions
    # ------------------------------------------------------------------
    active_indices = np.arange(n_initial, dtype=int)
    current_dc = _subset_dc_from_search_space(search_dc, search_geometry, active_indices)

    if verbose:
        print(f"\n[3] Initial active set: {n_initial} projections "
              f"(indices 0 … {n_initial - 1})")

    # ------------------------------------------------------------------
    # wandb run
    # ------------------------------------------------------------------
    wandb_run = None
    try:
        import wandb

        run_name = _make_run_name(data_path, n_initial, n_search_extra)
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "data_path": data_path,
                "n_initial": n_initial,
                "n_search_extra": n_search_extra,
                "n_total_search": n_total_search,
                "num_iterations": num_iterations,
                "num_experiments": num_experiments,
                "subsample_fraction": subsample_fraction,
                "b": b,
                "ell_max": ell_max,
                "maxiter": maxiter,
                "regularization_weight": regularization_weight,
                "use_cuda": use_cuda,
                "seed": seed,
            },
        )
    except ImportError:
        if verbose:
            print("[wandb] wandb not installed — metrics will not be logged.")
    except Exception as exc:
        if verbose:
            print(f"[wandb] Could not initialise run: {exc}")

    # ------------------------------------------------------------------  [4]
    # Active learning loop
    # ------------------------------------------------------------------
    selected_indices_this_iter = np.array([], dtype=int)

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'=' * 72}")
            print(f"  ITERATION {iteration + 1}/{num_iterations}  "
                  f"| active projections: {len(active_indices)}")
            print(f"{'=' * 72}")

        # ---- 4a: ensemble of reconstructions ----
        if verbose:
            print(f"\n  [4a] Computing {num_experiments} reconstructions …")

        ensemble = _compute_reconstruction_ensemble(
            current_dc=current_dc,
            search_dc=search_dc,
            search_geometry=search_geometry,
            active_indices=active_indices,
            num_experiments=num_experiments,
            subsample_fraction=subsample_fraction,
            ell_max=ell_max,
            maxiter=maxiter,
            regularization_weight=regularization_weight,
            use_cuda=use_cuda,
            rng=rng,
            verbose=verbose,
        )

        # ---- 4b: forward-project to full search space ----
        if verbose:
            print(f"\n  [4b] Forward-projecting ensemble to all "
                  f"{n_total_search} search directions …", end=" ", flush=True)

        t0 = time.perf_counter()
        forward_projections = _forward_project_ensemble(
            ensemble=ensemble,
            search_geometry=search_geometry,
            ell_max=ell_max,
            use_cuda=use_cuda,
        )
        elapsed = time.perf_counter() - t0

        if verbose:
            print(f"done in {elapsed:.1f} s")

        # ---- 4c: variance per direction ----
        variance_scores = _compute_variance_per_direction(forward_projections)
        del forward_projections, ensemble

        # ---- 4d: select top-b new directions ----
        selected_indices_this_iter = _select_top_b(
            variance_scores=variance_scores,
            active_indices=active_indices,
            b=b,
        )

        if verbose:
            print(f"\n  [4c] Selected {b} new projections:")
            for idx in selected_indices_this_iter:
                print(f"       index {idx:4d}  variance {variance_scores[idx]:.3e}")

        # ---- 4e: update active set ----
        active_indices = np.sort(
            np.concatenate([active_indices, selected_indices_this_iter])
        )
        current_dc = _subset_dc_from_search_space(
            search_dc, search_geometry, active_indices
        )

        if verbose:
            print(f"\n  Active set updated → {len(active_indices)} projections")

        # ---- 4f: evaluate metrics ----
        if verbose:
            print(f"\n  [4d] Evaluating metrics vs GT …")

        # Use the mean of the last ensemble (cheapest option; could re-run)
        # Actually re-compute one ensemble for a clean mean after updating set
        mean_reconstruction = np.mean(
            _compute_reconstruction_ensemble(
                current_dc=current_dc,
                search_dc=search_dc,
                search_geometry=search_geometry,
                active_indices=active_indices,
                num_experiments=num_experiments,
                subsample_fraction=1.0,  # full set for metric evaluation
                ell_max=ell_max,
                maxiter=maxiter,
                regularization_weight=regularization_weight,
                use_cuda=use_cuda,
                rng=rng,
                verbose=False,
            ),
            axis=0,
        )

        metrics = compare_reconstructions(
            reconstruction_pred=mean_reconstruction,
            reconstruction_gt=gt_reconstruction,
            geometry=current_dc.geometry,
            ell_max=ell_max,
            mask=None,
            weights=current_dc.projections.weights,
            compute_projection_metrics=True,
            compute_orientation_metrics=True,
            compute_real_space_metrics=True,
            real_space_resolution_in_degrees=real_space_resolution_deg,
            real_space_half_sphere=True,
            verbose=False,
        )

        if verbose:
            print_comparison_results(metrics)

        if wandb_run is not None:
            _log_iteration_to_wandb(
                run=wandb_run,
                iteration=iteration + 1,
                n_active=len(active_indices),
                metrics=metrics,
                selected_indices=selected_indices_this_iter,
                variance_scores=variance_scores,
            )

        del variance_scores

    # ------------------------------------------------------------------  [5]
    # Finish
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 72}")
        print("ACTIVE LEARNING COMPLETE")
        print(f"  Final active projections: {len(active_indices)}")
        print(f"  Final active indices:     {active_indices.tolist()}")
        print("=" * 72)

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mumott_al.active_learning_fibonacci",
        description=(
            "Fibonacci Active Learning Benchmark for tensor tomography.  "
            "Iteratively selects the most informative projection directions "
            "from a Fibonacci search space and logs metrics to wandb."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the input HDF5 DataContainer.",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=20,
        metavar="N",
        help="Number of initial Fibonacci directions in the active set.",
    )
    parser.add_argument(
        "--n-search-extra",
        type=int,
        default=980,
        metavar="N",
        help="Number of additional candidate Fibonacci directions (pool size).",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        metavar="T",
        help="Number of active learning iterations.",
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=5,
        metavar="M",
        help="Number of reconstructions per iteration (ensemble size).",
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.8,
        metavar="F",
        help=(
            "Fraction of active projections used per ensemble member "
            "(1.0 = no sub-sampling)."
        ),
    )
    parser.add_argument(
        "--b",
        type=int,
        default=5,
        metavar="B",
        help="Number of new projection directions added per iteration.",
    )
    parser.add_argument(
        "--ell-max",
        type=int,
        default=8,
        metavar="N",
        help="Maximum spherical harmonic degree.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        metavar="N",
        help="Maximum LBFGS iterations for each reconstruction.",
    )
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=1.0,
        metavar="W",
        help="Laplacian regularisation weight.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=_WANDB_PROJECT_DEFAULT,
        metavar="PROJECT",
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        metavar="ENTITY",
        help="wandb entity / team name (optional).",
    )
    parser.add_argument(
        "--real-space-resolution",
        type=int,
        default=10,
        metavar="DEG",
        help="Angular resolution in degrees for real-space SH metric evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CUDA even when a GPU is available.",
    )
    parser.add_argument(
        "--skip-wandb-check",
        action="store_true",
        help="Skip the wandb duplicate-run check and always run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser


def main() -> None:
    """Entry point for CLI / ``python -m mumott_al.active_learning_fibonacci``."""
    parser = _build_parser()
    args = parser.parse_args()

    run_active_learning_fibonacci(
        data_path=args.data_path,
        n_initial=args.n_initial,
        n_search_extra=args.n_search_extra,
        num_iterations=args.num_iterations,
        num_experiments=args.num_experiments,
        subsample_fraction=args.subsample_fraction,
        b=args.b,
        ell_max=args.ell_max,
        maxiter=args.maxiter,
        regularization_weight=args.regularization_weight,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        force_cpu=args.force_cpu,
        real_space_resolution_deg=args.real_space_resolution,
        seed=args.seed,
        verbose=not args.quiet,
        skip_wandb_check=args.skip_wandb_check,
    )


if __name__ == "__main__":
    main()
