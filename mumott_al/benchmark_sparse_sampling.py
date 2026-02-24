"""
Sparse Sampling Benchmark for tensor tomography.

This script evaluates three sparse projection-sampling strategies against a
ground-truth (GT) reconstruction built from the complete measured dataset:

  SNUS      – Static Non-Uniform Sampling.
              Takes the **first k** projections from the GT DataContainer as
              they appear in its geometry file.

  sparse_SNUS – Sparse Static Non-Uniform Sampling.
              Selects **k projections as uniformly spaced as possible** from
              the GT geometry (stride-based subset, random fill for any
              shortfall).

  FSUS      – Fibonacci Static Uniform Sampling.
              Computes **k entirely new directions** via the Fibonacci
              hemisphere method and synthesises the corresponding projections
              from the GT reconstruction.

For all strategies:
  1. Define the k-projection geometry.
  2. Forward-project the GT reconstruction to obtain noiseless synthetic
     measurements.
  3. Reconstruct the volume from those synthetic measurements.
  4. Compare the reconstruction against the GT with the full metric suite
     from ``mumott_al.metrics``.
  5. Log everything to Weights & Biases.

Before any computation the script queries wandb to check whether results for
the exact (benchmark, data_path, num_projections) triplet already exist.  If a
finished run is found the script exits early.

Usage
-----
    python -m mumott_al.benchmark_sparse_sampling \\
        --benchmark SNUS \\
        --data-path /path/to/dataset.h5 \\
        --num-projections 40 \\
        [options]

Shell launcher
--------------
Use ``scripts/run_sparse_benchmarks.sh`` to sweep over sparsity levels 20–240
in steps of 20 for a given benchmark and data path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

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

from mumott_al.geometry import (
    fibonacci_hemisphere,
    create_geometry_from_directions,
    generate_geometry_and_projections,
    create_synthetic_data_container,
)
from mumott_al.metrics import compare_reconstructions, print_comparison_results
from mumott_al.synthetic_data_processing import _perform_reconstruction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARKS = ("SNUS", "sparse_SNUS", "FSUS")
_WANDB_PROJECT_DEFAULT = "sparse-sampling-benchmark"

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _subset_geometry(reference_geometry: Geometry, indices: np.ndarray) -> Geometry:
    """Return a new Geometry containing only the projections at *indices*.

    All per-projection parameters (rotation, offsets, axes, angles) are copied
    directly from *reference_geometry*, so the physical setup is preserved
    exactly.

    Parameters
    ----------
    reference_geometry:
        Source geometry to copy projections from.
    indices:
        1-D integer array of projection indices (0-based) to include.

    Returns
    -------
    Geometry
        New Geometry with ``len(indices)`` projections.
    """
    indices = np.asarray(indices, dtype=int)
    new_geom = Geometry()

    # Copy scalar / array attributes that do not depend on the number of
    # projections.
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
        if hasattr(reference_geometry, attr):
            setattr(new_geom, attr, getattr(reference_geometry, attr).copy()
                    if hasattr(getattr(reference_geometry, attr), "copy")
                    else getattr(reference_geometry, attr))

    for i in indices:
        geom_tuple = GeometryTuple(
            rotation=np.array(reference_geometry.rotations[i]),
            j_offset=reference_geometry.j_offsets[i],
            k_offset=reference_geometry.k_offsets[i],
            inner_angle=reference_geometry.inner_angles[i],
            outer_angle=reference_geometry.outer_angles[i],
            inner_axis=np.array(reference_geometry.inner_axes[i]),
            outer_axis=np.array(reference_geometry.outer_axes[i]),
        )
        new_geom.append(geom_tuple)

    return new_geom


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------


def _select_snus_indices(n_total: int, k: int) -> np.ndarray:
    """Return the first *k* projection indices (SNUS).

    Parameters
    ----------
    n_total:
        Total number of projections in the GT geometry.
    k:
        Number of projections to select.
    """
    if k > n_total:
        raise ValueError(
            f"Requested {k} projections but only {n_total} are available in the GT geometry."
        )
    return np.arange(k, dtype=int)


def _select_sparse_snus_indices(
    n_total: int,
    k: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return *k* projection indices spread as uniformly as possible (sparse-SNUS).

    Algorithm
    ---------
    1. Compute ``stride = ceil(n_total / k)`` — the largest stride that keeps
       the uniform subset within [0, n_total).  This ensures indices are spread
       across the full angular range rather than packed toward the start.
    2. The uniform subset is ``np.arange(0, n_total, stride)``; it has at most
       *k* elements (and exactly *k* when *n_total* is divisible by *k*).
    3. If the uniform subset has fewer than *k* elements (which happens whenever
       ``n_total % k != 0`` and the ceiling causes a short last stride), the
       shortfall is filled by drawing at random from the remaining indices.

    Parameters
    ----------
    n_total:
        Total number of projections in the GT geometry.
    k:
        Number of projections to select.
    rng:
        Random number generator for the random fill step.
    """
    if k > n_total:
        raise ValueError(
            f"Requested {k} projections but only {n_total} are available in the GT geometry."
        )
    if rng is None:
        rng = np.random.default_rng()

    import math
    stride = max(1, math.ceil(n_total / k))
    uniform_indices = np.arange(0, n_total, stride)  # at most k elements
    n_uniform = len(uniform_indices)

    if n_uniform >= k:
        # Exact hit (n_total divisible by k) — truncate just in case.
        return uniform_indices[:k]

    # Fill remaining slots by sampling randomly from non-selected indices.
    used = set(uniform_indices.tolist())
    remaining = np.array([i for i in range(n_total) if i not in used])
    fill = rng.choice(remaining, size=k - n_uniform, replace=False)
    result = np.sort(np.concatenate([uniform_indices, fill]))
    return result.astype(int)


def _build_fsus_geometry_and_projections(
    gt_reconstruction: np.ndarray,
    reference_geometry: Geometry,
    k: int,
    ell_max: int,
) -> Tuple[Geometry, "ProjectionStack"]:
    """Create a Fibonacci-hemisphere geometry with *k* directions and synthesise projections.

    Parameters
    ----------
    gt_reconstruction:
        Ground-truth volume ``(x, y, z, n_coeffs)``.
    reference_geometry:
        Reference geometry used to copy detector / volume settings.
    k:
        Number of projections (Fibonacci points on the upper hemisphere).
    ell_max:
        Maximum spherical harmonic degree.

    Returns
    -------
    new_geometry, projection_stack
    """
    directions = fibonacci_hemisphere(k, upper=True)
    new_geometry, projection_stack = generate_geometry_and_projections(
        reconstruction=gt_reconstruction,
        directions=directions,
        reference_geometry=reference_geometry,
        ell_max=ell_max,
        return_data_container=True,
        copy_from_reference=False,
    )
    return new_geometry, projection_stack


def _build_subset_geometry_and_projections(
    gt_reconstruction: np.ndarray,
    reference_geometry: Geometry,
    indices: np.ndarray,
    ell_max: int,
) -> Tuple[Geometry, "ProjectionStack"]:
    """Extract subset geometry and synthesise projections for SNUS / sparse-SNUS.

    ``_subset_geometry`` copies rotation matrices, j/k offsets, and per-projection
    axes directly from ``reference_geometry`` for the chosen indices — no
    angular-to-rotation conversion is ever performed.  We then forward-project
    the GT reconstruction straight through that exact geometry.

    Parameters
    ----------
    gt_reconstruction:
        Ground-truth volume ``(x, y, z, n_coeffs)``.
    reference_geometry:
        Full GT geometry to copy projections from.
    indices:
        Indices of projections to select.
    ell_max:
        Maximum spherical harmonic degree.

    Returns
    -------
    subset_geometry, projection_stack
    """
    from mumott_al.geometry import create_synthetic_projections

    # _subset_geometry copies every per-projection parameter (rotation matrix,
    # j/k offsets, inner/outer axes) verbatim — the geometry is already exact.
    subset_geometry = _subset_geometry(reference_geometry, indices)

    # Forward-project the GT reconstruction through the exact subset geometry.
    # create_data_container=True returns a ProjectionStack directly.
    projection_stack = create_synthetic_projections(
        reconstruction=gt_reconstruction,
        new_geometry=subset_geometry,
        ell_max=ell_max,
        create_data_container=True,
    )
    return subset_geometry, projection_stack


# ---------------------------------------------------------------------------
# Ground-truth caching (disk)
# ---------------------------------------------------------------------------


def _gt_cache_path(data_path: str) -> Path:
    """Return a deterministic temp-file path for caching the GT reconstruction.

    The cache is keyed by the absolute path of the HDF5 file and its mtime,
    so changes to the input file automatically invalidate the cache.
    """
    abs_path = str(Path(data_path).resolve())
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = 0.0
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
    """Load or compute the GT reconstruction and DataContainer.

    Tries to load a cached reconstruction from disk first; falls back to a
    full mumott reconstruction if none is found.

    Returns
    -------
    gt_reconstruction : np.ndarray
    dc               : DataContainer
    """
    cache_path = _gt_cache_path(data_path)

    dc = DataContainer(str(data_path), nonfinite_replacement_value=0.0)

    if cache_path.exists():
        if verbose:
            print(f"[GT] Loading cached reconstruction from {cache_path}")
        loaded = np.load(cache_path)
        gt_reconstruction = loaded["gt_reconstruction"]
        if verbose:
            print(f"[GT] Loaded shape: {gt_reconstruction.shape}")
        return gt_reconstruction, dc

    if verbose:
        print(
            f"[GT] Computing ground-truth reconstruction from all "
            f"{len(dc.geometry.inner_angles)} projections …"
        )

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
        print(f"[GT] Done in {elapsed:.1f} s.  Shape: {gt_reconstruction.shape}")

    # Cache to disk
    np.savez_compressed(str(cache_path), gt_reconstruction=gt_reconstruction)
    if verbose:
        print(f"[GT] Cached to {cache_path}")

    return gt_reconstruction, dc


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------


def _make_run_name(benchmark: str, k: int, data_path: str) -> str:
    return f"{benchmark}_{k}proj_{Path(data_path).stem}"


def _check_wandb_run_exists(
    project: str,
    benchmark: str,
    data_path: str,
    num_projections: int,
    entity: Optional[str] = None,
) -> bool:
    """Return True if a *finished* wandb run with matching config already exists."""
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed — skipping duplicate check.")
        return False

    try:
        api = wandb.Api(timeout=30)
        project_path = f"{entity}/{project}" if entity else project
        runs = api.runs(
            path=project_path,
            filters={
                "config.benchmark": benchmark,
                "config.data_path": str(Path(data_path).resolve()),
                "config.num_projections": num_projections,
                "state": "finished",
            },
        )
        run_list = list(runs)
        if run_list:
            print(
                f"[wandb] Found existing finished run(s) for "
                f"benchmark={benchmark}, k={num_projections}, "
                f"data={Path(data_path).name}:"
            )
            for r in run_list:
                print(f"         id={r.id}  name={r.name}")
            return True
        return False
    except Exception as exc:
        print(f"[wandb] Could not query existing runs ({exc}). Proceeding.")
        return False


def _log_metrics_to_wandb(
    run,
    benchmark: str,
    k: int,
    metrics: "ReconstructionComparisonResult",
) -> None:
    """Log all metrics from *metrics* to the active wandb *run*."""
    log_dict = {
        "benchmark": benchmark,
        "num_projections": k,
        # Coefficient-based
        "mse_coefficients": metrics.mse_coefficients,
        "mae_coefficients": metrics.mae_coefficients,
        "normalized_mse_coefficients": metrics.normalized_mse_coefficients,
    }

    if metrics.projection_metrics is not None:
        pm = metrics.projection_metrics
        log_dict.update(
            {
                "projection/mse_global": pm.mse_global,
                "projection/mae_global": pm.mae_global,
                "projection/psnr_global_dB": pm.psnr_global,
                "projection/ssim_global": pm.ssim_global,
            }
        )
        for ch_i, (mse_c, mae_c, psnr_c, ssim_c) in enumerate(
            zip(pm.mse_per_channel, pm.mae_per_channel, pm.psnr_per_channel, pm.ssim_per_channel)
        ):
            log_dict[f"projection/mse_ch{ch_i:02d}"] = mse_c
            log_dict[f"projection/mae_ch{ch_i:02d}"] = mae_c
            log_dict[f"projection/psnr_ch{ch_i:02d}_dB"] = psnr_c
            log_dict[f"projection/ssim_ch{ch_i:02d}"] = ssim_c

    if metrics.orientation_metrics is not None:
        om = metrics.orientation_metrics
        log_dict.update(
            {
                "orientation/cosine_similarity_mean": om.cosine_similarity_mean,
                "orientation/cosine_similarity_std": om.cosine_similarity_std,
                "orientation/angular_error_mean_deg": om.angular_error_mean_degrees,
                "orientation/angular_error_std_deg": om.angular_error_std_degrees,
                "orientation/valid_voxels": om.valid_voxels,
            }
        )

    if metrics.real_space_metrics is not None:
        rs = metrics.real_space_metrics
        log_dict.update(
            {
                "real_space/mse": rs.mse,
                "real_space/mae": rs.mae,
                "real_space/psnr_dB": rs.psnr,
                "real_space/ssim": rs.ssim,
                "real_space/n_directions": rs.n_directions,
            }
        )

    run.log(log_dict)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def run_benchmark(
    benchmark: str,
    data_path: str,
    num_projections: int,
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
    """Run one sparse-sampling benchmark and log results to wandb.

    Parameters
    ----------
    benchmark:
        One of ``"SNUS"``, ``"sparse_SNUS"``, ``"FSUS"``.
    data_path:
        Path to the HDF5 file used as the GT DataContainer.
    num_projections:
        Number of projections to use (sparsity level *k*).
    ell_max:
        Maximum spherical harmonic degree.
    maxiter:
        Maximum LBFGS optimisation iterations.
    regularization_weight:
        Laplacian regularisation weight.
    wandb_project:
        Name of the wandb project.
    wandb_entity:
        wandb entity / team name (optional).
    force_cpu:
        Disable CUDA even when a GPU is available.
    real_space_resolution_deg:
        Angular resolution (in degrees) for the real-space SH evaluation grid.
    seed:
        Random seed (used for Sparse-SNUS random fill and for torch).
    verbose:
        Print progress information.
    skip_wandb_check:
        If True, skip the "already exists" check and always run.
    """
    benchmark = benchmark.strip()
    if benchmark not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. Choose from: {BENCHMARKS}."
        )

    data_path = str(Path(data_path).resolve())
    k = num_projections
    rng = np.random.default_rng(seed)
    use_cuda = torch.cuda.is_available() and not force_cpu

    if verbose:
        print("=" * 70)
        print(f"Sparse Sampling Benchmark")
        print(f"  benchmark       : {benchmark}")
        print(f"  data_path       : {data_path}")
        print(f"  num_projections : {k}")
        print(f"  ell_max         : {ell_max}")
        print(f"  maxiter         : {maxiter}")
        print(f"  reg_weight      : {regularization_weight}")
        print(f"  device          : {'CUDA' if use_cuda else 'CPU'}")
        print("=" * 70)

    # ------------------------------------------------------------------ [0]
    # wandb duplicate check
    # ------------------------------------------------------------------
    if not skip_wandb_check:
        if verbose:
            print("\n[0/5] Checking wandb for existing results …")
        already_done = _check_wandb_run_exists(
            project=wandb_project,
            benchmark=benchmark,
            data_path=data_path,
            num_projections=k,
            entity=wandb_entity,
        )
        if already_done:
            if verbose:
                print("[0/5] Results already exist. Skipping computation.")
            return

    # ------------------------------------------------------------------ [1]
    # Load / compute ground-truth reconstruction
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/5] Ground-truth reconstruction …")

    gt_reconstruction, dc = _load_gt_reconstruction(
        data_path=data_path,
        ell_max=ell_max,
        maxiter=maxiter,
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=verbose,
    )
    reference_geometry = dc.geometry
    n_total = len(reference_geometry.inner_angles)

    if verbose:
        print(f"[1/5] GT shape={gt_reconstruction.shape}  "
              f"total projections={n_total}")

    # ------------------------------------------------------------------ [2]
    # Build k-projection geometry and synthesise projections
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[2/5] Building {benchmark} geometry ({k} projections) …")

    if benchmark == "SNUS":
        indices = _select_snus_indices(n_total, k)
        if verbose:
            print(f"  SNUS: first {k} projection indices = {indices[:5].tolist()} …")
        new_geometry, projection_stack = _build_subset_geometry_and_projections(
            gt_reconstruction, reference_geometry, indices, ell_max
        )

    elif benchmark == "sparse_SNUS":
        indices = _select_sparse_snus_indices(n_total, k, rng=rng)
        if verbose:
            stride = max(1, n_total // k)
            print(f"  sparse-SNUS: stride={stride}, selected indices "
                  f"(first 5) = {indices[:5].tolist()} …")
        new_geometry, projection_stack = _build_subset_geometry_and_projections(
            gt_reconstruction, reference_geometry, indices, ell_max
        )

    else:  # FSUS
        if verbose:
            print(f"  FSUS: Fibonacci hemisphere with {k} directions …")
        new_geometry, projection_stack = _build_fsus_geometry_and_projections(
            gt_reconstruction, reference_geometry, k, ell_max
        )

    # Build DataContainer from synthetic projections
    synthetic_dc = create_synthetic_data_container(
        geometry=new_geometry,
        projection_stack=projection_stack,
        reference_dc=dc,
    )

    if verbose:
        print(
            f"[2/5] Synthetic DC: {len(synthetic_dc.projections)} projections, "
            f"data shape {synthetic_dc.data.shape}"
        )

    # ------------------------------------------------------------------ [3]
    # Reconstruct from synthetic projections
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[3/5] Reconstructing from {k} synthetic projections …")

    t0 = time.perf_counter()
    sparse_reconstruction = _perform_reconstruction(
        dc=synthetic_dc,
        ell_max=ell_max,
        maxiter=maxiter,
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"[3/5] Reconstruction done in {elapsed:.1f} s.  "
              f"Shape: {sparse_reconstruction.shape}")

    # ------------------------------------------------------------------ [4]
    # Compute metrics vs GT
    # ------------------------------------------------------------------
    if verbose:
        print("\n[4/5] Computing metrics …")

    metrics = compare_reconstructions(
        reconstruction_pred=sparse_reconstruction,
        reconstruction_gt=gt_reconstruction,
        geometry=reference_geometry,  # used only as reference for Fibonacci eval grid
        ell_max=ell_max,
        mask=None,
        weights=None,  # no bad pixels on Fibonacci eval grid
        compute_projection_metrics=True,
        compute_orientation_metrics=True,
        compute_real_space_metrics=True,
        real_space_resolution_in_degrees=real_space_resolution_deg,
        real_space_half_sphere=True,
        n_fibonacci_eval_projections=1000,  # unbiased held-out eval geometry
        verbose=verbose,
    )

    if verbose:
        print_comparison_results(metrics)

    # ------------------------------------------------------------------ [5]
    # Log to wandb
    # ------------------------------------------------------------------
    if verbose:
        print("\n[5/5] Logging to wandb …")

    try:
        import wandb

        run_name = _make_run_name(benchmark, k, data_path)
        init_kwargs = dict(
            project=wandb_project,
            name=run_name,
            config={
                "benchmark": benchmark,
                "data_path": data_path,
                "data_name": Path(data_path).stem,
                "num_projections": k,
                "ell_max": ell_max,
                "maxiter": maxiter,
                "regularization_weight": regularization_weight,
                "real_space_resolution_deg": real_space_resolution_deg,
                "use_cuda": use_cuda,
                "seed": seed,
            },
            tags=[benchmark, f"k{k}", Path(data_path).stem],
        )
        if wandb_entity:
            init_kwargs["entity"] = wandb_entity

        with wandb.init(**init_kwargs) as run:
            _log_metrics_to_wandb(run, benchmark, k, metrics)
            if verbose:
                print(f"[5/5] Logged to wandb run: {run.url}")

    except ImportError:
        print("[wandb] wandb not installed — skipping logging.")
    except Exception as exc:
        print(f"[wandb] Logging failed: {exc}")

    if verbose:
        print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mumott_al.benchmark_sparse_sampling",
        description=(
            "Sparse sampling benchmark for tensor tomography.  "
            "Evaluates SNUS, sparse_SNUS, or FSUS against the full-data GT "
            "reconstruction and logs metrics to wandb."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARKS),
        help="Sampling strategy to evaluate.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to the input HDF5 DataContainer.",
    )
    parser.add_argument(
        "--num-projections",
        type=int,
        required=True,
        metavar="K",
        help="Number of sparse projections (sparsity level k).",
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
        help="Random seed (used for sparse-SNUS random fill and torch).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CUDA even when a GPU is available.",
    )
    parser.add_argument(
        "--skip-wandb-check",
        action="store_true",
        help="Skip the wandb duplicate-run check and always run the benchmark.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser


def main() -> None:
    """Entry point for CLI / ``python -m mumott_al.benchmark_sparse_sampling``."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    run_benchmark(
        benchmark=args.benchmark,
        data_path=args.data_path,
        num_projections=args.num_projections,
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
