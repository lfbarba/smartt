"""
Synthetic data processing module for generating tensor tomography reconstruction datasets.

This module provides functionality to load real projection data, compute a ground truth
reconstruction from all projections, generate new synthetic geometries using various
sampling strategies (e.g. Fibonacci hemisphere, random sphere), forward-propagate the
ground truth to obtain synthetic projections, perform sparse reconstructions from those
synthetic projections, and save the dataset to HDF5 for machine learning training.

The key difference from `smartt.data_processing` is that projections are **not** subsampled
from the original dataset.  Instead, entirely new acquisition geometries are synthesised
and the corresponding projections are computed by forward-propagating the ground-truth
reconstruction.  This decouples the reconstruction quality from the original angular
sampling grid and allows arbitrary geometries to be explored.
"""

import copy
import h5py
import numpy as np
import os
import sys
import torch
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict

from mumott.data_handling import DataContainer
from mumott.methods.basis_sets import SphericalHarmonics
from mumott.methods.projectors import SAXSProjector, SAXSProjectorCUDA
from mumott.methods.residual_calculators import GradientResidualCalculator
from mumott.optimization.loss_functions import SquaredLoss
from mumott.optimization.optimizers import LBFGS
from mumott.optimization.regularizers import Laplacian

from .geometry import (
    fibonacci_hemisphere,
    cartesian_to_spherical,
    create_geometry_from_directions,
    create_synthetic_projections,
    create_synthetic_data_container,
    generate_geometry_and_projections,
)


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

SAMPLING_STRATEGIES = ("fibonacci", "random_hemisphere", "random_sphere")


def _sample_directions(
    n_points: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate *n_points* unit-vector directions according to *strategy*.

    Parameters
    ----------
    n_points : int
        Number of projection directions to generate.
    strategy : str
        One of ``"fibonacci"``, ``"random_hemisphere"``, ``"random_sphere"``.

        * **fibonacci** – quasi-uniformly distributed on the upper hemisphere
          via the Fibonacci spiral method.
        * **random_hemisphere** – uniformly distributed at random on the upper
          hemisphere (z >= 0).
        * **random_sphere** – uniformly distributed at random on the full unit
          sphere.
    rng : np.random.Generator
        Seeded random-number generator (used only for random strategies).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 3)`` with unit-vector directions.
    """
    strategy = strategy.lower()
    if strategy == "fibonacci":
        return fibonacci_hemisphere(n_points, upper=True)

    elif strategy == "random_hemisphere":
        # Uniform distribution on the upper hemisphere
        # Sample on full sphere and reflect below-equator points
        dirs = _uniform_sphere_directions(n_points, rng)
        dirs[:, 2] = np.abs(dirs[:, 2])  # fold to upper hemisphere
        return dirs

    elif strategy == "random_sphere":
        return _uniform_sphere_directions(n_points, rng)

    else:
        raise ValueError(
            f"Unknown sampling strategy '{strategy}'. "
            f"Choose one of: {SAMPLING_STRATEGIES}."
        )


def _uniform_sphere_directions(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Return *n_points* uniformly distributed unit vectors on the unit sphere."""
    # Marsaglia (1972) method
    vecs = rng.standard_normal((n_points, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# Internal helpers shared with smartt.data_processing
# ---------------------------------------------------------------------------

def _perform_reconstruction(
    dc: DataContainer,
    ell_max: int,
    maxiter: int,
    regularization_weight: float,
    use_cuda: bool,
    verbose: bool = False,
) -> np.ndarray:
    """
    Perform a single spherical harmonic reconstruction.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(*volume_shape, num_coeffs)``.
    """
    basis_set = SphericalHarmonics(ell_max=ell_max)

    if use_cuda:
        projector = SAXSProjectorCUDA(dc.geometry)
    else:
        projector = SAXSProjector(dc.geometry)

    residual_calculator = GradientResidualCalculator(
        data_container=dc,
        basis_set=basis_set,
        projector=projector,
    )

    loss_function = SquaredLoss(residual_calculator)
    regularizer = Laplacian()
    loss_function.add_regularizer(
        name="laplacian",
        regularizer=regularizer,
        regularization_weight=regularization_weight,
    )

    optimizer = LBFGS(loss_function, maxiter=maxiter)

    if verbose:
        print("    Running LBFGS optimisation …")

    results = optimizer.optimize()

    # Extract the array regardless of the exact dict key used by mumott
    if isinstance(results, dict):
        for key in ("x", "reconstruction", "coefficients"):
            if key in results:
                reconstruction = results[key]
                break
        else:
            for value in results.values():
                if hasattr(value, "shape"):
                    reconstruction = value
                    break
            else:
                raise ValueError(
                    f"Could not locate reconstruction in results dict. Keys: {list(results.keys())}"
                )
    elif hasattr(results, "reconstruction"):
        reconstruction = results.reconstruction
    else:
        reconstruction = results

    if torch.is_tensor(reconstruction):
        reconstruction = reconstruction.cpu().numpy()

    return np.asarray(reconstruction, dtype=np.float32)


def _get_h5_files(data_path: Path) -> List[Tuple[Path, str]]:
    """Return list of ``(file_path, file_identifier)`` tuples from a file or directory."""
    if data_path.is_file():
        return [(data_path, data_path.stem)]
    elif data_path.is_dir():
        h5_files: List[Path] = []
        for ext in ("*.h5", "*.hdf5", "*.H5", "*.HDF5"):
            h5_files.extend(data_path.glob(ext))
        h5_files = sorted(set(h5_files))
        if not h5_files:
            raise ValueError(f"No HDF5 files found in directory: {data_path}")
        return [(f, f.stem) for f in h5_files]
    else:
        raise ValueError(f"Path does not exist: {data_path}")


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_synthetic_reconstruction_dataset(
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    ell_max: int = 8,
    num_projections_min: int = 20,
    num_projections_max: int = 80,
    num_projections_step: int = 10,
    num_iterations: int = 100,
    maxiter: int = 20,
    regularization_weight: float = 1.0,
    sampling_strategy: str = "fibonacci",
    seed: Optional[int] = None,
    force_cpu: bool = False,
    verbose: bool = True,
) -> None:
    """
    Generate a dataset of tensor tomography reconstructions from synthetic projections.

    For every input HDF5 file the function will:

    1. Compute a **ground-truth reconstruction** using all available real projections.
    2. For each of the *num_iterations* samplings:

       a. Randomly pick a projection count from
          ``range(num_projections_min, num_projections_max + 1, num_projections_step)``.
       b. Draw that many new projection directions using *sampling_strategy*.
       c. Forward-propagate the ground truth through the synthetic geometry to obtain
          synthetic measurements.
       d. Reconstruct the volume from those synthetic measurements.
       e. Write the reconstruction and the corresponding directions to disk immediately
          so that memory usage stays bounded.

    The output HDF5 layout is compatible with the one produced by
    ``smartt.data_processing.generate_reconstruction_dataset``, with the following
    differences/extensions:

    * ``directions`` – ``(total_samples, num_projections_max, 3)`` float32 array,
      padded with zeros beyond the actual count for each sample.
    * ``num_projections_per_sample`` – ``(total_samples,)`` int32 array recording
      the actual number of projections used in each reconstruction.
    * ``sampling_strategy`` – scalar string attribute that records the strategy name.

    Parameters
    ----------
    data_path : str or Path
        Path to a single HDF5 file **or** a directory of HDF5 files produced by
        the acquisition pipeline (same format as accepted by ``DataContainer``).
    output_path : str or Path
        Destination HDF5 file.  Parent directories are created if needed.
    ell_max : int, default 8
        Maximum spherical harmonic degree.  Determines the number of coefficients:
        ``(ell_max + 1) ** 2``.
    num_projections_min : int, default 20
        Minimum number of synthetic projection directions (inclusive).
    num_projections_max : int, default 80
        Maximum number of synthetic projection directions (inclusive).
    num_projections_step : int, default 10
        Step size between candidate projection counts.  Each iteration randomly
        draws one value from
        ``np.arange(num_projections_min, num_projections_max + 1, num_projections_step)``.
    num_iterations : int, default 100
        Number of independent direction samplings to perform per input file.
    maxiter : int, default 20
        Maximum LBFGS iterations for each reconstruction.
    regularization_weight : float, default 1.0
        Weight for the Laplacian regularisation term.
    sampling_strategy : str, default ``"fibonacci"``
        Strategy for generating new projection directions.  One of:

        * ``"fibonacci"``       – quasi-uniform on the upper hemisphere.
        * ``"random_hemisphere"`` – random uniform on the upper hemisphere.
        * ``"random_sphere"``   – random uniform on the full sphere.
    seed : int, optional
        Global random seed for reproducibility.
    force_cpu : bool, default False
        Disable CUDA even when a GPU is available.
    verbose : bool, default True
        Print progress information.

    Returns
    -------
    None
        All results are written to *output_path*.

    HDF5 structure
    --------------
    ::

        /reconstructions             (total_samples, *vol_shape, num_coeffs)   float32
        /ground_truths               (num_files, *vol_shape, num_coeffs)        float32
        /file_identifiers            (num_files,)                               str
        /reconstruction_to_gt_index  (total_samples,)                           int32
        /directions                  (total_samples, num_projections_max, 3)    float32
        /num_projections_per_sample  (total_samples,)                           int32
        /num_projections_min         scalar                                     int
        /num_projections_max         scalar                                     int
        /num_projections_step        scalar                                     int
        /ell_max                     scalar                                     int
        /num_iterations              scalar                                     int
        /maxiter                     scalar                                     int
        /regularization_weight       scalar                                     float
        /sampling_strategy           scalar                                     str

    Examples
    --------
    >>> from mumott_al import generate_synthetic_reconstruction_dataset
    >>> generate_synthetic_reconstruction_dataset(
    ...     data_path="data/frogbone/dataset_qbin_0009.h5",
    ...     output_path="training/synthetic_dataset.h5",
    ...     ell_max=8,
    ...     num_projections_min=20,
    ...     num_projections_max=80,
    ...     num_projections_step=10,
    ...     num_iterations=50,
    ...     sampling_strategy="fibonacci",
    ...     seed=42,
    ... )
    """
    data_path = Path(data_path)
    output_path = Path(output_path)

    # ------------------------------------------------------------------ setup
    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available() and not force_cpu
    if verbose:
        if force_cpu:
            print("Forcing CPU usage.")
        print(f"Device: {'CUDA' if use_cuda else 'CPU'}")
        print(f"Sampling strategy: {sampling_strategy}")

    # -------------------------------------------- discover input files
    h5_files = _get_h5_files(data_path)
    if verbose:
        print(f"\nFound {len(h5_files)} HDF5 file(s):")
        for fp, fid in h5_files:
            print(f"  [{fid}]  {fp}")

    total_samples = len(h5_files) * num_iterations

    # Build the candidate projection counts for random selection each iteration
    proj_counts = np.arange(num_projections_min, num_projections_max + 1, num_projections_step)
    if len(proj_counts) == 0:
        raise ValueError(
            f"Empty projection count range: min={num_projections_min}, "
            f"max={num_projections_max}, step={num_projections_step}"
        )
    if verbose:
        print(f"Projection counts per iteration: {proj_counts.tolist()}")

    # ------- determine volume shape and actual num_coeffs from a dummy recon
    if verbose:
        print("\nProbing volume shape from the first file …")

    dc_probe = DataContainer(str(h5_files[0][0]), nonfinite_replacement_value=0)
    probe_recon = _perform_reconstruction(
        dc=dc_probe,
        ell_max=ell_max,
        maxiter=1,
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=False,
    )
    volume_shape: Tuple[int, ...] = probe_recon.shape[:3]
    num_coeffs: int = int(probe_recon.shape[3]) if probe_recon.ndim >= 4 else 1
    del probe_recon, dc_probe

    if verbose:
        print(f"Volume shape : {volume_shape}")
        print(f"Coefficients : {num_coeffs}  (ell_max={ell_max}, expected={(ell_max+1)**2})")

    # ----------------------------------------------- create output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\nCreating output file: {output_path}")
        print(
            f"Pre-allocating for {total_samples} reconstructions "
            f"and {len(h5_files)} ground-truth(s)."
        )

    with h5py.File(output_path, "w") as f:
        recon_dset = f.create_dataset(
            "reconstructions",
            shape=(total_samples, *volume_shape, num_coeffs),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        gt_dset = f.create_dataset(
            "ground_truths",
            shape=(len(h5_files), *volume_shape, num_coeffs),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        directions_dset = f.create_dataset(
            "directions",
            shape=(total_samples, int(num_projections_max), 3),
            dtype=np.float32,
        )
        n_proj_dset = f.create_dataset(
            "num_projections_per_sample",
            shape=(total_samples,),
            dtype=np.int32,
        )
        mapping_dset = f.create_dataset(
            "reconstruction_to_gt_index",
            shape=(total_samples,),
            dtype=np.int32,
        )

        file_identifiers_list: List[str] = []
        sparse_idx = 0

        # ================================================ iterate over files
        for file_idx, (file_path, file_identifier) in enumerate(h5_files):
            if verbose:
                print(f"\n{'='*70}")
                print(f"File {file_idx + 1}/{len(h5_files)}: {file_identifier}")
                print(f"{'='*70}")

            # ---- ground-truth reconstruction from all real projections ----
            if verbose:
                print("  Computing ground-truth reconstruction …")

            dc_gt = DataContainer(str(file_path), nonfinite_replacement_value=0)
            ground_truth = _perform_reconstruction(
                dc=dc_gt,
                ell_max=ell_max,
                maxiter=maxiter,
                regularization_weight=regularization_weight,
                use_cuda=use_cuda,
                verbose=verbose,
            )
            # Keep a reference geometry for synthetic geometry creation
            reference_geometry = dc_gt.geometry

            # Write GT immediately
            gt_dset[file_idx] = ground_truth
            file_identifiers_list.append(file_identifier)

            if verbose:
                print(f"  Ground truth written to index {file_idx}  shape={ground_truth.shape}")

            # ---- iterate synthetic samplings --------------------------------
            if verbose:
                print(f"\n  Generating {num_iterations} synthetic reconstructions …")

            for iteration in range(num_iterations):
                if verbose:
                    print(
                        f"\n    Iteration {iteration + 1}/{num_iterations} "
                        f"for {file_identifier}"
                    )

                # 1. Pick number of projections for this iteration and sample directions
                n_proj = int(rng.choice(proj_counts))
                directions = _sample_directions(n_proj, sampling_strategy, rng)

                if verbose:
                    inner_deg, outer_deg = cartesian_to_spherical(directions)
                    print(
                        f"    Sampled {n_proj} directions via '{sampling_strategy}' "
                        f"(inner ∈ [{np.degrees(inner_deg.min()):.1f}°, "
                        f"{np.degrees(inner_deg.max()):.1f}°], "
                        f"outer ∈ [{np.degrees(outer_deg.min()):.1f}°, "
                        f"{np.degrees(outer_deg.max()):.1f}°])"
                    )

                # 2. Create synthetic geometry + forward-projected DataContainer
                if verbose:
                    print("    Creating synthetic geometry and forward projections …")

                new_geometry, projection_stack = generate_geometry_and_projections(
                    reconstruction=ground_truth,
                    directions=directions,
                    reference_geometry=reference_geometry,
                    ell_max=ell_max,
                    return_data_container=True,
                )
                synthetic_dc = create_synthetic_data_container(
                    geometry=new_geometry,
                    projection_stack=projection_stack,
                    reference_dc=dc_gt,
                )

                if verbose:
                    print(
                        f"    Synthetic DC ready: {len(synthetic_dc.projections)} projections, "
                        f"data shape {synthetic_dc.data.shape}"
                    )

                # 3. Reconstruct from synthetic projections
                if verbose:
                    print("    Reconstructing from synthetic projections …")

                reconstruction = _perform_reconstruction(
                    dc=synthetic_dc,
                    ell_max=ell_max,
                    maxiter=maxiter,
                    regularization_weight=regularization_weight,
                    use_cuda=use_cuda,
                    verbose=verbose,
                )

                # 4. Write to HDF5 immediately
                recon_dset[sparse_idx] = reconstruction
                # Pad directions to num_projections_max width
                padded = np.zeros((num_projections_max, 3), dtype=np.float32)
                padded[:n_proj] = directions.astype(np.float32)
                directions_dset[sparse_idx] = padded
                n_proj_dset[sparse_idx] = n_proj
                mapping_dset[sparse_idx] = file_idx

                if verbose:
                    print(f"    Reconstruction written to index {sparse_idx}")

                # Clean up
                del reconstruction, synthetic_dc, projection_stack, new_geometry
                sparse_idx += 1

            # Done with this file – release GT
            del ground_truth, dc_gt

        # ----------------------------------------- save metadata datasets
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("file_identifiers", data=file_identifiers_list, dtype=dt)

        f.create_dataset("num_projections_min", data=num_projections_min)
        f.create_dataset("num_projections_max", data=num_projections_max)
        f.create_dataset("num_projections_step", data=num_projections_step)
        f.create_dataset("ell_max", data=ell_max)
        f.create_dataset("num_iterations", data=num_iterations)
        f.create_dataset("maxiter", data=maxiter)
        f.create_dataset("regularization_weight", data=regularization_weight)
        f.create_dataset("sampling_strategy", data=sampling_strategy)

        f.attrs["data_path"] = str(data_path)
        f.attrs["num_coefficients"] = num_coeffs
        f.attrs["volume_shape"] = list(volume_shape)
        f.attrs["num_files"] = len(h5_files)
        f.attrs["total_sparse_reconstructions"] = sparse_idx
        f.attrs["sampling_strategy"] = sampling_strategy

    if verbose:
        print(f"\n{'='*70}")
        print("Done.")
        print(f"  {sparse_idx} sparse reconstructions")
        print(f"  {len(file_identifiers_list)} ground-truth reconstruction(s)")
        print(f"  Volume shape : {volume_shape}")
        print(f"  Output       : {output_path}")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ``mumott-al-synthetic-dataset`` command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mumott-al-synthetic-dataset",
        description=(
            "Generate a tensor tomography dataset whose sparse reconstructions "
            "are computed from **synthetic** projections rather than from random "
            "subsets of the original measured projections."
        ),
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to input HDF5 file (or directory of HDF5 files) with real projection data.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Destination HDF5 file for the generated dataset.",
    )
    parser.add_argument(
        "--ell-max",
        type=int,
        default=8,
        metavar="N",
        help="Maximum spherical harmonic degree (default: 8).",
    )
    parser.add_argument(
        "--num-projections-min",
        type=int,
        default=20,
        metavar="N",
        help="Minimum number of synthetic projection directions (default: 20).",
    )
    parser.add_argument(
        "--num-projections-max",
        type=int,
        default=80,
        metavar="N",
        help="Maximum number of synthetic projection directions (default: 80).",
    )
    parser.add_argument(
        "--num-projections-step",
        type=int,
        default=10,
        metavar="N",
        help="Step between candidate projection counts (default: 10).",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        metavar="N",
        help="Number of independent samplings per input file (default: 100).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        metavar="N",
        help="Maximum LBFGS iterations per reconstruction (default: 20).",
    )
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=1.0,
        metavar="W",
        help="Laplacian regularisation weight (default: 1.0).",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="fibonacci",
        choices=list(SAMPLING_STRATEGIES),
        help=(
            "Strategy for generating new projection directions "
            "(default: fibonacci)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed for reproducibility (default: none).",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CUDA even when a GPU is available.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    generate_synthetic_reconstruction_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        ell_max=args.ell_max,
        num_projections_min=args.num_projections_min,
        num_projections_max=args.num_projections_max,
        num_projections_step=args.num_projections_step,
        num_iterations=args.num_iterations,
        maxiter=args.maxiter,
        regularization_weight=args.regularization_weight,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
        force_cpu=args.force_cpu,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
