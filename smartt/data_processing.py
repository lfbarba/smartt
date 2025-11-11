"""
Data processing module for generating tensor tomography reconstruction datasets.

This module provides functionality to load projection data, randomly sample subsets,
perform mumott reconstructions, and save the results for machine learning training.
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


def _perform_reconstruction(
    dc: DataContainer,
    ell_max: int,
    maxiter: int,
    regularization_weight: float,
    use_cuda: bool,
    verbose: bool = False
) -> np.ndarray:
    """
    Perform a single spherical harmonic reconstruction.
    
    Parameters
    ----------
    dc : DataContainer
        The data container with projections.
    ell_max : int
        Maximum degree for spherical harmonics expansion.
    maxiter : int
        Maximum number of iterations for the LBFGS optimizer.
    regularization_weight : float
        Weight for the Laplacian regularization term.
    use_cuda : bool
        Whether to use CUDA for computation.
    verbose : bool, default=False
        Whether to print progress information.
    
    Returns
    -------
    np.ndarray
        The reconstruction as a numpy array of shape (*volume_shape, num_coeffs).
    """
    basis_set = SphericalHarmonics(ell_max=ell_max)
    
    # Create projector (CUDA or CPU)
    if use_cuda:
        projector = SAXSProjectorCUDA(dc.geometry)
    else:
        projector = SAXSProjector(dc.geometry)
    
    residual_calculator = GradientResidualCalculator(
        data_container=dc,
        basis_set=basis_set,
        projector=projector
    )
    
    loss_function = SquaredLoss(residual_calculator)
    regularizer = Laplacian()
    loss_function.add_regularizer(
        name='laplacian',
        regularizer=regularizer,
        regularization_weight=regularization_weight
    )
    
    optimizer = LBFGS(loss_function, maxiter=maxiter)
    
    # Run optimization
    if verbose:
        print("Running optimization...")
    results = optimizer.optimize()
    
    # Extract reconstruction
    if isinstance(results, dict):
        if 'x' in results:
            reconstruction = results['x']
        elif 'reconstruction' in results:
            reconstruction = results['reconstruction']
        elif 'coefficients' in results:
            reconstruction = results['coefficients']
        else:
            for key, value in results.items():
                if isinstance(value, (np.ndarray, torch.Tensor)) or hasattr(value, 'shape'):
                    reconstruction = value
                    if verbose:
                        print(f"Using key '{key}' as reconstruction")
                    break
            else:
                raise ValueError(f"Could not find reconstruction in results. Keys: {list(results.keys())}")
    elif hasattr(results, 'reconstruction'):
        reconstruction = results.reconstruction
    else:
        reconstruction = results
    
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(reconstruction):
        reconstruction = reconstruction.cpu().numpy()
    
    # Ensure it's a numpy array
    if not isinstance(reconstruction, np.ndarray):
        reconstruction = np.asarray(reconstruction)
    
    return reconstruction.astype(np.float32)


def _get_h5_files(data_path: Path) -> List[Tuple[Path, str]]:
    """
    Get list of HDF5 files from path (single file or directory).
    
    Parameters
    ----------
    data_path : Path
        Path to a single HDF5 file or directory containing HDF5 files.
    
    Returns
    -------
    List[Tuple[Path, str]]
        List of tuples containing (file_path, file_identifier).
    """
    if data_path.is_file():
        # Single file
        return [(data_path, data_path.stem)]
    elif data_path.is_dir():
        # Directory with multiple files
        h5_files = []
        for ext in ['*.h5', '*.hdf5', '*.H5', '*.HDF5']:
            h5_files.extend(data_path.glob(ext))
        h5_files = sorted(set(h5_files))  # Remove duplicates and sort
        if not h5_files:
            raise ValueError(f"No HDF5 files found in directory: {data_path}")
        return [(f, f.stem) for f in h5_files]
    else:
        raise ValueError(f"Path does not exist: {data_path}")


def generate_reconstruction_dataset(
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    ell_max: int = 8,
    num_projections: int = 60,
    num_iterations: int = 100,
    maxiter: int = 20,
    regularization_weight: float = 1.0,
    seed: Optional[int] = None,
    force_cpu: bool = False,
    verbose: bool = True
) -> None:
    """
    Generate a dataset of tensor tomography reconstructions from random projection subsets.
    
    This function loads projection dataset(s), randomly samples subsets of projections,
    performs spherical harmonic reconstructions using mumott, and saves the results
    to an HDF5 file for machine learning training. It also computes ground truth 
    reconstructions using all available projections for each input file.
    
    Parameters
    ----------
    data_path : str or Path
        Path to either:
        - A single HDF5 file containing projection data, or
        - A directory containing multiple HDF5 files with the same format.
    output_path : str or Path
        Path where the output HDF5 file will be saved.
    ell_max : int, default=8
        Maximum degree for spherical harmonics expansion. This determines the number
        of coefficients: (ell_max + 1)^2.
    num_projections : int, default=60
        Number of projections to randomly sample for each sparse reconstruction.
    num_iterations : int, default=100
        Number of different random samplings to perform per input file.
    maxiter : int, default=20
        Maximum number of iterations for the LBFGS optimizer.
    regularization_weight : float, default=1.0
        Weight for the Laplacian regularization term.
    seed : int, optional
        Random seed for reproducibility. If None, results will be non-deterministic.
    force_cpu : bool, default=False
        If True, force CPU usage even if CUDA is available. Use this to avoid
        CUDA compatibility issues.
    verbose : bool, default=True
        Whether to print progress information.
    
    Returns
    -------
    None
        Results are saved to output_path as an HDF5 file with the following structure:
        - 'reconstructions': Dataset of shape (total_samples, *volume_shape, num_coeffs)
          containing sparse reconstructions from randomly sampled projections
        - 'ground_truths': Dataset of shape (num_files, *volume_shape, num_coeffs)
          containing full reconstructions using all projections
        - 'file_identifiers': Dataset of strings identifying source files
        - 'reconstruction_to_gt_index': Dataset mapping each reconstruction to its ground truth
        - 'num_projections': Scalar with number of projections used for sparse reconstructions
        - 'ell_max': Scalar with the ell_max parameter
        - 'projection_indices': Dataset of shape (total_samples, num_projections)
    
    Examples
    --------
    >>> from smartt import generate_reconstruction_dataset
    >>> # Single file
    >>> generate_reconstruction_dataset(
    ...     data_path='trabecular_bone_9.h5',
    ...     output_path='training_data.h5',
    ...     ell_max=8,
    ...     num_projections=60,
    ...     num_iterations=100,
    ...     seed=42
    ... )
    >>> 
    >>> # Multiple files in a directory
    >>> generate_reconstruction_dataset(
    ...     data_path='data/bone_scans/',
    ...     output_path='training_data.h5',
    ...     ell_max=8,
    ...     num_projections=60,
    ...     num_iterations=100,
    ...     seed=42
    ... )
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available() and not force_cpu
    if verbose:
        if force_cpu:
            print("Forcing CPU usage (--force-cpu flag set)")
        print(f"Using {'CUDA' if use_cuda else 'CPU'} for computations")
    
    # Get list of HDF5 files to process
    h5_files = _get_h5_files(data_path)
    if verbose:
        print(f"\nFound {len(h5_files)} HDF5 file(s) to process:")
        for file_path, file_id in h5_files:
            print(f"  - {file_id}: {file_path}")
    
    if verbose:
        print(f"\nSpherical harmonics with ell_max={ell_max}")
        print(f"Sampling {num_projections} projections per sparse reconstruction")
        print(f"Running {num_iterations} iterations per file")
    
    # Calculate total samples and get volume shape from first file
    total_samples = len(h5_files) * num_iterations
    dc_template = DataContainer(str(h5_files[0][0]), nonfinite_replacement_value=0)
    total_projections_template = len(dc_template.projections)
    
    # Get volume shape by doing a test reconstruction
    if verbose:
        print(f"\nDetermining volume shape from first file...")
    test_recon = _perform_reconstruction(
        dc=dc_template,
        ell_max=ell_max,
        maxiter=1,  # Just 1 iteration to get shape
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=False
    )
    volume_shape = test_recon.shape[:3]

    # Derive number of coefficients from the reconstruction returned by mumott.
    # Sometimes the optimizer/basis implementation returns a different number
    # of coefficients than the naive (ell_max+1)**2; prefer the actual shape
    # returned to avoid broadcasting errors when writing to HDF5.
    if test_recon.ndim >= 4:
        num_coeffs = int(test_recon.shape[3])
    else:
        # Fallback: single coefficient per voxel
        num_coeffs = 1

    if verbose:
        print(f"Volume shape: {volume_shape}, Coefficients (inferred): {num_coeffs}")
        expected = (ell_max + 1) ** 2
        if expected != num_coeffs:
            print(f"Warning: expected {(ell_max + 1) ** 2} coefficients for ell_max={ell_max},"
                  f" but reconstruction returned {num_coeffs}. Using inferred value.")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create HDF5 file with pre-allocated datasets
    if verbose:
        print(f"\nCreating output file: {output_path}")
        print(f"Pre-allocating space for {total_samples} reconstructions and {len(h5_files)} ground truths")
    
    with h5py.File(output_path, 'w') as f:
        # Pre-allocate datasets
        recon_dset = f.create_dataset(
            'reconstructions',
            shape=(total_samples, *volume_shape, num_coeffs),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )
        
        gt_dset = f.create_dataset(
            'ground_truths',
            shape=(len(h5_files), *volume_shape, num_coeffs),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )
        
        indices_dset = f.create_dataset(
            'projection_indices',
            shape=(total_samples, num_projections),
            dtype=np.int32
        )
        
        mapping_dset = f.create_dataset(
            'reconstruction_to_gt_index',
            shape=(total_samples,),
            dtype=np.int32
        )
        
        # File identifiers list (will populate at end)
        file_identifiers_list = []
        
        # Counter for sparse reconstructions
        sparse_idx = 0
        
        # Process each HDF5 file
        for file_idx, (file_path, file_identifier) in enumerate(h5_files):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Processing file {file_idx + 1}/{len(h5_files)}: {file_identifier}")
                print(f"{'='*70}")
            
            # Load data container to get total number of projections
            # Set nonfinite_replacement_value=0 to handle NaN/Inf values
            dc_template = DataContainer(str(file_path), nonfinite_replacement_value=0)
            total_projections = len(dc_template.projections)
            
            if num_projections > total_projections:
                print(f"Warning: num_projections ({num_projections}) exceeds available "
                      f"projections ({total_projections}) in {file_identifier}. Skipping this file.")
                continue
            
            if verbose:
                print(f"Total projections available: {total_projections}")
            
            # First, compute and write ground truth
            if verbose:
                print(f"\nComputing ground truth reconstruction (all {total_projections} projections)...")
            
            dc_gt = DataContainer(str(file_path), nonfinite_replacement_value=0)
            ground_truth = _perform_reconstruction(
                dc=dc_gt,
                ell_max=ell_max,
                maxiter=maxiter,
                regularization_weight=regularization_weight,
                use_cuda=use_cuda,
                verbose=verbose
            )
            
            # Write ground truth immediately
            gt_dset[file_idx] = ground_truth
            file_identifiers_list.append(file_identifier)
            
            if verbose:
                print(f"Ground truth written to index {file_idx}")
                print(f"Ground truth shape: {ground_truth.shape}")
            
            # Clear memory
            del ground_truth, dc_gt
            
            # Now compute and write sparse reconstructions
            if verbose:
                print(f"\nComputing {num_iterations} sparse reconstructions...")
            
            for iteration in range(num_iterations):
                if verbose:
                    print(f"\n  Iteration {iteration + 1}/{num_iterations} for {file_identifier}")
                
                # Load a fresh copy of the data container
                dc = DataContainer(str(file_path), nonfinite_replacement_value=0)
                projections = dc.projections
                
                # Randomly sample projection indices
                all_indices_available = np.arange(total_projections)
                selected_indices = np.random.choice(
                    all_indices_available, 
                    size=num_projections, 
                    replace=False
                )
                selected_indices = np.sort(selected_indices)
                
                if verbose:
                    print(f"  Selected indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
                
                # Remove projections not in the selected subset
                indices_to_delete = [i for i in range(total_projections) if i not in selected_indices]
                for i in sorted(indices_to_delete, reverse=True):
                    del projections[i]
                
                if verbose:
                    print(f"  Remaining projections: {len(projections)}")
                
                # Perform reconstruction
                reconstruction = _perform_reconstruction(
                    dc=dc,
                    ell_max=ell_max,
                    maxiter=maxiter,
                    regularization_weight=regularization_weight,
                    use_cuda=use_cuda,
                    verbose=verbose
                )
                
                # Write immediately to HDF5
                recon_dset[sparse_idx] = reconstruction
                indices_dset[sparse_idx] = selected_indices
                mapping_dset[sparse_idx] = file_idx
                
                if verbose:
                    print(f"  Reconstruction written to index {sparse_idx}")
                
                # Clear memory
                del reconstruction, dc
                
                sparse_idx += 1
        
        # Save file identifiers
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('file_identifiers', data=file_identifiers_list, dtype=dt)
        
        # Save metadata
        f.create_dataset('num_projections', data=num_projections)
        f.create_dataset('ell_max', data=ell_max)
        f.create_dataset('num_iterations', data=num_iterations)
        f.create_dataset('maxiter', data=maxiter)
        f.create_dataset('regularization_weight', data=regularization_weight)
        
        # Store attributes
        f.attrs['data_path'] = str(data_path)
        f.attrs['num_coefficients'] = num_coeffs
        f.attrs['volume_shape'] = volume_shape
        f.attrs['num_files'] = len(h5_files)
        f.attrs['total_sparse_reconstructions'] = sparse_idx
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Successfully saved results:")
        print(f"  - {sparse_idx} sparse reconstructions")
        print(f"  - {len(file_identifiers_list)} ground truth reconstructions")
        print(f"  - From {len(h5_files)} input file(s)")
        print(f"  - Volume shape: {volume_shape}")
        print(f"  - Output saved to: {output_path}")
        print(f"{'='*70}")



def main():
    """Command-line interface for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate tensor tomography reconstruction dataset"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to input HDF5 file with projection data"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path for output HDF5 file"
    )
    parser.add_argument(
        "--ell-max",
        type=int,
        default=8,
        help="Maximum spherical harmonic degree (default: 8)"
    )
    parser.add_argument(
        "--num-projections",
        type=int,
        default=60,
        help="Number of projections per reconstruction (default: 60)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of reconstruction iterations (default: 100)"
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        help="LBFGS max iterations (default: 20)"
    )
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=1.0,
        help="Laplacian regularization weight (default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if CUDA is available"
    )
    
    args = parser.parse_args()
    
    generate_reconstruction_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        ell_max=args.ell_max,
        num_projections=args.num_projections,
        num_iterations=args.num_iterations,
        maxiter=args.maxiter,
        regularization_weight=args.regularization_weight,
        seed=args.seed,
        force_cpu=args.force_cpu,
        verbose=True
    )


if __name__ == "__main__":
    main()
