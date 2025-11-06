"""
Data processing module for generating tensor tomography reconstruction datasets.

This module provides functionality to load projection data, randomly sample subsets,
perform mumott reconstructions, and save the results for machine learning training.
"""

import copy
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, List
from mumott.data_handling import DataContainer
from mumott.methods.basis_sets import SphericalHarmonics
from mumott.methods.projectors import SAXSProjector, SAXSProjectorCUDA
from mumott.methods.residual_calculators import GradientResidualCalculator
from mumott.optimization.loss_functions import SquaredLoss
from mumott.optimization.optimizers import LBFGS
from mumott.optimization.regularizers import Laplacian


def generate_reconstruction_dataset(
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    ell_max: int = 8,
    num_projections: int = 60,
    num_iterations: int = 100,
    maxiter: int = 20,
    regularization_weight: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = True
) -> None:
    """
    Generate a dataset of tensor tomography reconstructions from random projection subsets.
    
    This function loads a projection dataset, randomly samples subsets of projections,
    performs spherical harmonic reconstructions using mumott, and saves the results
    to an HDF5 file for machine learning training.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the input HDF5 file containing the projection data.
    output_path : str or Path
        Path where the output HDF5 file will be saved.
    ell_max : int, default=8
        Maximum degree for spherical harmonics expansion. This determines the number
        of coefficients: (ell_max + 1)^2.
    num_projections : int, default=60
        Number of projections to randomly sample for each reconstruction.
    num_iterations : int, default=100
        Number of different random samplings to perform.
    maxiter : int, default=20
        Maximum number of iterations for the LBFGS optimizer.
    regularization_weight : float, default=1.0
        Weight for the Laplacian regularization term.
    seed : int, optional
        Random seed for reproducibility. If None, results will be non-deterministic.
    verbose : bool, default=True
        Whether to print progress information.
    
    Returns
    -------
    None
        Results are saved to output_path as an HDF5 file with the following structure:
        - 'reconstructions': Dataset of shape (num_iterations, *volume_shape, num_coeffs)
        - 'num_projections': Scalar dataset with the number of projections used
        - 'ell_max': Scalar dataset with the ell_max parameter
        - 'projection_indices': Dataset of shape (num_iterations, num_projections)
          containing the indices used for each reconstruction
    
    Examples
    --------
    >>> from smartt import generate_reconstruction_dataset
    >>> generate_reconstruction_dataset(
    ...     data_path='trabecular_bone_9.h5',
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
    use_cuda = torch.cuda.is_available()
    if verbose:
        print(f"Using {'CUDA' if use_cuda else 'CPU'} for computations")
    
    # Load the data container once to get total number of projections
    if verbose:
        print(f"Loading data from {data_path}")
    dc_template = DataContainer(str(data_path))
    total_projections = len(dc_template.projections)
    
    if num_projections > total_projections:
        raise ValueError(
            f"num_projections ({num_projections}) cannot exceed total "
            f"available projections ({total_projections})"
        )
    
    # Calculate number of spherical harmonic coefficients
    num_coeffs = (ell_max + 1) ** 2
    
    if verbose:
        print(f"Total projections available: {total_projections}")
        print(f"Sampling {num_projections} projections per iteration")
        print(f"Spherical harmonics with ell_max={ell_max} -> {num_coeffs} coefficients")
        print(f"Running {num_iterations} iterations")
    
    # Storage for results
    reconstructions_list = []
    indices_list = []
    
    # Main loop
    for iteration in range(num_iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Load a fresh copy of the data container
        dc = DataContainer(str(data_path))
        projections = dc.projections
        
        # Randomly sample projection indices
        all_indices = np.arange(total_projections)
        selected_indices = np.random.choice(
            all_indices, 
            size=num_projections, 
            replace=False
        )
        selected_indices = np.sort(selected_indices)  # Sort for consistency
        
        if verbose:
            print(f"Selected indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}")
        
        # Determine which indices to delete
        indices_to_delete = [i for i in range(total_projections) if i not in selected_indices]
        
        # Remove projections not in the selected subset
        for i in sorted(indices_to_delete, reverse=True):
            del projections[i]
        
        if verbose:
            print(f"Remaining projections: {len(projections)}")
        
        # Set up the reconstruction
        basis_set = SphericalHarmonics(ell_max=ell_max)
        projector = SAXSProjectorCUDA(dc.geometry) if use_cuda else SAXSProjector(dc.geometry)
        
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
        # The LBFGS optimizer returns a dictionary with the reconstruction
        if isinstance(results, dict):
            # Check for common keys in mumott optimizer results
            if 'x' in results:
                reconstruction = results['x']
            elif 'reconstruction' in results:
                reconstruction = results['reconstruction']
            elif 'coefficients' in results:
                reconstruction = results['coefficients']
            else:
                # If none of these keys exist, print available keys for debugging
                if verbose:
                    print(f"Warning: Unexpected result keys: {list(results.keys())}")
                # Try to get the first value that looks like an array
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
            # Assume results is the reconstruction itself
            reconstruction = results
        
        # Convert to numpy if it's a torch tensor
        if torch.is_tensor(reconstruction):
            reconstruction = reconstruction.cpu().numpy()
        
        # Ensure it's a numpy array
        if not isinstance(reconstruction, np.ndarray):
            reconstruction = np.asarray(reconstruction)
        
        reconstruction = reconstruction.astype(np.float32)
        
        if verbose:
            print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Store results
        reconstructions_list.append(reconstruction)
        indices_list.append(selected_indices)
    
    # Save to HDF5
    if verbose:
        print(f"\nSaving results to {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Stack all reconstructions
        reconstructions_array = np.stack(reconstructions_list, axis=0)
        f.create_dataset(
            'reconstructions', 
            data=reconstructions_array,
            compression='gzip',
            compression_opts=4
        )
        
        # Save indices
        indices_array = np.stack(indices_list, axis=0)
        f.create_dataset('projection_indices', data=indices_array)
        
        # Save metadata
        f.create_dataset('num_projections', data=num_projections)
        f.create_dataset('ell_max', data=ell_max)
        f.create_dataset('num_iterations', data=num_iterations)
        f.create_dataset('maxiter', data=maxiter)
        f.create_dataset('regularization_weight', data=regularization_weight)
        
        # Store attributes
        f.attrs['data_path'] = str(data_path)
        f.attrs['num_coefficients'] = num_coeffs
        f.attrs['volume_shape'] = reconstruction.shape[:3]
    
    if verbose:
        print(f"Successfully saved {num_iterations} reconstructions to {output_path}")
        print(f"Dataset shape: {reconstructions_array.shape}")


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
        verbose=True
    )


if __name__ == "__main__":
    main()
