#!/usr/bin/env python
"""
Active Learning Pipeline for Sparse Tensor Tomography

This script implements an active learning loop that:
1. Starts with a small subset of projection angles
2. Computes multiple reconstructions using 80% subsampling
3. Forward projects to all available angles
4. Selects angles with highest variability to add to the training set
5. Tracks performance metrics against ground truth

Note: Uses even ℓ spherical harmonics only (ℓ = 0, 2, 4, 6, 8, ...)
For ell_max=8, this gives 45 coefficients (not 81).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Callable, List, Optional, Tuple
import json

# Import mumott modules
from mumott.data_handling import DataContainer
from mumott.methods.basis_sets import SphericalHarmonics
from mumott.methods.projectors import SAXSProjector, SAXSProjectorCUDA

# Import custom functions
sys.path.insert(0, '/das/home/barbaf_l/smartTT')
from smartt.data_processing import _perform_reconstruction


def get_num_coeffs(ell_max: int) -> int:
    """
    Compute number of spherical harmonic coefficients for even ℓ only.
    
    Mumott uses only even ℓ values: ℓ = 0, 2, 4, 6, 8, ...
    For each ℓ, there are (2ℓ + 1) coefficients (m = -ℓ, ..., +ℓ)
    
    For ell_max=8:
    - ℓ=0: 1 coeff
    - ℓ=2: 5 coeffs
    - ℓ=4: 9 coeffs
    - ℓ=6: 13 coeffs
    - ℓ=8: 17 coeffs
    Total: 45 coefficients
    
    Args:
        ell_max: Maximum spherical harmonic degree
    
    Returns:
        Number of coefficients for even ℓ values only
    """
    num_coeffs = 0
    for ell in range(0, ell_max + 1, 2):  # Only even ℓ
        num_coeffs += 2 * ell + 1
    return num_coeffs


def compute_ground_truth(
    data_path: str,
    ell_max: int,
    maxiter: int,
    regularization_weight: float,
    use_cuda: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, DataContainer]:
    """
    Compute ground truth reconstruction using all available projections.
    
    Returns:
        ground_truth: Reconstruction volume
        dc_full: DataContainer with all projections
    """
    print("Loading data container for ground truth...")
    dc_full = DataContainer(data_path, nonfinite_replacement_value=0)
    total_projections = len(dc_full.projections)
    
    print(f"Computing ground truth with all {total_projections} projections...")
    ground_truth = _perform_reconstruction(
        dc=dc_full,
        ell_max=ell_max,
        maxiter=maxiter,
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=verbose
    )
    
    return ground_truth, dc_full


def compute_multiple_reconstructions(
    data_path: str,
    current_indices: np.ndarray,
    num_experiments: int,
    subsample_fraction: float,
    ell_max: int,
    maxiter: int,
    regularization_weight: float,
    use_cuda: bool = True,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute multiple reconstructions by subsampling current projection set.
    
    Args:
        data_path: Path to HDF5 dataset
        current_indices: Current set of projection indices
        num_experiments: Number of reconstructions to compute (m parameter)
        subsample_fraction: Fraction of current indices to use (default 0.8)
        ell_max: Maximum spherical harmonic degree
        maxiter: Maximum optimization iterations
        regularization_weight: Regularization weight
        use_cuda: Use CUDA projector
        verbose: Print reconstruction details
    
    Returns:
        all_reconstructions: Array of shape (num_experiments, *volume_shape, num_coeffs)
    """
    total_projections_available = len(DataContainer(data_path, nonfinite_replacement_value=0).projections)
    num_subsamples = int(subsample_fraction * len(current_indices))
    
    # Get volume shape from first reconstruction
    dc_temp = DataContainer(data_path, nonfinite_replacement_value=0)
    volume_shape = dc_temp.geometry.volume_shape
    num_coeffs = get_num_coeffs(ell_max)
    del dc_temp
    
    # Pre-allocate array
    all_reconstructions = np.zeros(
        (num_experiments, *volume_shape, num_coeffs),
        dtype=np.float32
    )
    
    for exp_idx in range(num_experiments):
        # Randomly sample subsample_fraction of current indices
        subsample_indices = np.random.choice(
            current_indices,
            size=num_subsamples,
            replace=False
        )
        subsample_indices = np.sort(subsample_indices)
        
        # Create data container with subsampled projections
        dc_subsample = DataContainer(data_path, nonfinite_replacement_value=0)
        
        # Remove projections not in the subsample
        all_indices = np.arange(total_projections_available)
        indices_to_delete = [i for i in all_indices if i not in subsample_indices]
        
        for i in sorted(indices_to_delete, reverse=True):
            del dc_subsample.projections[i]
        
        # Perform reconstruction
        reconstruction = _perform_reconstruction(
            dc=dc_subsample,
            ell_max=ell_max,
            maxiter=maxiter,
            regularization_weight=regularization_weight,
            use_cuda=use_cuda,
            verbose=verbose
        )
        
        all_reconstructions[exp_idx] = reconstruction
        
        # Clean up
        del dc_subsample, reconstruction
    
    return all_reconstructions


def forward_project_reconstructions(
    reconstructions: np.ndarray,
    dc_full: DataContainer,
    ell_max: int,
    use_cuda: bool = False
) -> np.ndarray:
    """
    Forward project reconstructions through all projection angles.
    
    Args:
        reconstructions: Array of shape (num_experiments, *volume_shape, num_coeffs)
        dc_full: DataContainer with full geometry
        ell_max: Maximum spherical harmonic degree
        use_cuda: Use CUDA projector
    
    Returns:
        forward_projections: Array of shape (num_experiments, num_angles, *proj_shape, num_detector_angles)
    """
    num_experiments = reconstructions.shape[0]
    
    # Create projector and basis set
    if use_cuda:
        projector = SAXSProjectorCUDA(dc_full.geometry)
    else:
        projector = SAXSProjector(dc_full.geometry)
    
    basis_set = SphericalHarmonics(
        ell_max=ell_max,
        probed_coordinates=dc_full.geometry.probed_coordinates
    )
    
    num_angles = len(dc_full.geometry.inner_angles)
    proj_shape = dc_full.geometry.projection_shape
    num_detector_angles = len(dc_full.geometry.detector_angles)
    
    # Pre-allocate array
    all_forward_projections = np.zeros(
        (num_experiments, num_angles, proj_shape[0], proj_shape[1], num_detector_angles),
        dtype=np.float32
    )
    
    for exp_idx in range(num_experiments):
        reconstruction = reconstructions[exp_idx].astype(np.float32)
        
        # Forward project
        spatial_projection = projector.forward(reconstruction)
        forward_proj = basis_set.forward(spatial_projection)
        
        all_forward_projections[exp_idx] = forward_proj.astype(np.float32)
        
        # Clean up
        del reconstruction, spatial_projection, forward_proj
    
    return all_forward_projections


def compute_angle_variability(forward_projections: np.ndarray) -> np.ndarray:
    """
    Compute variability (sum of squared std) for each projection angle.
    
    Args:
        forward_projections: Array of shape (num_experiments, num_angles, *proj_shape, num_detector_angles)
    
    Returns:
        variability_per_angle: Array of shape (num_angles,)
    """
    # Compute std across experiments (axis 0)
    std_per_angle = np.std(forward_projections, axis=0)
    
    # Sum of squared std across all pixels and detector angles
    variability_per_angle = np.sum(std_per_angle**2, axis=(1, 2, 3))
    
    return variability_per_angle


def select_top_angles(
    variability_per_angle: np.ndarray,
    current_indices: np.ndarray,
    b: int = 1
) -> np.ndarray:
    """
    Select top b angles with highest variability that are not in current set.
    
    Args:
        variability_per_angle: Variability for each angle
        current_indices: Current set of projection indices
        b: Number of angles to select
    
    Returns:
        selected_indices: Top b angle indices to add
    """
    # Create mask for angles not in current set
    available_mask = np.ones(len(variability_per_angle), dtype=bool)
    available_mask[current_indices] = False
    
    # Get variability only for available angles
    available_variability = variability_per_angle.copy()
    available_variability[~available_mask] = -np.inf
    
    # Get top b indices
    selected_indices = np.argsort(available_variability)[-b:][::-1]
    
    return selected_indices


def compute_metrics(
    mean_reconstruction: np.ndarray,
    ground_truth: np.ndarray,
    metrics: Dict[str, Callable]
) -> Dict[str, float]:
    """
    Compute metrics between mean reconstruction and ground truth.
    
    Args:
        mean_reconstruction: Mean of subsampled reconstructions
        ground_truth: Ground truth reconstruction
        metrics: Dictionary mapping metric names to callable functions
    
    Returns:
        results: Dictionary mapping metric names to computed values
    """
    results = {}
    for name, metric_func in metrics.items():
        results[name] = metric_func(mean_reconstruction, ground_truth)
    
    return results


def mse_metric(reconstruction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Mean Squared Error metric."""
    return float(np.mean((reconstruction - ground_truth) ** 2))


def mae_metric(reconstruction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Mean Absolute Error metric."""
    return float(np.mean(np.abs(reconstruction - ground_truth)))


def run_active_learning(
    data_path: str,
    initial_indices: Optional[np.ndarray],
    num_iterations: int,
    num_experiments: int,
    subsample_fraction: float,
    b: int,
    ell_max: int,
    maxiter: int,
    regularization_weight: float,
    metrics: Dict[str, Callable],
    use_cuda: bool = True,
    use_cuda_forward: bool = False,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Run active learning loop for sparse tensor tomography.
    
    Args:
        data_path: Path to HDF5 dataset
        initial_indices: Initial projection indices (if None, randomly sample)
        num_iterations: Number of active learning iterations (t parameter)
        num_experiments: Number of reconstructions per iteration (m parameter)
        subsample_fraction: Fraction of projections to use in each experiment (default 0.8)
        b: Number of top angles to add per iteration
        ell_max: Maximum spherical harmonic degree
        maxiter: Maximum optimization iterations
        regularization_weight: Regularization weight
        metrics: Dictionary of metrics to track
        use_cuda: Use CUDA for reconstruction
        use_cuda_forward: Use CUDA for forward projection
        seed: Random seed for reproducibility
        save_path: Path to save results (optional)
        verbose: Print progress
    
    Returns:
        results: Dictionary containing:
            - history: List of metric dictionaries at each iteration
            - indices_history: List of projection indices at each iteration
            - selected_angles_history: List of newly selected angles at each iteration
            - ground_truth: Ground truth reconstruction
            - final_mean_reconstruction: Final mean reconstruction
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Compute ground truth
    if verbose:
        print("=" * 80)
        print("COMPUTING GROUND TRUTH")
        print("=" * 80)
    ground_truth, dc_full = compute_ground_truth(
        data_path=data_path,
        ell_max=ell_max,
        maxiter=maxiter,
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=verbose
    )
    
    total_projections = len(dc_full.projections)
    
    # Compute ground truth forward projection (only once)
    if verbose:
        print("\nComputing ground truth forward projection...")
    ground_truth_forward = forward_project_reconstructions(
        reconstructions=ground_truth[np.newaxis, ...],  # Add batch dimension
        dc_full=dc_full,
        ell_max=ell_max,
        use_cuda=use_cuda_forward
    )[0]  # Remove batch dimension
    
    # Initialize with provided or random indices
    if initial_indices is None:
        num_initial = 20  # Default starting size
        current_indices = np.random.choice(total_projections, size=num_initial, replace=False)
        current_indices = np.sort(current_indices)
    else:
        current_indices = np.array(initial_indices)
    
    # Storage for results
    history = []
    projection_history = []  # Metrics in projection space
    indices_history = [current_indices.copy()]
    selected_angles_history = []
    variability_history = []  # Variability per angle at each iteration
    
    if verbose:
        print(f"\n{'=' * 80}")
        print("STARTING ACTIVE LEARNING LOOP")
        print(f"{'=' * 80}")
        print(f"Total projections available: {total_projections}")
        print(f"Initial projections: {len(current_indices)}")
        print(f"Iterations: {num_iterations}")
        print(f"Experiments per iteration: {num_experiments}")
        print(f"Subsample fraction: {subsample_fraction}")
        print(f"Angles added per iteration: {b}")
        print(f"{'=' * 80}\n")
    
    # Active learning loop
    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"Current number of projections: {len(current_indices)}")
            print(f"{'=' * 80}")
        
        # Step 1: Compute multiple reconstructions with subsampling
        if verbose:
            print(f"\n[1/4] Computing {num_experiments} reconstructions...")
        
        all_reconstructions = compute_multiple_reconstructions(
            data_path=data_path,
            current_indices=current_indices,
            num_experiments=num_experiments,
            subsample_fraction=subsample_fraction,
            ell_max=ell_max,
            maxiter=maxiter,
            regularization_weight=regularization_weight,
            use_cuda=use_cuda,
            verbose=False
        )
        
        # Compute mean reconstruction
        mean_reconstruction = np.mean(all_reconstructions, axis=0)
        
        # Step 2: Compute metrics in reconstruction space
        if verbose:
            print(f"[2/4] Computing metrics...")
        
        metric_results = compute_metrics(mean_reconstruction, ground_truth, metrics)
        history.append(metric_results)
        
        if verbose:
            print(f"      Reconstruction metrics: {metric_results}")
        
        # Step 3: Forward project and compute variability
        if verbose:
            print(f"[3/4] Forward projecting to all {total_projections} angles...")
        
        forward_projections = forward_project_reconstructions(
            reconstructions=all_reconstructions,
            dc_full=dc_full,
            ell_max=ell_max,
            use_cuda=use_cuda_forward
        )
        
        # Compute mean forward projection and metrics in projection space
        mean_forward_projection = np.mean(forward_projections, axis=0)
        projection_metric_results = compute_metrics(mean_forward_projection, ground_truth_forward, metrics)
        projection_history.append(projection_metric_results)
        
        if verbose:
            print(f"      Projection metrics: {projection_metric_results}")
        
        variability_per_angle = compute_angle_variability(forward_projections)
        variability_history.append(variability_per_angle.copy())
        
        # Step 4: Select top b angles to add
        if verbose:
            print(f"[4/4] Selecting top {b} angles with highest variability...")
        
        selected_angles = select_top_angles(
            variability_per_angle=variability_per_angle,
            current_indices=current_indices,
            b=b
        )
        
        selected_angles_history.append(selected_angles)
        
        if verbose:
            print(f"      Selected angles: {selected_angles}")
            print(f"      Variability: {[f'{variability_per_angle[idx]:.2e}' for idx in selected_angles]}")
        
        # Update current indices
        current_indices = np.sort(np.concatenate([current_indices, selected_angles]))
        indices_history.append(current_indices.copy())
        
        # Clean up
        del all_reconstructions, forward_projections
    
    # Final reconstruction
    if verbose:
        print(f"\n{'=' * 80}")
        print("COMPUTING FINAL RECONSTRUCTION")
        print(f"{'=' * 80}")
    
    final_reconstructions = compute_multiple_reconstructions(
        data_path=data_path,
        current_indices=current_indices,
        num_experiments=num_experiments,
        subsample_fraction=subsample_fraction,
        ell_max=ell_max,
        maxiter=maxiter,
        regularization_weight=regularization_weight,
        use_cuda=use_cuda,
        verbose=False
    )
    
    final_mean_reconstruction = np.mean(final_reconstructions, axis=0)
    final_metrics = compute_metrics(final_mean_reconstruction, ground_truth, metrics)
    history.append(final_metrics)
    
    # Compute final projection metrics
    final_forward_projections = forward_project_reconstructions(
        reconstructions=final_reconstructions,
        dc_full=dc_full,
        ell_max=ell_max,
        use_cuda=use_cuda_forward
    )
    final_mean_forward_projection = np.mean(final_forward_projections, axis=0)
    final_projection_metrics = compute_metrics(final_mean_forward_projection, ground_truth_forward, metrics)
    projection_history.append(final_projection_metrics)
    
    if verbose:
        print(f"Final reconstruction metrics: {final_metrics}")
        print(f"Final projection metrics: {final_projection_metrics}")
        print(f"Final number of projections: {len(current_indices)}")
    
    # Compile results
    results = {
        'history': history,
        'projection_history': projection_history,
        'indices_history': indices_history,
        'selected_angles_history': selected_angles_history,
        'variability_history': variability_history,
        'ground_truth': ground_truth,
        'ground_truth_forward': ground_truth_forward,
        'final_mean_reconstruction': final_mean_reconstruction,
        'final_indices': current_indices,
        'total_projections': total_projections
    }
    
    # Save results if requested
    if save_path is not None:
        save_results(results, save_path, verbose=verbose)
    
    return results


def save_results(results: Dict, save_path: str, verbose: bool = True):
    """Save results to HDF5 file."""
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"SAVING RESULTS")
        print(f"{'=' * 80}")
        print(f"Output path: {save_path}")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        # Save reconstructions
        f.create_dataset('ground_truth', data=results['ground_truth'], compression='gzip')
        f.create_dataset('final_mean_reconstruction', 
                        data=results['final_mean_reconstruction'], compression='gzip')
        
        # Save indices history
        for i, indices in enumerate(results['indices_history']):
            f.create_dataset(f'indices_history/{i:04d}', data=indices)
        
        # Save selected angles history
        for i, angles in enumerate(results['selected_angles_history']):
            f.create_dataset(f'selected_angles_history/{i:04d}', data=angles)
        
        # Save variability history (variability per angle at each iteration)
        if 'variability_history' in results and results['variability_history']:
            variability_array = np.array(results['variability_history'])
            f.create_dataset('variability_history', data=variability_array, compression='gzip')
        
        # Save final indices
        f.create_dataset('final_indices', data=results['final_indices'])
        
        # Save ground truth forward projection if available
        if 'ground_truth_forward' in results:
            f.create_dataset('ground_truth_forward', 
                            data=results['ground_truth_forward'], compression='gzip')
        
        # Save metadata
        f.attrs['total_projections'] = results['total_projections']
        f.attrs['num_iterations'] = len(results['selected_angles_history'])
        
        # Save metrics history as JSON string
        f.attrs['history'] = json.dumps(results['history'])
        
        # Save projection metrics history as JSON string
        if 'projection_history' in results:
            f.attrs['projection_history'] = json.dumps(results['projection_history'])
    
    if verbose:
        print("Results saved successfully!")


def plot_metrics_history(history: List[Dict], save_path: Optional[str] = None):
    """Plot metric values over iterations."""
    if len(history) == 0:
        print("No history to plot")
        return
    
    metric_names = list(history[0].keys())
    num_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(metric_names):
        values = [h[metric_name] for h in history]
        axes[i].plot(values, 'o-', linewidth=2, markersize=8)
        axes[i].set_xlabel('Iteration', fontsize=12)
        axes[i].set_ylabel(metric_name, fontsize=12)
        axes[i].set_title(f'{metric_name} vs Iteration', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage with dummy parameters
    print("Active Learning Script - Test Run")
    print("=" * 80)
    
    # Configuration
    data_path = '/das/home/barbaf_l/p20639/Mads/frog/frogbone/dataset_qbin_0000.h5'
    
    # Active learning parameters
    initial_indices = None  # Will randomly sample 20 angles
    num_iterations = 3  # Small number for testing
    num_experiments = 3  # m parameter - number of reconstructions per iteration
    subsample_fraction = 0.8
    b = 1  # Add 1 angle per iteration
    
    # Reconstruction parameters
    ell_max = 8  # Reduced for faster testing
    maxiter = 10  # Reduced for faster testing
    regularization_weight = 1.0
    use_cuda = True
    use_cuda_forward = True  # CPU forward projection for compatibility
    
    # Metrics to track
    metrics = {
        'mse': mse_metric,
        'mae': mae_metric
    }
    
    # Run active learning
    results = run_active_learning(
        data_path=data_path,
        initial_indices=initial_indices,
        num_iterations=num_iterations,
        num_experiments=num_experiments,
        subsample_fraction=subsample_fraction,
        b=b,
        ell_max=ell_max,
        maxiter=maxiter,
        regularization_weight=regularization_weight,
        metrics=metrics,
        use_cuda=use_cuda,
        use_cuda_forward=use_cuda_forward,
        seed=42,
        save_path='/das/home/barbaf_l/smartTT/results/active_learning_test.h5',
        verbose=True
    )
    
    # Plot results
    print("\nPlotting metrics history...")
    plot_metrics_history(results['history'], 
                         save_path='/das/home/barbaf_l/smartTT/results/active_learning_metrics.png')
    
    print("\n" + "=" * 80)
    print("ACTIVE LEARNING TEST COMPLETE")
    print("=" * 80)
