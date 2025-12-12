"""
Example script for running active learning pipeline.

This demonstrates how to use the active_learning module with custom parameters.
"""

import sys
sys.path.insert(0, '/das/home/barbaf_l/smartTT')

from scripts.active_learning import (
    run_active_learning,
    mse_metric,
    mae_metric,
    plot_metrics_history
)

# Configuration
data_path = '/das/home/barbaf_l/p20639/Mads/frog/frogbone/dataset_qbin_0000.h5'

# Active learning parameters
initial_indices = None  # Will randomly sample 20 angles
num_iterations = 20  # Number of active learning iterations
num_experiments = 10  # Number of reconstructions per iteration
subsample_fraction = 0.8  # Use 80% of current angles per experiment
b = 2  # Add 2 angles per iteration (can be increased to add multiple angles)

# Reconstruction parameters
ell_max = 8  # Maximum spherical harmonic degree
maxiter = 20  # Optimization iterations
regularization_weight = 1.0
use_cuda = True
use_cuda_forward = True  # Set to True if CUDA is available for forward projection

# Metrics to track
metrics = {
    'mse': mse_metric,
    'mae': mae_metric
}

# Output path
output_dir = '/das/home/barbaf_l/smartTT/results'
save_path = f'{output_dir}/active_learning_results.h5'
plot_path = f'{output_dir}/active_learning_metrics.png'

# Run active learning
print("Starting Active Learning Pipeline")
print("=" * 80)

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
    save_path=save_path,
    verbose=True
)

# Plot results
print("\nPlotting metrics history...")
plot_metrics_history(results['history'], save_path=plot_path)

print("\n" + "=" * 80)
print("ACTIVE LEARNING COMPLETE")
print("=" * 80)
print(f"Results saved to: {save_path}")
print(f"Plot saved to: {plot_path}")
