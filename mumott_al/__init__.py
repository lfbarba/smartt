"""
mumott_al - Active Learning utilities for mumott tensor tomography.

This package provides utilities for:
- Generating new geometries with arbitrary projection directions
- Creating synthetic projections from reconstructions
- Computing metrics to compare reconstructions
"""

from .geometry import (
    fibonacci_hemisphere,
    cartesian_to_spherical,
    create_rotation_matrix,
    create_geometry_from_directions,
    create_synthetic_projections,
    generate_geometry_and_projections,
    create_synthetic_data_container,
    generate_and_save_synthetic_data,
)

from .metrics import (
    mse_coefficients,
    normalized_mse_coefficients,
    projection_metrics,
    orientation_similarity,
    real_space_metrics,
    compare_reconstructions,
    RealSpaceMetricsResult,
)

from .visualization import (
    plot_projection_residuals,
    plot_projection_residuals_comparison,
)

__version__ = "0.1.0"
__all__ = [
    # Geometry functions
    "fibonacci_hemisphere",
    "cartesian_to_spherical",
    "create_rotation_matrix",
    "create_geometry_from_directions",
    "create_synthetic_projections",
    "generate_geometry_and_projections",
    "create_synthetic_data_container",
    "generate_and_save_synthetic_data",
    # Metrics functions
    "mse_coefficients",
    "normalized_mse_coefficients",
    "projection_metrics",
    "orientation_similarity",
    "compare_reconstructions",
    # Visualization functions
    "plot_projection_residuals",
    "plot_projection_residuals_comparison",
]
