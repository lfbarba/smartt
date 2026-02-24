"""
Metrics for comparing tensor tomography reconstructions.

This module provides various metrics to compare reconstructions:
- Coefficient-based metrics (MSE, normalized MSE)
- Projection-based metrics (MSE, PSNR, SSIM per channel and global)
- Orientation-based metrics (cosine similarity of principal directions)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
from scipy.ndimage import uniform_filter as _scipy_uniform_filter

from mumott import Geometry, ProbedCoordinates
from mumott.methods.projectors import SAXSProjector
from mumott.methods.basis_sets import SphericalHarmonics


@dataclass
class ProjectionMetricsResult:
    """Container for projection-based metrics results."""
    mse_per_channel: np.ndarray
    mse_global: float
    psnr_per_channel: np.ndarray
    psnr_global: float
    ssim_per_channel: np.ndarray
    ssim_global: float
    n_channels: int


@dataclass
class OrientationMetricsResult:
    """Container for orientation-based metrics results."""
    cosine_similarity_mean: float
    cosine_similarity_std: float
    cosine_similarity_map: np.ndarray
    angular_error_mean_degrees: float
    angular_error_std_degrees: float
    valid_voxels: int


@dataclass
class RealSpaceMetricsResult:
    """Container for real-space SH evaluation metrics."""
    psnr: float
    ssim: float
    mse: float
    resolution_in_degrees: int
    n_directions: int


@dataclass
class ReconstructionComparisonResult:
    """Container for all reconstruction comparison metrics."""
    mse_coefficients: float
    normalized_mse_coefficients: float
    projection_metrics: Optional[ProjectionMetricsResult]
    orientation_metrics: Optional[OrientationMetricsResult]
    real_space_metrics: Optional['RealSpaceMetricsResult'] = None


def mse_coefficients(
    reconstruction_pred: np.ndarray,
    reconstruction_gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Mean Squared Error of spherical harmonic coefficients.
    
    Parameters
    ----------
    reconstruction_pred : np.ndarray
        Predicted reconstruction with shape (x, y, z, n_coefficients).
    reconstruction_gt : np.ndarray
        Ground truth reconstruction with shape (x, y, z, n_coefficients).
    mask : np.ndarray, optional
        Binary mask of shape (x, y, z) to select voxels for comparison.
        If None, all voxels are used.
        
    Returns
    -------
    float
        Mean squared error of the coefficients.
        
    Notes
    -----
    This metric treats all coefficients equally, but energy is spread
    non-uniformly across spherical harmonic orders. Consider using
    `normalized_mse_coefficients` for a more balanced comparison.
    """
    if mask is not None:
        pred_masked = reconstruction_pred[mask > 0]
        gt_masked = reconstruction_gt[mask > 0]
    else:
        pred_masked = reconstruction_pred.flatten()
        gt_masked = reconstruction_gt.flatten()
    
    return np.mean((pred_masked - gt_masked) ** 2)


def normalized_mse_coefficients(
    reconstruction_pred: np.ndarray,
    reconstruction_gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ell_max: int = 8,
) -> float:
    """
    Compute normalized MSE with per-order weighting to account for energy distribution.
    
    This metric normalizes each spherical harmonic order by its variance in the
    ground truth, providing a more balanced comparison across orders.
    
    Parameters
    ----------
    reconstruction_pred : np.ndarray
        Predicted reconstruction with shape (x, y, z, n_coefficients).
    reconstruction_gt : np.ndarray
        Ground truth reconstruction with shape (x, y, z, n_coefficients).
    mask : np.ndarray, optional
        Binary mask of shape (x, y, z) to select voxels for comparison.
    ell_max : int
        Maximum spherical harmonic order (default: 8).
        
    Returns
    -------
    float
        Normalized mean squared error.
    """
    if mask is not None:
        pred_masked = reconstruction_pred[mask > 0]
        gt_masked = reconstruction_gt[mask > 0]
    else:
        pred_masked = reconstruction_pred.reshape(-1, reconstruction_pred.shape[-1])
        gt_masked = reconstruction_gt.reshape(-1, reconstruction_gt.shape[-1])
    
    n_coeffs = reconstruction_pred.shape[-1]
    
    # Compute per-coefficient normalized MSE
    normalized_mse = 0.0
    valid_coeffs = 0
    
    for c in range(n_coeffs):
        gt_var = np.var(gt_masked[:, c])
        if gt_var > 1e-10:  # Only include coefficients with significant variance
            coeff_mse = np.mean((pred_masked[:, c] - gt_masked[:, c]) ** 2)
            normalized_mse += coeff_mse / gt_var
            valid_coeffs += 1
    
    if valid_coeffs > 0:
        normalized_mse /= valid_coeffs
    
    return normalized_mse


def _compute_ssim_2d(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float = None,
    win_size: int = 7,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) for 2D images.

    Uses scipy.ndimage.uniform_filter for fast C-level convolution.
    """
    if data_range is None:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        if data_range == 0:
            data_range = 1.0

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    p = img1.astype(np.float64)
    g = img2.astype(np.float64)

    mu1 = _scipy_uniform_filter(p, size=win_size, mode='reflect')
    mu2 = _scipy_uniform_filter(g, size=win_size, mode='reflect')
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = np.maximum(_scipy_uniform_filter(p ** 2, size=win_size, mode='reflect') - mu1_sq, 0)
    sigma2_sq = np.maximum(_scipy_uniform_filter(g ** 2, size=win_size, mode='reflect') - mu2_sq, 0)
    sigma12   = _scipy_uniform_filter(p * g,    size=win_size, mode='reflect') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def _compute_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    data_range: float = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    
    if data_range is None:
        data_range = max(gt.max() - gt.min(), 1e-10)
    
    return 10 * np.log10((data_range ** 2) / mse)


def _percentile_data_range(arr: np.ndarray, low: float = 5.0, high: float = 95.0) -> float:
    """Compute robust data range using percentiles."""
    p_low = np.percentile(arr, low)
    p_high = np.percentile(arr, high)
    return max(p_high - p_low, 1e-10)


def _build_sh_evaluation_matrix(
    basis_set: SphericalHarmonics,
    resolution_in_degrees: int = 10,
    map_half_sphere: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the SH basis evaluation matrix for a spherical grid.

    Returns
    -------
    Y : np.ndarray, shape (n_directions, n_coefficients)
        Each row holds all SH basis functions evaluated at one direction.
    map_theta : np.ndarray, shape (n_theta, n_phi)
        Polar angles of the grid.
    map_phi : np.ndarray, shape (n_theta, n_phi)
        Azimuthal angles of the grid.
    """
    if map_half_sphere:
        steps_theta = int(np.ceil(90 / resolution_in_degrees))
        map_theta_1d = np.linspace(0, np.pi / 2, steps_theta + 1)
    else:
        steps_theta = int(np.ceil(180 / resolution_in_degrees))
        map_theta_1d = np.linspace(0, np.pi, steps_theta + 1)

    steps_phi = int(np.ceil(360 / resolution_in_degrees))
    map_phi_1d = np.linspace(0, 2 * np.pi, steps_phi + 1)

    map_theta, map_phi = np.meshgrid(map_theta_1d, map_phi_1d, indexing='ij')

    x = np.cos(map_phi) * np.sin(map_theta)
    y = np.sin(map_phi) * np.sin(map_theta)
    z = np.cos(map_theta)
    vector = np.stack((x, y, z), axis=-1)  # (n_theta, n_phi, 3)

    probed_coordinates = ProbedCoordinates()
    probed_coordinates.vector = vector[:, :, np.newaxis, :]  # (n_theta, n_phi, 1, 3)

    # basis_matrix shape: (n_theta, n_phi, 1, n_coefficients)
    basis_matrix = basis_set._get_projection_matrix(probed_coordinates)[:, :, 0, :]
    Y = basis_matrix.reshape(-1, basis_matrix.shape[-1])  # (n_directions, n_coefficients)

    return Y, map_theta, map_phi


def real_space_metrics(
    reconstruction_pred: np.ndarray,
    reconstruction_gt: np.ndarray,
    ell_max: int = 8,
    mask: Optional[np.ndarray] = None,
    resolution_in_degrees: int = 10,
    map_half_sphere: bool = True,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    verbose: bool = False,
) -> RealSpaceMetricsResult:
    """
    Compute PSNR and SSIM of the real-valued SH functions, not the coefficients.

    The reconstructions are first evaluated on a spherical grid of directions,
    yielding a real-space representation of shape ``(nx, ny, nz, n_directions)``.
    MSE, PSNR and SSIM are then computed on this representation.

    SSIM is estimated by computing 2D SSIM on every ``(nx, ny)`` spatial slice
    (at each z-index and each direction) and averaging the results, to preserve
    the spatial-correlation semantics of the metric.

    Parameters
    ----------
    reconstruction_pred : np.ndarray
        Predicted reconstruction, shape ``(nx, ny, nz, n_coefficients)``.
    reconstruction_gt : np.ndarray
        Ground truth reconstruction, shape ``(nx, ny, nz, n_coefficients)``.
    ell_max : int
        Maximum spherical harmonic order (default: 8).
    mask : np.ndarray, optional
        Binary mask of shape ``(nx, ny, nz)``; voxels outside the mask are
        zeroed before evaluation.
    resolution_in_degrees : int
        Angular resolution of the evaluation grid in degrees (default: 10).
        Coarser values are faster and use less memory.
    map_half_sphere : bool
        If ``True`` (default) only the upper hemisphere (z >= 0) is sampled.
        For symmetric SH functions this is equivalent to the full sphere.
    percentile_low : float
        Lower percentile for robust data-range estimation (default: 5.0).
    percentile_high : float
        Upper percentile for robust data-range estimation (default: 95.0).

    Returns
    -------
    RealSpaceMetricsResult
        Container with ``psnr``, ``ssim``, ``mse``, ``resolution_in_degrees``
        and ``n_directions``.
    """
    # Apply mask
    if mask is not None:
        pred = reconstruction_pred * mask[:, :, :, None]
        gt = reconstruction_gt * mask[:, :, :, None]
    else:
        pred = reconstruction_pred
        gt = reconstruction_gt

    nx, ny, nz, n_coeffs = pred.shape

    # Build SH evaluation matrix once
    if verbose:
        print(f"  [real_space] Building SH evaluation matrix ({resolution_in_degrees}° grid) ...")
    basis_set = SphericalHarmonics(ell_max=ell_max)
    Y, _, _ = _build_sh_evaluation_matrix(
        basis_set, resolution_in_degrees=resolution_in_degrees, map_half_sphere=map_half_sphere
    )
    n_directions = Y.shape[0]  # n_theta * n_phi
    if verbose:
        print(f"  [real_space] Evaluating SH functions on {n_directions} directions "
              f"for {nx*ny*nz} voxels ...")

    # Batch-evaluate all voxels: (n_voxels, n_coeffs) @ (n_coeffs, n_directions)
    pred_flat = pred.astype(np.float64).reshape(-1, n_coeffs)   # (n_voxels, n_coeffs)
    gt_flat = gt.astype(np.float64).reshape(-1, n_coeffs)

    real_pred = pred_flat @ Y.T  # (n_voxels, n_directions)
    real_gt = gt_flat @ Y.T

    # Global MSE and PSNR
    mse_val = float(np.mean((real_pred - real_gt) ** 2))
    data_range = _percentile_data_range(real_gt, percentile_low, percentile_high)
    psnr_val = _compute_psnr(real_pred, real_gt, data_range)

    # SSIM: averaged over z-slices and directions
    # Reshape to (nx, ny, nz, n_directions) for spatial slicing
    real_pred_vol = real_pred.reshape(nx, ny, nz, n_directions)
    real_gt_vol = real_gt.reshape(nx, ny, nz, n_directions)

    # SSIM: vectorized over all directions per z-slice.
    # Each slab has shape (nx, ny, n_directions); scipy uniform_filter is called with
    # size=(win_size, win_size, 1) so it filters only along the two spatial axes.
    # This replaces nz × n_directions individual SSIM calls with just nz scipy calls.
    WIN = 7
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    sz = (WIN, WIN, 1)  # filter spatial axes only; leave direction axis untouched

    ssim_sum = 0.0
    ssim_count = 0
    z_iter = tqdm(range(nz), desc="  SSIM z-slices", leave=False) if verbose else range(nz)
    for z_idx in z_iter:
        p = real_pred_vol[:, :, z_idx, :]   # (nx, ny, n_directions)
        g = real_gt_vol[:, :, z_idx, :]
        mu1 = _scipy_uniform_filter(p, size=sz, mode='reflect')
        mu2 = _scipy_uniform_filter(g, size=sz, mode='reflect')
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1_sq = np.maximum(_scipy_uniform_filter(p ** 2, size=sz, mode='reflect') - mu1_sq, 0)
        s2_sq = np.maximum(_scipy_uniform_filter(g ** 2, size=sz, mode='reflect') - mu2_sq, 0)
        s12   = _scipy_uniform_filter(p * g,    size=sz, mode='reflect') - mu1_mu2
        ssim_slab = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
                    ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
        ssim_sum   += float(np.sum(ssim_slab))
        ssim_count += ssim_slab.size
    ssim_val = ssim_sum / ssim_count

    return RealSpaceMetricsResult(
        psnr=psnr_val,
        ssim=ssim_val,
        mse=mse_val,
        resolution_in_degrees=resolution_in_degrees,
        n_directions=n_directions,
    )


def projection_metrics(
    reconstruction_pred: np.ndarray,
    reconstruction_gt: np.ndarray,
    geometry: Geometry,
    ell_max: int = 8,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    verbose: bool = False,
) -> ProjectionMetricsResult:
    """
    Compute projection-based metrics between predicted and ground truth reconstructions.

    This function forward-projects both reconstructions using the provided geometry
    and computes MSE, PSNR, and SSIM for each detector channel and globally.
    Data range for PSNR and SSIM is computed per-channel using percentiles of the
    ground truth, making it robust to sparse high-value regions.

    Bad pixels are excluded via the ``weights`` array (``dc.projections.weights``).
    Pixels with ``weight == 0`` are treated as invalid and excluded from all
    computations. Non-zero weights are applied multiplicatively to the squared
    error before averaging, so downweighted pixels contribute less.

    Parameters
    ----------
    reconstruction_pred : np.ndarray
        Predicted reconstruction with shape (x, y, z, n_coefficients).
    reconstruction_gt : np.ndarray
        Ground truth reconstruction with shape (x, y, z, n_coefficients).
    geometry : Geometry
        Geometry object defining the projection directions.
    ell_max : int
        Maximum spherical harmonic order (default: 8).
    mask : np.ndarray, optional
        Binary mask of shape (x, y, z) to apply before projection.
    weights : np.ndarray, optional
        Pixel weights from ``dc.projections.weights``.  Shape must be either
        ``(n_projections, det_x, det_y)`` (same mask for all channels) or
        ``(n_projections, det_x, det_y, n_channels)`` (per-channel mask).
        Pixels with weight 0 are excluded; non-zero weights are used as
        importance weights in the error average.
    percentile_low : float
        Lower percentile for robust data range computation (default: 5.0).
    percentile_high : float
        Upper percentile for robust data range computation (default: 95.0).

    Returns
    -------
    ProjectionMetricsResult
        Container with per-channel and global metrics.
    """
    # Apply spatial mask if provided
    if mask is not None:
        pred_masked = reconstruction_pred * mask[:, :, :, None]
        gt_masked = reconstruction_gt * mask[:, :, :, None]
    else:
        pred_masked = reconstruction_pred
        gt_masked = reconstruction_gt

    # Create projector and basis set
    projector = SAXSProjector(geometry)
    basis_set = SphericalHarmonics(ell_max=ell_max, probed_coordinates=geometry.probed_coordinates)

    # Forward project both reconstructions
    n_proj = len(geometry.inner_angles)
    if verbose:
        print(f"  [projection] Forward projecting pred ({n_proj} projections) ...")
    proj_pred = basis_set.forward(projector.forward(pred_masked.astype(np.float64)))
    if verbose:
        print(f"  [projection] Forward projecting gt  ({n_proj} projections) ...")
    proj_gt = basis_set.forward(projector.forward(gt_masked.astype(np.float64)))

    n_channels = proj_pred.shape[-1]  # (n_proj, det_x, det_y, n_channels)

    # ------------------------------------------------------------------ weights
    # Broadcast weights to (n_proj, det_x, det_y, n_channels) if needed.
    # dc.projections.weights has shape (n_proj, det_x, det_y).
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape[0] != n_proj:
            raise ValueError(
                f"weights.shape[0]={w.shape[0]} does not match the number of projections "
                f"in geometry ({n_proj}). Make sure you pass the weights from the same "
                f"DataContainer whose geometry is used (e.g. new_dc.projections.weights)."
            )
        if w.ndim == 3:  # (n_proj, det_x, det_y) → add channel dim
            w = w[:, :, :, np.newaxis]  # broadcast across channels
        # Normalise per-channel so weights sum to the number of valid pixels
        # (keeps the MSE on the same scale as the unweighted version)
    else:
        w = np.ones_like(proj_gt)  # all pixels equally valid

    # Validity mask: True where the pixel should be included
    valid = w > 0  # shape (n_proj, det_x, det_y, n_channels)

    # Compute per-channel metrics
    mse_per_channel = np.zeros(n_channels)
    psnr_per_channel = np.zeros(n_channels)
    ssim_per_channel = np.zeros(n_channels)

    if verbose:
        print(f"  [projection] Computing per-channel metrics "
              f"({n_channels} channels × {n_proj} projections) ...")
    ch_iter = tqdm(range(n_channels), desc="  channels", leave=False) if verbose else range(n_channels)
    for c in ch_iter:
        pred_c = proj_pred[..., c]   # (n_proj, det_x, det_y)
        gt_c   = proj_gt[...,  c]
        w_c    = w[..., 0] if w.shape[-1] == 1 else w[..., c]  # (n_proj, det_x, det_y)
        valid_c = w_c > 0

        # Per-channel robust data range using *valid* ground truth pixels only
        gt_valid = gt_c[valid_c]
        if gt_valid.size == 0:
            continue
        channel_data_range = _percentile_data_range(gt_valid, percentile_low, percentile_high)

        # Weighted MSE: sum(w * err²) / sum(w)
        sq_err_c = (pred_c - gt_c) ** 2
        w_sum_c = w_c[valid_c].sum()
        mse_per_channel[c] = float(np.sum(w_c[valid_c] * sq_err_c[valid_c]) / w_sum_c)

        # PSNR from weighted MSE
        psnr_per_channel[c] = (
            float('inf') if mse_per_channel[c] == 0
            else 10 * np.log10((channel_data_range ** 2) / mse_per_channel[c])
        )

        # SSIM: vectorized over all projections at once.
        # Replace bad pixels with gt so they contribute zero local error.
        WIN = 7
        C1 = (0.01 * channel_data_range) ** 2
        C2 = (0.03 * channel_data_range) ** 2
        sz = (1, WIN, WIN)  # filter along det_x, det_y only; leave proj axis untouched
        p_vol = pred_c.copy()                       # (n_proj, det_x, det_y)
        g_vol = gt_c
        p_vol[~valid_c] = g_vol[~valid_c]           # mask bad pixels
        mu1 = _scipy_uniform_filter(p_vol, size=sz, mode='reflect')
        mu2 = _scipy_uniform_filter(g_vol, size=sz, mode='reflect')
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1_sq = np.maximum(_scipy_uniform_filter(p_vol ** 2, size=sz, mode='reflect') - mu1_sq, 0)
        s2_sq = np.maximum(_scipy_uniform_filter(g_vol ** 2, size=sz, mode='reflect') - mu2_sq, 0)
        s12   = _scipy_uniform_filter(p_vol * g_vol, size=sz, mode='reflect') - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
        ssim_per_channel[c] = float(np.mean(ssim_map))

    # Global metrics across all channels
    # Weighted global MSE
    sq_err_all = (proj_pred - proj_gt) ** 2
    w_sum_all = w[valid].sum()
    mse_global = float(np.sum(w[valid] * sq_err_all[valid]) / w_sum_all) if w_sum_all > 0 else 0.0
    global_data_range = _percentile_data_range(proj_gt[valid], percentile_low, percentile_high)
    psnr_global = (
        float('inf') if mse_global == 0
        else 10 * np.log10((global_data_range ** 2) / mse_global)
    )
    ssim_global = float(np.mean(ssim_per_channel))

    return ProjectionMetricsResult(
        mse_per_channel=mse_per_channel,
        mse_global=mse_global,
        psnr_per_channel=psnr_per_channel,
        psnr_global=psnr_global,
        ssim_per_channel=ssim_per_channel,
        ssim_global=ssim_global,
        n_channels=n_channels,
    )


def orientation_similarity(
    reconstruction_pred: np.ndarray,
    reconstruction_gt: np.ndarray,
    ell_max: int = 8,
    mask: Optional[np.ndarray] = None,
    min_intensity_threshold: float = 0.01,
) -> OrientationMetricsResult:
    """
    Compute orientation similarity based on principal eigenvector directions.
    
    This metric extracts the principal orientation (first eigenvector) from
    the spherical harmonics representation and computes cosine similarity
    between predicted and ground truth orientations.
    
    Parameters
    ----------
    reconstruction_pred : np.ndarray
        Predicted reconstruction with shape (x, y, z, n_coefficients).
    reconstruction_gt : np.ndarray
        Ground truth reconstruction with shape (x, y, z, n_coefficients).
    ell_max : int
        Maximum spherical harmonic order (default: 8).
    mask : np.ndarray, optional
        Binary mask of shape (x, y, z) to select voxels for comparison.
    min_intensity_threshold : float
        Minimum intensity threshold (relative to max) to consider a voxel valid.
        Voxels with intensity below this are excluded from orientation comparison.
        
    Returns
    -------
    OrientationMetricsResult
        Container with cosine similarity statistics and angular error.
    """
    # Create basis set for output computation
    basis_set = SphericalHarmonics(ell_max=ell_max)
    
    # Apply mask if provided
    if mask is not None:
        pred_masked = reconstruction_pred * mask[:, :, :, None]
        gt_masked = reconstruction_gt * mask[:, :, :, None]
    else:
        pred_masked = reconstruction_pred
        gt_masked = reconstruction_gt
    
    # Get principal directions using mumott's output functionality
    output_pred = basis_set.get_output(pred_masked.astype(np.float64))
    output_gt = basis_set.get_output(gt_masked.astype(np.float64))
    
    directions_pred = output_pred.eigenvector_1
    directions_gt = output_gt.eigenvector_1
    
    # Get intensity for masking low-signal voxels
    intensity_pred = pred_masked[..., 0]  # First coefficient is usually intensity
    intensity_gt = gt_masked[..., 0]
    
    # Create validity mask based on intensity
    max_intensity = max(intensity_gt.max(), 1e-10)
    intensity_mask = (intensity_gt > min_intensity_threshold * max_intensity)
    
    if mask is not None:
        validity_mask = (mask > 0) & intensity_mask
    else:
        validity_mask = intensity_mask
    
    # Extract valid directions
    valid_indices = np.where(validity_mask)
    
    if len(valid_indices[0]) == 0:
        return OrientationMetricsResult(
            cosine_similarity_mean=0.0,
            cosine_similarity_std=0.0,
            cosine_similarity_map=np.zeros_like(validity_mask, dtype=np.float32),
            angular_error_mean_degrees=90.0,
            angular_error_std_degrees=0.0,
            valid_voxels=0,
        )
    
    pred_dirs = directions_pred[valid_indices]
    gt_dirs = directions_gt[valid_indices]
    
    # Normalize directions
    pred_norms = np.linalg.norm(pred_dirs, axis=-1, keepdims=True)
    gt_norms = np.linalg.norm(gt_dirs, axis=-1, keepdims=True)
    
    # Avoid division by zero
    pred_norms = np.maximum(pred_norms, 1e-10)
    gt_norms = np.maximum(gt_norms, 1e-10)
    
    pred_dirs_normalized = pred_dirs / pred_norms
    gt_dirs_normalized = gt_dirs / gt_norms
    
    # Compute cosine similarity (take absolute value since directions are antipodal)
    cosine_sim = np.abs(np.sum(pred_dirs_normalized * gt_dirs_normalized, axis=-1))
    cosine_sim = np.clip(cosine_sim, 0, 1)  # Numerical stability
    
    # Create full cosine similarity map
    cosine_similarity_map = np.zeros(validity_mask.shape, dtype=np.float32)
    cosine_similarity_map[valid_indices] = cosine_sim
    
    # Compute angular error in degrees
    angular_error = np.arccos(cosine_sim) * 180 / np.pi
    
    return OrientationMetricsResult(
        cosine_similarity_mean=float(np.mean(cosine_sim)),
        cosine_similarity_std=float(np.std(cosine_sim)),
        cosine_similarity_map=cosine_similarity_map,
        angular_error_mean_degrees=float(np.mean(angular_error)),
        angular_error_std_degrees=float(np.std(angular_error)),
        valid_voxels=len(valid_indices[0]),
    )


def compare_reconstructions(
    reconstruction_pred: np.ndarray,
    reconstruction_gt: np.ndarray,
    geometry: Optional[Geometry] = None,
    ell_max: int = 8,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    compute_projection_metrics: bool = True,
    compute_orientation_metrics: bool = True,
    compute_real_space_metrics: bool = True,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    real_space_resolution_in_degrees: int = 10,
    real_space_half_sphere: bool = True,
    verbose: bool = False,
) -> ReconstructionComparisonResult:
    """
    Comprehensive comparison of two reconstructions using multiple metrics.
    
    Parameters
    ----------
    reconstruction_pred : np.ndarray
        Predicted reconstruction with shape (x, y, z, n_coefficients).
    reconstruction_gt : np.ndarray
        Ground truth reconstruction with shape (x, y, z, n_coefficients).
    geometry : Geometry, optional
        Geometry object for projection-based metrics. Required if
        compute_projection_metrics=True.
    ell_max : int
        Maximum spherical harmonic order (default: 8).
    mask : np.ndarray, optional
        Binary mask of shape (x, y, z) to select voxels for comparison.
    weights : np.ndarray, optional
        Pixel weights from ``dc.projections.weights``, shape
        ``(n_projections, det_x, det_y)`` or
        ``(n_projections, det_x, det_y, n_channels)``.  Passed directly to
        :func:`projection_metrics`.  Pixels with weight 0 are excluded.
    compute_projection_metrics : bool
        Whether to compute projection-based metrics (default: True).
    compute_orientation_metrics : bool
        Whether to compute orientation-based metrics (default: True).
    compute_real_space_metrics : bool
        Whether to compute PSNR/SSIM on the real-valued SH function evaluated
        on a spherical grid (default: True).
    percentile_low : float
        Lower percentile for robust data range in PSNR/SSIM (default: 5.0).
    percentile_high : float
        Upper percentile for robust data range in PSNR/SSIM (default: 95.0).
    real_space_resolution_in_degrees : int
        Angular resolution of the SH evaluation grid in degrees (default: 10).
    real_space_half_sphere : bool
        If True (default) evaluate only the upper hemisphere.
        
    Returns
    -------
    ReconstructionComparisonResult
        Container with all computed metrics.
        
    Example
    -------
    >>> result = compare_reconstructions(
    ...     reconstruction_pred=results_new['x'],
    ...     reconstruction_gt=results['x'],
    ...     geometry=dc.geometry,
    ...     ell_max=8,
    ... )
    >>> print(f"MSE: {result.mse_coefficients:.6f}")
    >>> print(f"Normalized MSE: {result.normalized_mse_coefficients:.6f}")
    >>> print(f"Projection PSNR: {result.projection_metrics.psnr_global:.2f} dB")
    >>> print(f"Orientation similarity: {result.orientation_metrics.cosine_similarity_mean:.4f}")
    """
    # Coefficient-based metrics
    if verbose:
        print("[1/4] Coefficient-based metrics ...")
    mse = mse_coefficients(reconstruction_pred, reconstruction_gt, mask)
    normalized_mse = normalized_mse_coefficients(
        reconstruction_pred, reconstruction_gt, mask, ell_max
    )
    if verbose:
        print(f"      MSE={mse:.4e}  norm-MSE={normalized_mse:.4e}")

    # Projection-based metrics
    proj_metrics = None
    if compute_projection_metrics:
        if geometry is None:
            raise ValueError("geometry is required for projection-based metrics")
        n_proj = len(geometry.inner_angles)
        if verbose:
            print(f"[2/4] Projection-based metrics  ({n_proj} projections) ...")
        proj_metrics = projection_metrics(
            reconstruction_pred, reconstruction_gt, geometry, ell_max, mask,
            weights=weights,
            percentile_low=percentile_low, percentile_high=percentile_high,
            verbose=verbose,
        )
        if verbose:
            print(f"      PSNR={proj_metrics.psnr_global:.2f} dB  "
                  f"SSIM={proj_metrics.ssim_global:.4f}")
    elif verbose:
        print("[2/4] Projection-based metrics  (skipped)")

    # Orientation-based metrics
    orient_metrics = None
    if compute_orientation_metrics:
        if verbose:
            print("[3/4] Orientation-based metrics ...")
        orient_metrics = orientation_similarity(
            reconstruction_pred, reconstruction_gt, ell_max, mask
        )
        if verbose:
            print(f"      cosine_sim={orient_metrics.cosine_similarity_mean:.4f}  "
                  f"angular_err={orient_metrics.angular_error_mean_degrees:.2f}°")
    elif verbose:
        print("[3/4] Orientation-based metrics  (skipped)")

    # Real-space metrics
    rs_metrics = None
    if compute_real_space_metrics:
        if verbose:
            print(f"[4/4] Real-space SH metrics  "
                  f"(resolution={real_space_resolution_in_degrees}°) ...")
        rs_metrics = real_space_metrics(
            reconstruction_pred, reconstruction_gt,
            ell_max=ell_max, mask=mask,
            resolution_in_degrees=real_space_resolution_in_degrees,
            map_half_sphere=real_space_half_sphere,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            verbose=verbose,
        )
        if verbose:
            print(f"      PSNR={rs_metrics.psnr:.2f} dB  SSIM={rs_metrics.ssim:.4f}")
    elif verbose:
        print("[4/4] Real-space SH metrics  (skipped)")

    return ReconstructionComparisonResult(
        mse_coefficients=mse,
        normalized_mse_coefficients=normalized_mse,
        projection_metrics=proj_metrics,
        orientation_metrics=orient_metrics,
        real_space_metrics=rs_metrics,
    )


def print_comparison_results(result: ReconstructionComparisonResult) -> None:
    """
    Pretty-print the comparison results.
    
    Parameters
    ----------
    result : ReconstructionComparisonResult
        The comparison results to print.
    """
    print("=" * 60)
    print("RECONSTRUCTION COMPARISON RESULTS")
    print("=" * 60)
    
    print("\n--- Coefficient-based Metrics ---")
    print(f"  MSE (raw):        {result.mse_coefficients:.6e}")
    print(f"  MSE (normalized): {result.normalized_mse_coefficients:.6e}")
    
    if result.projection_metrics is not None:
        pm = result.projection_metrics
        print(f"\n--- Projection-based Metrics ({pm.n_channels} channels) ---")
        print(f"  Global MSE:  {pm.mse_global:.6e}")
        print(f"  Global PSNR: {pm.psnr_global:.2f} dB")
        print(f"  Global SSIM: {pm.ssim_global:.4f}")
        print(f"\n  Per-channel PSNR: {np.array2string(pm.psnr_per_channel, precision=2)} dB")
        print(f"  Per-channel SSIM: {np.array2string(pm.ssim_per_channel, precision=4)}")
    
    if result.orientation_metrics is not None:
        om = result.orientation_metrics
        print(f"\n--- Orientation-based Metrics ---")
        print(f"  Valid voxels:              {om.valid_voxels}")
        print(f"  Cosine similarity (mean):  {om.cosine_similarity_mean:.4f}")
        print(f"  Cosine similarity (std):   {om.cosine_similarity_std:.4f}")
        print(f"  Angular error (mean):      {om.angular_error_mean_degrees:.2f}°")
        print(f"  Angular error (std):       {om.angular_error_std_degrees:.2f}°")

    if result.real_space_metrics is not None:
        rs = result.real_space_metrics
        print(f"\n--- Real-Space SH Function Metrics ---")
        print(f"  Grid resolution:  {rs.resolution_in_degrees}°  ({rs.n_directions} directions)")
        print(f"  MSE:              {rs.mse:.6e}")
        print(f"  PSNR:             {rs.psnr:.2f} dB")
        print(f"  SSIM:             {rs.ssim:.4f}")

    print("=" * 60)
