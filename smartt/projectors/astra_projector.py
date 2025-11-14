"""PyTorch wrapper for ASTRA projections using mumott Geometry.

This module provides GPU-friendly PyTorch functions for tomographic projection
using ASTRA toolbox with mumott geometry definitions.

Public functions
----------------
forward_project(...):
    Perform forward projection of a volume using mumott Geometry.

backproject(...):
    Perform backprojection (adjoint) of projections using mumott Geometry.

fbp_reconstruction(...):
    Perform FBP (Filtered Back Projection) reconstruction using mumott Geometry.

gd_reconstruction(...):
    Perform gradient descent reconstruction using mumott Geometry.

build_mumott_projector(...):
    Factory that returns a torch.autograd.Function for differentiable projection.

Notes
-----
* Requires the ASTRA toolbox compiled with CUDA support.
* All tensors are float32 and (by default) CUDA if available.
* Volume tensor layout is (X, Y, Z) or (X, Y, Z, B) for batched operations.
* Projection tensor layout is (I, J, K) or (I, J, K, B) where I is projection index, 
  J and K are detector dimensions, and B is batch size.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

try:
    import astra  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("astra toolbox is required for smartt.projectors.astra_projector module") from e

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry setup helpers
# ---------------------------------------------------------------------------

def _create_astra_geometries_from_mumott(geometry):
    """Create ASTRA geometries from mumott Geometry object.
    
    Parameters
    ----------
    geometry : mumott.Geometry
        mumott Geometry object containing the projection geometry
        
    Returns
    -------
    vol_geom : dict
        ASTRA volume geometry
    proj_geom : dict
        ASTRA projection geometry
        
    Notes
    -----
    This function transforms mumott's geometric representation to ASTRA's format.
    mumott uses basis vectors (projection direction, j-direction, k-direction) 
    for each projection angle, which maps to ASTRA's parallel3d_vec format.
    """
    # Get volume shape from geometry
    # ASTRA expects (nx, ny, nz) but mumott uses (x, y, z)
    vol_shape = geometry.volume_shape
    
    # Create ASTRA volume geometry
    # ASTRA syntax is (y, x, z) - see https://www.astra-toolbox.com/docs/geom3d.html
    vol_geom = astra.create_vol_geom(vol_shape[1], vol_shape[0], vol_shape[2])
    
    # Get projection shape
    proj_shape = geometry.projection_shape  # (j, k) in mumott
    
    # Get number of projections
    n_projections = len(geometry)
    
    # Create ASTRA vector geometry
    # For parallel3d_vec, each row contains 12 values:
    # [ray_x, ray_y, ray_z, det_x, det_y, det_z, u_x, u_y, u_z, v_x, v_y, v_z]
    # where:
    # - ray: direction of the parallel beam
    # - det: center of the detector
    # - u: detector pixel direction (horizontal)
    # - v: detector pixel direction (vertical)
    
    astra_vec = np.zeros((n_projections, 12))
    
    # Compute basis vectors from geometry (same way as SAXSProjector does)
    # These are computed by rotating the standard basis vectors according to the geometry
    from mumott.methods.projectors import SAXSProjector
    temp_projector = SAXSProjector(geometry)
    basis_vector_projection = temp_projector._basis_vector_projection  # Ray direction
    basis_vector_j = temp_projector._basis_vector_j  # Detector u-direction
    basis_vector_k = temp_projector._basis_vector_k  # Detector v-direction
    
    # Compute detector center shifts from offsets
    j_offsets = geometry.j_offsets_as_array
    k_offsets = geometry.k_offsets_as_array
    shift_in_xyz = basis_vector_j * j_offsets[:, None] + basis_vector_k * k_offsets[:, None]
    
    # Build ASTRA vectors
    # Ray direction (parallel beam)
    astra_vec[:, 0:3] = basis_vector_projection
    
    # Detector center position (can be arbitrary for parallel beam, we use shifts)
    astra_vec[:, 3:6] = shift_in_xyz
    
    # Detector u-direction (j-direction)
    astra_vec[:, 6:9] = basis_vector_j
    
    # Detector v-direction (k-direction)
    astra_vec[:, 9:12] = basis_vector_k
    
    # Create ASTRA projection geometry
    proj_geom = astra.create_proj_geom(
        'parallel3d_vec',
        proj_shape[1],  # detector rows (k-dimension)
        proj_shape[0],  # detector cols (j-dimension)
        astra_vec
    )
    
    return vol_geom, proj_geom


# ---------------------------------------------------------------------------
# Filtering utilities for FBP
# ---------------------------------------------------------------------------

def _apply_ramp_filter(
    sino_kij: torch.Tensor,
    filter_type: str = 'ram-lak',
    det_spacing: float = 1.0,
) -> torch.Tensor:
    """Apply ramp filter to sinogram for FBP reconstruction.
    
    This function applies the ramp filter in the frequency domain to all projections
    simultaneously for maximum GPU efficiency. The filter can be windowed to reduce
    noise and artifacts.
    
    Parameters
    ----------
    sino_kij : torch.Tensor
        Sinogram with shape (K, I, J) where K=det_rows, I=n_angles, J=det_cols
        or batched (B, K, I, J)
    filter_type : str, optional
        Filter type: 'ram-lak' (default), 'shepp-logan', 'cosine', 'hamming', 'hann'
        - 'ram-lak': Pure ramp filter (no windowing)
        - 'shepp-logan': Multiplies ramp by sinc function
        - 'cosine': Multiplies ramp by cosine
        - 'hamming': Hamming window on ramp
        - 'hann': Hann window on ramp
    det_spacing : float
        Detector pixel spacing
        
    Returns
    -------
    filtered_sino : torch.Tensor
        Filtered sinogram with same shape as input
        
    Notes
    -----
    The ramp filter is the derivative operator in Fourier space, which compensates
    for the 1/r weighting in backprojection. Windowing reduces high-frequency noise
    at the cost of slightly reduced resolution.
    """
    device = sino_kij.device
    batched = sino_kij.ndim == 4
    
    if batched:
        B, K, I, J = sino_kij.shape
    else:
        K, I, J = sino_kij.shape

    if det_spacing <= 0:
        raise ValueError("det_spacing must be positive for ramp filtering")

    # Pad to next power-of-two to avoid wrap-around
    padded_cols = int(2 ** math.ceil(math.log2(max(64, 2 * J))))
    pad_cols = padded_cols - J

    # Remove mean along detector columns to suppress low-frequency bias
    sino_zero_mean = sino_kij - torch.mean(sino_kij, dim=-1, keepdim=True)

    if pad_cols > 0:
        # Pad along last dimension (detector columns)
        sino_padded = torch.nn.functional.pad(sino_zero_mean, (0, pad_cols), mode='replicate')
    else:
        sino_padded = sino_zero_mean

    # Work in Fourier domain along detector axis using rFFT for efficiency
    sino_fft = torch.fft.rfft(sino_padded, dim=-1)

    # Create frequency array
    freq = torch.fft.rfftfreq(padded_cols, d=det_spacing, device=device)
    freq_abs = torch.abs(freq)
    freq_abs[0] = 0.0

    freq_max = float(freq_abs.max()) if freq_abs.numel() > 0 else 1.0
    if freq_max == 0:
        freq_max = 1.0
    freq_norm = freq_abs / freq_max

    # Create window
    window = torch.ones_like(freq_abs)
    ftype = filter_type.lower()

    if ftype == 'ram-lak':
        pass
    elif ftype == 'shepp-logan':
        window[1:] = torch.sinc(freq_norm[1:])
    elif ftype == 'cosine':
        window = torch.cos((math.pi / 2.0) * freq_norm)
    elif ftype == 'hamming':
        window = 0.54 + 0.46 * torch.cos(math.pi * freq_norm)
    elif ftype == 'hann':
        window = 0.5 * (1.0 + torch.cos(math.pi * freq_norm))
    else:
        raise ValueError(
            f"Unknown filter type: {filter_type}. Choose from: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'"
        )

    # Build ramp filter
    ramp = 2.0 * freq_abs
    filter_kernel = ramp * window

    # Apply filter (broadcast over batch and angle dimensions)
    if batched:
        filter_kernel = filter_kernel.view(1, 1, 1, -1)
    else:
        filter_kernel = filter_kernel.view(1, 1, -1)
    filtered_fft = sino_fft * filter_kernel

    # Inverse FFT
    filtered_sino = torch.fft.irfft(filtered_fft, n=padded_cols, dim=-1)

    # Remove padding
    if pad_cols > 0:
        filtered_sino = filtered_sino[..., :J]

    # Scale by detector sampling step to approximate continuous integral
    filtered_sino = filtered_sino * det_spacing

    return filtered_sino


# ---------------------------------------------------------------------------
# Core projection functions
# ---------------------------------------------------------------------------

def _forward_project_single_gpu(volume_slice: torch.Tensor, vol_geom, proj_geom, 
                                 proj_shape: Tuple[int, int], n_projections: int, 
                                 device: torch.device) -> torch.Tensor:
    """Forward project a single volume on GPU using CuPy."""
    # Ensure volume is contiguous and make it float32
    volume_slice = volume_slice.contiguous().float()
    
    # Create CuPy view of PyTorch tensor
    vol_cp = cp.ndarray(
        shape=volume_slice.shape,
        dtype=cp.float32,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(volume_slice.data_ptr(), volume_slice.numel() * 4, volume_slice),
            0
        )
    )
    # Transpose to ASTRA format (z, y, x) and make contiguous copy to avoid aliasing
    vol_astra = cp.transpose(vol_cp, (2, 1, 0))
    vol_astra = cp.ascontiguousarray(vol_astra).copy()  # Explicit copy to avoid memory aliasing
    
    # Allocate output - ASTRA sinogram format is (det_rows, n_angles, det_cols)
    proj_shape_astra = (proj_shape[1], n_projections, proj_shape[0])
    proj_astra = cp.zeros(proj_shape_astra, dtype=cp.float32)
    
    # Create ASTRA objects and run forward projection
    vol_id = astra.data3d.link('-vol', vol_geom, vol_astra)
    sino_id = astra.data3d.link('-sino', proj_geom, proj_astra)
    
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeDataId'] = vol_id
    cfg['ProjectionDataId'] = sino_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)
    
    # Cleanup ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(sino_id)
    astra.data3d.delete(vol_id)
    
    # Convert to PyTorch and transpose to (I, J, K)
    # ASTRA returns (det_rows, n_angles, det_cols) = (K, I, J)
    # We want (I, J, K)
    # Important: Make a COPY to avoid memory aliasing issues when used in batch loops
    proj_torch = torch.as_tensor(proj_astra, device=device).permute(1, 2, 0).clone()
    
    return proj_torch


def _forward_project_single_cpu(volume_slice: torch.Tensor, vol_geom, proj_geom,
                                 proj_shape: Tuple[int, int], n_projections: int,
                                 device: torch.device) -> torch.Tensor:
    """Forward project a single volume on CPU using NumPy."""
    vol_np = volume_slice.detach().cpu().numpy()
    # Transpose to ASTRA format (z, y, x)
    vol_astra = np.transpose(vol_np, (2, 1, 0)).copy()
    
    # Allocate output - ASTRA sinogram format is (det_rows, n_angles, det_cols)
    proj_shape_astra = (proj_shape[1], n_projections, proj_shape[0])
    proj_astra = np.zeros(proj_shape_astra, dtype=np.float32)
    
    # Create ASTRA objects and run forward projection
    vol_id = astra.data3d.create('-vol', vol_geom, vol_astra)
    sino_id = astra.data3d.create('-sino', proj_geom, proj_astra)
    
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeDataId'] = vol_id
    cfg['ProjectionDataId'] = sino_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)
    
    # Get result from ASTRA (CRITICAL: must retrieve before cleanup!)
    proj_astra = astra.data3d.get(sino_id)
    
    # Cleanup ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(sino_id)
    astra.data3d.delete(vol_id)
    
    # Convert to PyTorch and transpose to (I, J, K)
    # ASTRA returns (det_rows, n_angles, det_cols) = (K, I, J)
    # We want (I, J, K)
    proj_torch = torch.from_numpy(proj_astra).to(torch.float32).to(device)
    proj_torch = proj_torch.permute(1, 2, 0)  # (K, I, J) -> (I, J, K)
    
    return proj_torch


def forward_project(
    volume: torch.Tensor,
    geometry,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Forward project a volume using mumott Geometry.
    
    Parameters
    ----------
    volume : torch.Tensor
        Volume to project with shape (X, Y, Z) or (X, Y, Z, B) where B is batch size
    geometry : mumott.Geometry
        mumott Geometry object defining the projection geometry
    device : torch.device, optional
        Target device (default: volume.device)
        
    Returns
    -------
    projections : torch.Tensor
        Forward projections with shape (I, J, K) or (I, J, K, B)
        where I is the number of projections, J and K are detector dimensions,
        and B is the batch size (if input was batched)
        
    Notes
    -----
    This function uses ASTRA's FP3D_CUDA algorithm for GPU-accelerated forward projection.
    """
    if device is None:
        device = volume.device if isinstance(volume, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle batched input - convert to list of 3D volumes
    batched = volume.ndim == 4
    if batched:
        B = volume.shape[3]
        volume_list = [volume[..., b] for b in range(B)]  # List of (X, Y, Z) volumes
    elif volume.ndim == 3:
        volume_list = [volume]  # Single volume
    else:
        raise ValueError(f"Expected volume shape (X, Y, Z) or (X, Y, Z, B), got {volume.shape}")
    
    # Create ASTRA geometries
    vol_geom, proj_geom = _create_astra_geometries_from_mumott(geometry)
    
    # Get projection dimensions
    n_projections = len(geometry)
    proj_shape = geometry.projection_shape
    
    # Choose projection function based on device
    vol_device = volume_list[0].device
    if vol_device.type == 'cuda':
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is required for GPU-accelerated projection. "
                "Install it with: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)"
            )
        project_func = _forward_project_single_gpu
    else:
        project_func = _forward_project_single_cpu
    
    # Process each volume individually and stack results
    projs_list = []
    for vol in volume_list:
        proj_torch = project_func(vol, vol_geom, proj_geom, proj_shape, n_projections, device)
        projs_list.append(proj_torch)
        
        # Synchronize and cleanup between batches for GPU to avoid memory interference
        if vol_device.type == 'cuda' and CUPY_AVAILABLE and len(volume_list) > 1:
            cp.cuda.runtime.deviceSynchronize()
    
    # Stack results if batched, otherwise return single projection
    if batched:
        projections = torch.stack(projs_list, dim=3)  # Stack along last dimension: (I, J, K, B)
    else:
        projections = projs_list[0]  # (I, J, K)
    
    # Final cleanup CuPy memory if using GPU
    if vol_device.type == 'cuda' and CUPY_AVAILABLE:
        cp.cuda.runtime.deviceSynchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    return projections


def backproject(
    projections: torch.Tensor,
    geometry,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Backproject (adjoint) projections using mumott Geometry.
    
    Parameters
    ----------
    projections : torch.Tensor
        Projections to backproject with shape (I, J, K) or (I, J, K, B)
        where I is the number of projections, J and K are detector dimensions,
        and B is the batch size (if batched)
    geometry : mumott.Geometry
        mumott Geometry object defining the projection geometry
    device : torch.device, optional
        Target device (default: projections.device)
        
    Returns
    -------
    volume : torch.Tensor
        Backprojected volume with shape (X, Y, Z) or (X, Y, Z, B)
        
    Notes
    -----
    This function uses ASTRA's BP3D_CUDA algorithm for GPU-accelerated backprojection.
    """
    if device is None:
        device = projections.device if isinstance(projections, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle batched input
    batched = projections.ndim == 4
    if batched:
        B = projections.shape[3]
        proj_list = [projections[..., b] for b in range(B)]  # List of (I, J, K) projections
    elif projections.ndim == 3:
        proj_list = [projections]  # Single projection stack
    else:
        raise ValueError(f"Expected projections shape (I, J, K) or (I, J, K, B), got {projections.shape}")
    
    # Create ASTRA geometries
    vol_geom, proj_geom = _create_astra_geometries_from_mumott(geometry)
    
    # Get volume shape
    vol_shape = geometry.volume_shape
    
    # Choose backprojection function based on device
    proj_device = proj_list[0].device
    
    # Process each projection individually and stack results
    vols_list = []
    for proj in proj_list:
        # Convert to ASTRA format (K, I, J) = (det_rows, n_angles, det_cols)
        proj_astra = proj.permute(2, 0, 1).contiguous()
        
        if proj_device.type == 'cuda':
            if not CUPY_AVAILABLE:
                raise RuntimeError(
                    "CuPy is required for GPU-accelerated projection. "
                    "Install it with: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)"
                )
            # GPU path using CuPy
            # Create CuPy view
            proj_cp = cp.ndarray(
                shape=proj_astra.shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(proj_astra.data_ptr(), proj_astra.numel() * 4, proj_astra),
                    0
                )
            )
            
            # Allocate output volume in ASTRA format (z, y, x)
            vol_astra_shape = (vol_shape[2], vol_shape[1], vol_shape[0])
            vol_astra = cp.zeros(vol_astra_shape, dtype=cp.float32)
            
            # Create ASTRA objects and run backprojection
            vol_id = astra.data3d.link('-vol', vol_geom, vol_astra)
            sino_id = astra.data3d.link('-sino', proj_geom, proj_cp)
            
            cfg = astra.astra_dict('BP3D_CUDA')
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, 1)
            
            # Cleanup
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(sino_id)
            astra.data3d.delete(vol_id)
            
            # Convert to PyTorch and transpose to (X, Y, Z)
            vol_torch = torch.as_tensor(vol_astra, device=device).permute(2, 1, 0).clone()
        else:
            # CPU path
            proj_np = proj_astra.detach().cpu().numpy()
            
            # Allocate output volume in ASTRA format (z, y, x)
            vol_astra_shape = (vol_shape[2], vol_shape[1], vol_shape[0])
            vol_astra = np.zeros(vol_astra_shape, dtype=np.float32)
            
            # Create ASTRA objects and run backprojection
            vol_id = astra.data3d.create('-vol', vol_geom, vol_astra)
            sino_id = astra.data3d.create('-sino', proj_geom, proj_np)
            
            cfg = astra.astra_dict('BP3D_CUDA')
            cfg['ReconstructionDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, 1)
            
            # Get result
            vol_astra = astra.data3d.get(vol_id)
            
            # Cleanup
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(sino_id)
            astra.data3d.delete(vol_id)
            
            # Convert to PyTorch and transpose to (X, Y, Z)
            vol_torch = torch.from_numpy(vol_astra).to(torch.float32).to(device)
            vol_torch = vol_torch.permute(2, 1, 0)
        
        vols_list.append(vol_torch)
        
        # Synchronize between batches for GPU
        if proj_device.type == 'cuda' and CUPY_AVAILABLE and len(proj_list) > 1:
            cp.cuda.runtime.deviceSynchronize()
    
    # Stack results if batched, otherwise return single volume
    if batched:
        volume = torch.stack(vols_list, dim=3)  # Stack along last dimension: (X, Y, Z, B)
    else:
        volume = vols_list[0]  # (X, Y, Z)
    
    # Final cleanup CuPy memory if using GPU
    if proj_device.type == 'cuda' and CUPY_AVAILABLE:
        cp.cuda.runtime.deviceSynchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    
    return volume


# ---------------------------------------------------------------------------
# Reconstruction functions
# ---------------------------------------------------------------------------

def fbp_reconstruction(
    projections: torch.Tensor,
    geometry,
    filter_type: str = 'hann',
    det_spacing: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Perform FBP (Filtered Back Projection) reconstruction using mumott Geometry.
    
    FBP is an analytical reconstruction method that applies a ramp filter to the
    projections before backprojecting them. It's fast and provides good quality
    for well-sampled data.
    
    Parameters
    ----------
    projections : torch.Tensor
        Projections with shape (I, J, K) where I is number of projections,
        J and K are detector dimensions
    geometry : mumott.Geometry
        mumott Geometry object defining the projection geometry
    filter_type : str, optional
        Ramp filter type: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann' (default)
        'hann' provides good balance between noise and resolution
    det_spacing : float
        Detector pixel spacing (default: 1.0)
    device : torch.device, optional
        Target device (default: projections.device)
        
    Returns
    -------
    volume : torch.Tensor
        Reconstructed volume with shape (X, Y, Z)
        
    Notes
    -----
    FBP is particularly useful for:
    - Well-sampled data with good angular coverage
    - Fast reconstruction
    - When you need a good initial guess for iterative methods
    
    Examples
    --------
    >>> volume = fbp_reconstruction(projections, geometry, filter_type='hann')
    """
    if device is None:
        device = projections.device if isinstance(projections, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure projections are on the right device
    projections = projections.to(device)
    
    # Convert to ASTRA format (K, I, J) = (det_rows, n_angles, det_cols)
    # Input is (I, J, K)
    sino_kij = projections.permute(2, 0, 1)  # (I, J, K) -> (K, I, J)
    
    # Apply ramp filter
    filtered_sino = _apply_ramp_filter(sino_kij, filter_type=filter_type, det_spacing=det_spacing)
    
    # Convert back to (I, J, K) format for backprojection
    filtered_projs = filtered_sino.permute(1, 2, 0)  # (K, I, J) -> (I, J, K)
    
    # Backproject filtered sinogram
    volume = backproject(filtered_projs, geometry, device=device)
    
    # Normalize by pi / (2 * n_angles) following standard FBP scaling
    n_angles = len(geometry)
    norm_factor = math.pi / (2.0 * n_angles) if n_angles > 0 else 1.0
    volume = volume * norm_factor
    
    return volume


def gd_reconstruction(
    projections: torch.Tensor,
    geometry,
    num_iterations: int = 100,
    learning_rate: float = 1e-3,
    batch_size: Optional[int] = None,
    vol_init: Optional[torch.Tensor] = None,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    optimizer_type: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Perform gradient descent reconstruction using mumott Geometry.
    
    This function performs iterative reconstruction by minimizing the mean squared error
    between measured projections and forward projections of the volume using mini-batches.
    
    Parameters
    ----------
    projections : torch.Tensor
        Measured projections with shape (I, J, K) where I is number of projections
    geometry : mumott.Geometry
        mumott Geometry object defining the projection geometry
    num_iterations : int
        Number of optimization iterations (default: 100)
    learning_rate : float
        Learning rate for optimizer (default: 1e-3)
    batch_size : int, optional
        Number of projections per mini-batch (default: all projections)
    vol_init : torch.Tensor, optional
        Initial volume guess (default: zeros)
    clamp_min : float
        Minimum value to clamp volume (default: 0.0 for non-negativity)
    clamp_max : float, optional
        Maximum value to clamp volume (default: None for no upper limit)
    optimizer_type : str
        Type of optimizer: "adam" (default) or "sgd"
    momentum : float
        Momentum for SGD optimizer (default: 0.9)
    weight_decay : float
        Weight decay (L2 regularization) parameter (default: 0.0)
    device : torch.device, optional
        Target device (default: projections.device)
    verbose : bool
        Show progress bar (default: False)
        
    Returns
    -------
    volume : torch.Tensor
        Reconstructed volume with shape (X, Y, Z)
        
    Notes
    -----
    Gradient descent reconstruction is useful for:
    - Noisy data
    - Limited angle tomography
    - Sparse view reconstruction
    - When you need constraints (non-negativity, value range)
    
    Examples
    --------
    >>> # Basic reconstruction with 200 iterations
    >>> volume = gd_reconstruction(projections, geometry, num_iterations=200)
    >>> 
    >>> # With custom initialization (e.g., from FBP)
    >>> fbp_vol = fbp_reconstruction(projections, geometry)
    >>> volume = gd_reconstruction(projections, geometry, vol_init=fbp_vol, num_iterations=50)
    """
    if device is None:
        device = projections.device if isinstance(projections, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure projections are on the right device
    projections = projections.to(device)
    
    # Get volume shape from geometry
    vol_shape = geometry.volume_shape
    
    # Initialize volume
    if vol_init is not None:
        if vol_init.ndim == 4:
            # Batched: (X, Y, Z, B) - take first
            recon = vol_init[..., 0].clone().to(device)
        elif vol_init.ndim == 3:
            # Single: (X, Y, Z)
            recon = vol_init.clone().to(device)
        else:
            raise ValueError(f"vol_init must have shape (X,Y,Z) or (X,Y,Z,B), got {vol_init.shape}")
    else:
        # Default: zero initialization
        recon = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    
    # Volume stays as (X, Y, Z) for the new projector API
    recon.requires_grad_(True)
    
    # Build differentiable projector
    projector = build_mumott_projector(geometry, device=device)
    
    # Setup optimizer
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam([recon], lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD([recon], lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose 'adam' or 'sgd'")
    
    # Prepare measurements
    meas = projections.to(device)  # (I, J, K)
    n_projections = meas.shape[0]
    
    # Setup batch size
    if batch_size is None:
        batch_size = n_projections
    
    # Training loop
    losses = []
    iterator = range(num_iterations)
    if verbose:
        iterator = tqdm(iterator, desc="GD Reconstruction")
    
    for iteration in iterator:
        # Sample random batch of projections
        if batch_size < n_projections:
            indices = torch.randperm(n_projections, device='cpu')[:batch_size].numpy()
            meas_batch = meas[indices]
            
            # Create subset geometry with only selected angles
            # Extract GeometryTuples for the batch
            batch_geom_tuples = [geometry[int(i)] for i in indices]
            
            # Build a temporary geometry-like object that contains only the batch
            # We'll create ASTRA geometries directly for this batch
            from mumott import Geometry as MumottGeometry
            
            # Create a minimal geometry-like object with the subset
            class BatchGeometry:
                def __init__(self, base_geometry, indices):
                    self.volume_shape = base_geometry.volume_shape
                    self.projection_shape = base_geometry.projection_shape
                    self._indices = indices
                    self._base_geometry = base_geometry
                    self._geom_tuples = [base_geometry[int(i)] for i in indices]
                    
                    # Delegate direction properties from base geometry
                    self.p_direction_0 = base_geometry.p_direction_0
                    self.j_direction_0 = base_geometry.j_direction_0
                    self.k_direction_0 = base_geometry.k_direction_0
                
                def __len__(self):
                    return len(self._indices)
                
                def __getitem__(self, idx):
                    return self._geom_tuples[idx]
                
                @property
                def rotations(self):
                    return [g.rotation for g in self._geom_tuples]
                
                @property
                def rotations_as_array(self):
                    """Return rotation matrices as numpy array (N, 3, 3)"""
                    return np.array([g.rotation for g in self._geom_tuples])
                
                @property
                def j_offsets(self):
                    return [g.j_offset for g in self._geom_tuples]
                
                @property
                def k_offsets(self):
                    return [g.k_offset for g in self._geom_tuples]
                
                @property
                def j_offsets_as_array(self):
                    return np.array(self.j_offsets)
                
                @property
                def k_offsets_as_array(self):
                    return np.array(self.k_offsets)
            
            batch_geometry = BatchGeometry(geometry, indices)
            
            # Build projector for this batch
            batch_projector = build_mumott_projector(batch_geometry, device=device)
            
            # Forward projection (only for selected angles)
            pred_batch = batch_projector(recon)  # (batch_size, J, K)
        else:
            meas_batch = meas
            
            # Forward projection (all angles)
            pred_batch = projector(recon)  # (I, J, K)
        
        # Compute loss (MSE)
        loss = torch.mean((pred_batch - meas_batch) ** 2)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Apply constraints
        with torch.no_grad():
            if clamp_min is not None or clamp_max is not None:
                recon.clamp_(min=clamp_min, max=clamp_max)
        
        # Update progress bar
        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # Extract final volume - already in correct shape (X, Y, Z)
    final_vol = recon.detach()
    
    return final_vol


# ---------------------------------------------------------------------------
# Differentiable projector
# ---------------------------------------------------------------------------

class _AstraMumottOp:
    """ASTRA forward/adjoint operator using mumott geometry."""
    
    def __init__(self, geometry):
        self.geometry = geometry
        self.vol_geom, self.proj_geom = _create_astra_geometries_from_mumott(geometry)
        self.vol_shape = geometry.volume_shape
        self.proj_shape = geometry.projection_shape
        self.n_projections = len(geometry)
    
    def forward(self, vol_t):
        """Forward projection: volume -> sinogram."""
        # vol_t is (X, Y, Z)
        # Need to convert to ASTRA format (Z, Y, X)
        if vol_t.is_cuda:
            vol_cp = cp.ndarray(
                shape=vol_t.shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(vol_t.data_ptr(), vol_t.numel() * 4, vol_t),
                    0
                )
            )
            vol_astra = cp.transpose(vol_cp, (2, 1, 0))
            vol_astra = cp.ascontiguousarray(vol_astra)
            
            # ASTRA sinogram format is (det_rows, n_angles, det_cols)
            proj_shape_astra = (self.proj_shape[1], self.n_projections, self.proj_shape[0])
            proj_astra = cp.zeros(proj_shape_astra, dtype=cp.float32)
            
            vol_id = astra.data3d.link('-vol', self.vol_geom, vol_astra)
            sino_id = astra.data3d.link('-sino', self.proj_geom, proj_astra)
        else:
            vol_np = vol_t.detach().cpu().numpy()
            vol_astra = np.transpose(vol_np, (2, 1, 0)).copy()
            
            # ASTRA sinogram format is (det_rows, n_angles, det_cols)
            proj_shape_astra = (self.proj_shape[1], self.n_projections, self.proj_shape[0])
            proj_astra = np.zeros(proj_shape_astra, dtype=np.float32)
            
            vol_id = astra.data3d.create('-vol', self.vol_geom, vol_astra)
            sino_id = astra.data3d.create('-sino', self.proj_geom, proj_astra)
        
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = vol_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)
        
        # Convert result back - proj_astra is (K, I, J), need (I, J, K)
        # ASTRA returns (det_rows, n_angles, det_cols) = (K, I, J)
        if vol_t.is_cuda:
            out = torch.as_tensor(proj_astra, device=vol_t.device).permute(1, 2, 0)
        else:
            if not isinstance(proj_astra, np.ndarray):
                proj_astra = astra.data3d.get(sino_id)
            out = torch.from_numpy(proj_astra).to(torch.float32).to(vol_t.device)
            out = out.permute(1, 2, 0)  # (K, I, J) -> (I, J, K)
        
        return out

    def adjoint(self, sino_t):
        """Adjoint/backprojection: sinogram -> volume."""
        # sino_t is (I, J, K)
        # Need to convert to ASTRA format (K, I, J) = (det_rows, n_angles, det_cols)
        sino_astra = sino_t.permute(2, 0, 1).contiguous()
        
        if sino_t.is_cuda:
            sino_cp = cp.ndarray(
                shape=sino_astra.shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(sino_astra.data_ptr(), sino_astra.numel() * 4, sino_astra),
                    0
                )
            )
            
            vol_astra_shape = (self.vol_shape[2], self.vol_shape[1], self.vol_shape[0])
            vol_astra = cp.zeros(vol_astra_shape, dtype=cp.float32)
            
            vol_id = astra.data3d.link('-vol', self.vol_geom, vol_astra)
            sino_id = astra.data3d.link('-sino', self.proj_geom, sino_cp)
        else:
            sino_np = sino_astra.detach().cpu().numpy()
            
            vol_astra_shape = (self.vol_shape[2], self.vol_shape[1], self.vol_shape[0])
            vol_astra = np.zeros(vol_astra_shape, dtype=np.float32)
            
            vol_id = astra.data3d.create('-vol', self.vol_geom, vol_astra)
            sino_id = astra.data3d.create('-sino', self.proj_geom, sino_np)
        
        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)
        
        # Convert result back - vol_astra is (Z, Y, X), need (X, Y, Z)
        if sino_t.is_cuda:
            out = torch.as_tensor(vol_astra, device=sino_t.device).permute(2, 1, 0)
        else:
            if not isinstance(vol_astra, np.ndarray):
                vol_astra = astra.data3d.get(vol_id)
            out = torch.from_numpy(vol_astra).to(torch.float32).to(sino_t.device)
            out = out.permute(2, 1, 0)
        
        return out


def build_mumott_projector(
    geometry,
    device: Optional[torch.device] = None,
):
    """Return a differentiable projection layer using mumott Geometry.

    Returns a function handle that acts like: y = projector(x) where
    x is (X, Y, Z, B) and y is (I, J, K, B). Autograd is supported.
    Both single and batched inputs are supported.
    
    Parameters
    ----------
    geometry : mumott.Geometry
        mumott Geometry object defining the projection geometry
    device : torch.device, optional
        Compute device
        
    Returns
    -------
    layer : callable
        Function that performs differentiable projection
        
    Examples
    --------
    >>> from mumott import Geometry
    >>> projector = build_mumott_projector(geometry)
    >>> # Single volume
    >>> volume = torch.randn(65, 82, 65, requires_grad=True)
    >>> projections = projector(volume)  # Output: (240, 73, 100)
    >>> # Batched volumes
    >>> volumes = torch.randn(65, 82, 65, 5, requires_grad=True)
    >>> projections = projector(volumes)  # Output: (240, 73, 100, 5)
    >>> loss = projections.sum()
    >>> loss.backward()
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    op = _AstraMumottOp(geometry)

    class MumottProjectorFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            # Handle both (X, Y, Z) and (X, Y, Z, B)
            if x.ndim == 3:
                # Single volume
                proj_t = op.forward(x)  # (I, J, K)
                ctx.batched = False
                ctx.op = op
                return proj_t
            elif x.ndim == 4:
                # Batched volumes
                B = x.shape[3]
                y_out = []
                for i in range(B):
                    vol_t = x[..., i]  # (X, Y, Z)
                    proj_t = op.forward(vol_t)  # (I, J, K)
                    y_out.append(proj_t)
                y_out = torch.stack(y_out, dim=3)  # (I, J, K, B)
                ctx.batched = True
                ctx.op = op
                return y_out
            else:
                raise ValueError(f'Input must have shape (X, Y, Z) or (X, Y, Z, B), got {x.shape}')

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            op_local = ctx.op
            if not ctx.batched:
                # Single volume gradient
                g_vol = op_local.adjoint(grad_out)  # (X, Y, Z)
                return g_vol
            else:
                # Batched gradient
                B = grad_out.shape[3]
                g_vols = []
                for i in range(B):
                    g_y = grad_out[..., i]  # (I, J, K)
                    g_vol = op_local.adjoint(g_y)  # (X, Y, Z)
                    g_vols.append(g_vol)
                g_stack = torch.stack(g_vols, dim=3)  # (X, Y, Z, B)
                return g_stack

    def layer(x: torch.Tensor) -> torch.Tensor:
        return MumottProjectorFn.apply(x)

    return layer
