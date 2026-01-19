"""Native PyTorch projector using affine_grid and grid_sample.

This module provides a GPU-efficient tomographic projector that works entirely
in PyTorch, avoiding the overhead of ASTRA for sequential coefficient processing.

The key idea is to:
1. Rotate the 3D volume so the projection direction aligns with Z-axis
2. Sum along Z to get the projection (parallel beam approximation)
3. Batch all coefficients together for maximum GPU efficiency

Public functions
----------------
build_pytorch_projector(...):
    Factory that returns a differentiable projection layer.

PyTorchProjector:
    Class implementing the projector with full batching support.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def _rotation_matrix_from_vectors(from_vec: torch.Tensor, to_vec: torch.Tensor) -> torch.Tensor:
    """Compute rotation matrix that rotates from_vec to to_vec.
    
    Uses Rodrigues' rotation formula.
    
    Parameters
    ----------
    from_vec : torch.Tensor
        Source unit vector, shape (3,)
    to_vec : torch.Tensor
        Target unit vector, shape (3,)
        
    Returns
    -------
    R : torch.Tensor
        3x3 rotation matrix
    """
    # Normalize vectors
    from_vec = from_vec / (torch.norm(from_vec) + 1e-8)
    to_vec = to_vec / (torch.norm(to_vec) + 1e-8)
    
    # Cross product gives rotation axis
    v = torch.linalg.cross(from_vec, to_vec)
    
    # Dot product gives cosine of angle
    c = torch.dot(from_vec, to_vec)
    
    # Handle near-parallel vectors
    if torch.abs(c + 1.0) < 1e-6:
        # Vectors are anti-parallel, rotate 180 degrees around any perpendicular axis
        # Find a perpendicular vector
        if torch.abs(from_vec[0]) < 0.9:
            perp = torch.tensor([1.0, 0.0, 0.0], device=from_vec.device, dtype=from_vec.dtype)
        else:
            perp = torch.tensor([0.0, 1.0, 0.0], device=from_vec.device, dtype=from_vec.dtype)
        perp = perp - torch.dot(perp, from_vec) * from_vec
        perp = perp / torch.norm(perp)
        # 180 degree rotation around perp
        return 2 * torch.outer(perp, perp) - torch.eye(3, device=from_vec.device, dtype=from_vec.dtype)
    
    if torch.abs(c - 1.0) < 1e-6:
        # Vectors are parallel, no rotation needed
        return torch.eye(3, device=from_vec.device, dtype=from_vec.dtype)
    
    # Skew-symmetric cross-product matrix
    vx = torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], device=from_vec.device, dtype=from_vec.dtype)
    
    # Rodrigues formula: R = I + vx + vx^2 * (1 - c) / s^2
    s = torch.norm(v)
    R = torch.eye(3, device=from_vec.device, dtype=from_vec.dtype) + vx + torch.mm(vx, vx) * (1 - c) / (s * s + 1e-8)
    
    return R


def _create_rotation_matrices_from_geometry(geometry, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create rotation matrices from mumott geometry.
    
    For each projection angle, compute:
    1. Rotation matrix to align ray direction with Z-axis
    2. Detector u-direction (j-direction) after rotation
    3. Detector v-direction (k-direction) after rotation
    
    Parameters
    ----------
    geometry : mumott.Geometry
        mumott Geometry object
    device : torch.device
        Target device
        
    Returns
    -------
    rotation_matrices : torch.Tensor
        Shape (N, 3, 3) rotation matrices for each projection
    rotated_u : torch.Tensor
        Shape (N, 3) rotated detector u-directions
    rotated_v : torch.Tensor
        Shape (N, 3) rotated detector v-directions
    """
    # Get ASTRA geometry vectors directly - these are the ground truth
    from smartt.projectors.astra_projector import _create_astra_geometries_from_mumott
    vol_geom, proj_geom = _create_astra_geometries_from_mumott(geometry)
    vectors = proj_geom['Vectors']  # (N, 12)
    
    n_projections = vectors.shape[0]
    
    # ASTRA vector format: ray(0:3), det_center(3:6), u(6:9), v(9:12)
    # ray = direction of X-rays (projection direction)
    # u = detector row direction (horizontal on detector)
    # v = detector column direction (vertical on detector)
    
    rotation_matrices = torch.zeros((n_projections, 3, 3), device=device, dtype=torch.float32)
    rotated_u = torch.zeros((n_projections, 3), device=device, dtype=torch.float32)
    rotated_v = torch.zeros((n_projections, 3), device=device, dtype=torch.float32)
    
    for i in range(n_projections):
        ray = torch.tensor(vectors[i, 0:3], device=device, dtype=torch.float32)
        u = torch.tensor(vectors[i, 6:9], device=device, dtype=torch.float32)
        v = torch.tensor(vectors[i, 9:12], device=device, dtype=torch.float32)
        
        # Normalize vectors
        ray = ray / (torch.norm(ray) + 1e-8)
        u = u / (torch.norm(u) + 1e-8)
        v = v / (torch.norm(v) + 1e-8)
        
        # Build rotation matrix that transforms from standard axes to projection axes
        # After rotation: X -> u, Y -> v, Z -> ray
        # The inverse rotation (which we use for sampling) does: u -> X, v -> Y, ray -> Z
        R = torch.stack([u, v, ray], dim=0).T  # Columns are u, v, ray
        rotation_matrices[i] = R
        rotated_u[i] = u
        rotated_v[i] = v
    
    return rotation_matrices, rotated_u, rotated_v


def _build_affine_matrix_for_rotation(R: torch.Tensor, vol_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Build 3x4 affine matrix for grid_sample from rotation matrix.
    
    grid_sample expects coordinates in [-1, 1] range.
    We need to:
    1. Scale from [-1, 1] to volume coordinates
    2. Apply rotation around volume center
    3. Scale back to [-1, 1]
    
    Parameters
    ----------
    R : torch.Tensor
        3x3 rotation matrix
    vol_shape : tuple
        Volume shape (X, Y, Z)
        
    Returns
    -------
    theta : torch.Tensor
        3x4 affine transformation matrix for grid_sample
    """
    # For grid_sample, the affine matrix transforms output coordinates to input coordinates
    # We want to rotate the volume, which means we apply the inverse rotation to the sampling grid
    R_inv = R.T  # Inverse of rotation is transpose
    
    # Create 3x4 matrix (rotation only, no translation for center rotation)
    theta = torch.zeros(3, 4, device=R.device, dtype=R.dtype)
    theta[:, :3] = R_inv
    
    return theta


class PyTorchProjector:
    """Native PyTorch parallel beam projector using grid_sample.
    
    This projector rotates the volume to align the ray direction with Z,
    then sums along Z to compute the projection. All coefficients are
    processed in a single batched operation for maximum GPU efficiency.
    
    Parameters
    ----------
    geometry : mumott.Geometry
        mumott Geometry object defining the projection geometry
    device : torch.device, optional
        Compute device (default: CUDA if available)
    """
    
    def __init__(self, geometry, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.geometry = geometry
        self.n_projections = len(geometry)
        self.vol_shape = tuple(geometry.volume_shape)  # (X, Y, Z)
        self.proj_shape = tuple(geometry.projection_shape)  # (J, K) detector shape
        
        # Pre-compute rotation matrices for all projection angles
        self.rotation_matrices, self.rotated_u, self.rotated_v = \
            _create_rotation_matrices_from_geometry(geometry, device)
        
        # Pre-compute affine grids for each projection angle
        # This is the key optimization - we create the grids once
        self._precompute_grids()
        
        # Get detector offsets
        self.j_offsets = torch.tensor(
            geometry.j_offsets_as_array, device=device, dtype=torch.float32
        )
        self.k_offsets = torch.tensor(
            geometry.k_offsets_as_array, device=device, dtype=torch.float32
        )
    
    def _precompute_grids(self):
        """Pre-compute sampling grids for all projection angles."""
        X, Y, Z = self.vol_shape
        J, K = self.proj_shape
        
        # Create base grid in normalized coordinates [-1, 1]
        # For projection, we want to sample along rays parallel to Z after rotation
        # The output will be J x K detector pixels
        
        # Create detector coordinates
        j_coords = torch.linspace(-1, 1, J, device=self.device)
        k_coords = torch.linspace(-1, 1, K, device=self.device)
        z_coords = torch.linspace(-1, 1, Z, device=self.device)  # Sampling along depth
        
        # Store grids for each projection - shape (N, Z, K, J, 3)
        # grid_sample expects (N, D, H, W, 3) for 3D
        self.grids = []
        
        for i in range(self.n_projections):
            R = self.rotation_matrices[i]
            R_inv = R.T  # Apply inverse rotation to grid
            
            # Create 3D grid of sample points
            # We want to sample along rays parallel to rotated Z
            # For each detector pixel (j, k), sample at all z depths
            grid = torch.zeros(Z, K, J, 3, device=self.device, dtype=torch.float32)
            
            for zi, z in enumerate(z_coords):
                for ki, k in enumerate(k_coords):
                    for ji, j in enumerate(j_coords):
                        # Point in detector plane at depth z (after rotation)
                        # In rotated coordinates: x=j, y=k, z=z
                        point_rotated = torch.tensor([j, k, z], device=self.device, dtype=torch.float32)
                        
                        # Transform back to original coordinates
                        point_original = torch.mv(R_inv, point_rotated)
                        
                        # grid_sample expects (x, y, z) in range [-1, 1]
                        # Note: grid_sample uses (W, H, D) order = (x, y, z)
                        grid[zi, ki, ji, 0] = point_original[0]  # x
                        grid[zi, ki, ji, 1] = point_original[1]  # y
                        grid[zi, ki, ji, 2] = point_original[2]  # z
            
            self.grids.append(grid)
        
        # Stack all grids
        self.grids = torch.stack(self.grids, dim=0)  # (N, Z, K, J, 3)
    
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Forward projection: volume -> sinogram.
        
        Parameters
        ----------
        volume : torch.Tensor
            Input volume, shape (X, Y, Z) or (X, Y, Z, C) for multi-channel
            
        Returns
        -------
        projections : torch.Tensor
            Output projections, shape (N, J, K) or (N, J, K, C)
            where N is number of projection angles
        """
        # Handle batched (multi-channel) volumes
        if volume.ndim == 3:
            volume = volume.unsqueeze(-1)  # (X, Y, Z) -> (X, Y, Z, 1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        X, Y, Z, C = volume.shape
        N = self.n_projections
        J, K = self.proj_shape
        
        # grid_sample expects input as (N, C, D, H, W)
        # Our volume is (X, Y, Z, C), need to reshape to (1, C, Z, Y, X)
        # Then broadcast over N projections
        vol_for_sample = volume.permute(3, 2, 1, 0).unsqueeze(0)  # (1, C, Z, Y, X)
        vol_for_sample = vol_for_sample.expand(N, -1, -1, -1, -1)  # (N, C, Z, Y, X)
        
        # Reshape grid for batch processing: (N, Z, K, J, 3) -> already correct shape for grid_sample
        # grid_sample expects grid as (N, D_out, H_out, W_out, 3)
        # Our grid is (N, Z_samples, K, J, 3)
        
        # Sample the volume at grid points
        # This rotates the volume implicitly by sampling at rotated coordinates
        sampled = F.grid_sample(
            vol_for_sample,  # (N, C, Z, Y, X)
            self.grids,  # (N, Z, K, J, 3)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # Output: (N, C, Z, K, J)
        
        # Sum along the Z dimension (depth) to compute projection
        # This is the parallel beam projection integral
        projections = sampled.sum(dim=2)  # (N, C, K, J)
        
        # Reorder to (N, J, K, C)
        projections = projections.permute(0, 3, 2, 1)  # (N, J, K, C)
        
        if squeeze_output:
            projections = projections.squeeze(-1)
        
        return projections
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """Forward projection (callable interface)."""
        return self.forward(volume)


class PyTorchProjectorOptimized:
    """Optimized PyTorch projector with vectorized grid creation.
    
    This version creates grids more efficiently using tensor operations
    instead of nested loops.
    """
    
    def __init__(self, geometry, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.geometry = geometry
        self.n_projections = len(geometry)
        self.vol_shape = tuple(geometry.volume_shape)  # (X, Y, Z)
        self.proj_shape = tuple(geometry.projection_shape)  # (J, K) detector shape
        
        # Pre-compute rotation matrices
        self.rotation_matrices, self.rotated_u, self.rotated_v = \
            _create_rotation_matrices_from_geometry(geometry, device)
        
        # Pre-compute grids efficiently
        self._precompute_grids_from_astra()
    
    def _precompute_grids_from_astra(self):
        """Pre-compute sampling grids using ASTRA geometry vectors directly.
        
        For parallel beam projection, each ray is:
            point(t) = detector_point + t * ray_direction
        
        where:
            - detector_point = j * u + k * v + detector_center
            - j, k are detector coordinates (centered at 0)
            - t parameterizes the ray depth
        
        We sample along each ray and sum to get the projection.
        """
        from smartt.projectors.astra_projector import _create_astra_geometries_from_mumott
        vol_geom, proj_geom = _create_astra_geometries_from_mumott(self.geometry)
        vectors = proj_geom['Vectors']  # (N, 12)
        
        X, Y, Z = self.vol_shape
        J, K = self.proj_shape
        N = self.n_projections
        
        # Get volume extent from ASTRA vol_geom
        opt = vol_geom['option']
        x_min, x_max = float(opt['WindowMinX']), float(opt['WindowMaxX'])
        y_min, y_max = float(opt['WindowMinY']), float(opt['WindowMaxY'])
        z_min, z_max = float(opt['WindowMinZ']), float(opt['WindowMaxZ'])
        
        # Compute volume center and half-extent
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        x_extent = (x_max - x_min) / 2
        y_extent = (y_max - y_min) / 2
        z_extent = (z_max - z_min) / 2
        
        # Determine depth range for ray traversal
        # Use the diagonal to ensure we cover the entire volume
        max_extent = np.sqrt(x_extent**2 + y_extent**2 + z_extent**2) * 1.5
        
        # Number of samples along each ray (should be at least the diagonal of the volume)
        n_depth_samples = int(np.ceil(np.sqrt(X**2 + Y**2 + Z**2)))
        self.n_depth_samples = n_depth_samples
        
        # Create CENTERED detector coordinate arrays
        # ASTRA indexes pixels from 0 to J-1, but coordinates should be centered
        # Each pixel center is at offset (i - (J-1)/2) * pixel_size from detector center
        # Since u and v already include pixel size, we use integer offsets centered at 0
        j_coords = torch.linspace(-(J-1)/2, (J-1)/2, J, device=self.device, dtype=torch.float32)
        k_coords = torch.linspace(-(K-1)/2, (K-1)/2, K, device=self.device, dtype=torch.float32)
        
        # Depth samples along ray (centered at 0)
        t_coords = torch.linspace(-max_extent, max_extent, n_depth_samples, 
                                   device=self.device, dtype=torch.float32)
        
        # Store base coordinates for on-the-fly computation
        self.j_coords = j_coords
        self.k_coords = k_coords
        self.t_coords = t_coords
        self.vectors = torch.tensor(vectors, device=self.device, dtype=torch.float32)
        
        # Volume normalization factors for grid_sample (convert world coords to [-1, 1])
        # grid_sample maps [-1, 1] to the volume extent, so we need to scale world coords
        self.vol_scale = torch.tensor([x_extent, y_extent, z_extent], 
                                       device=self.device, dtype=torch.float32)
        self.vol_center = torch.tensor([x_center, y_center, z_center], 
                                        device=self.device, dtype=torch.float32)
        
        # Compute scale factor to match ASTRA's projection magnitude
        # When summing along rays, we sum over n_depth_samples points
        # The spacing between samples is: depth_step = 2 * max_extent / n_depth_samples
        # For ASTRA with unit voxel size, the integral is the sum weighted by voxel size (1.0)
        # Our sum should be weighted by depth_step to approximate the integral
        depth_step = 2 * max_extent / n_depth_samples
        self.scale_factor = depth_step  # Weight each sample by the step size
    
    def _compute_grid_batch(self, start_idx: int, end_idx: int) -> torch.Tensor:
        """Compute sampling grids for a batch of projections on-the-fly.
        
        Fully vectorized implementation using broadcasting instead of loops.
        """
        X, Y, Z = self.vol_shape
        J, K = self.proj_shape
        batch_size = end_idx - start_idx
        n_depth = self.n_depth_samples
        
        # Extract batch of vectors: (batch_size, 12)
        vecs_batch = self.vectors[start_idx:end_idx]
        
        # Extract components for the batch
        rays = vecs_batch[:, 0:3]  # (batch_size, 3)
        det_centers = vecs_batch[:, 3:6]  # (batch_size, 3)
        u_vecs = vecs_batch[:, 6:9]  # (batch_size, 3)
        v_vecs = vecs_batch[:, 9:12]  # (batch_size, 3)
        
        # Build meshgrid of detector coordinates and depth
        # Shape: (n_depth, K, J)
        t_grid, k_grid, j_grid = torch.meshgrid(
            self.t_coords, self.k_coords, self.j_coords, indexing='ij'
        )
        
        # Reshape for broadcasting: add batch dimension
        # t_grid, k_grid, j_grid: (1, n_depth, K, J)
        t_grid = t_grid.unsqueeze(0)
        k_grid = k_grid.unsqueeze(0)
        j_grid = j_grid.unsqueeze(0)
        
        # Reshape geometry vectors for broadcasting: (batch_size, 1, 1, 1, 3)
        rays = rays.view(batch_size, 1, 1, 1, 3)
        det_centers = det_centers.view(batch_size, 1, 1, 1, 3)
        u_vecs = u_vecs.view(batch_size, 1, 1, 1, 3)
        v_vecs = v_vecs.view(batch_size, 1, 1, 1, 3)
        
        # Compute 3D sample points using broadcasting:
        # point = det_center + j * u + k * v + t * ray
        # Shape: (batch_size, n_depth, K, J, 3)
        points = (det_centers +
                 j_grid.unsqueeze(-1) * u_vecs +
                 k_grid.unsqueeze(-1) * v_vecs +
                 t_grid.unsqueeze(-1) * rays)
        
        # Normalize to [-1, 1] for grid_sample
        # vol_center and vol_scale: (3,) -> reshape to (1, 1, 1, 1, 3)
        points_normalized = (points - self.vol_center.view(1, 1, 1, 1, 3)) / self.vol_scale.view(1, 1, 1, 1, 3)
        
        return points_normalized  # (batch_size, n_depth, K, J, 3)
    
    def forward(self, volume: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        """Forward projection with memory-efficient batching.
        
        Args:
            volume: Input volume (X, Y, Z) or (X, Y, Z, C)
            batch_size: Number of projections to process at once
        """
        # Ensure volume is on correct device
        if volume.device != self.device:
            volume = volume.to(self.device)
        
        # Handle batched (multi-channel) volumes
        if volume.ndim == 3:
            volume = volume.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        X, Y, Z, C = volume.shape
        N = self.n_projections
        J, K = self.proj_shape
        n_depth = self.n_depth_samples
        
        # Prepare volume for grid_sample: (1, C, Z, Y, X)
        # grid_sample expects input as (N, C, D, H, W) and grid as (N, D_out, H_out, W_out, 3)
        # The grid coordinates (x, y, z) map to (W, H, D) = (X, Y, Z) dimensions
        vol_for_sample = volume.permute(3, 2, 1, 0).unsqueeze(0)  # (1, C, Z, Y, X)
        
        # Process projections in batches to avoid OOM
        all_projections = []
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            current_batch = end_idx - start_idx
            
            # Compute grids for this batch: (batch, n_depth, K, J, 3)
            batch_grids = self._compute_grid_batch(start_idx, end_idx)
            
            # Expand volume for this batch
            vol_batch = vol_for_sample.expand(current_batch, -1, -1, -1, -1)
            
            # Sample: input (batch, C, Z, Y, X), grid (batch, n_depth, K, J, 3)
            # Output: (batch, C, n_depth, K, J)
            sampled = F.grid_sample(
                vol_batch,
                batch_grids,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False  # Use False for half-pixel alignment like ASTRA
            )
            
            # Sum along depth direction (n_depth) to get projection
            projections_batch = sampled.sum(dim=2)  # (batch, C, K, J)
            
            # Reorder to (batch, J, K, C)
            projections_batch = projections_batch.permute(0, 3, 2, 1)
            
            all_projections.append(projections_batch)
            
            # Free memory
            del batch_grids, sampled
        
        # Concatenate all batches
        projections = torch.cat(all_projections, dim=0)  # (N, J, K, C)
        
        # Apply scale factor to match ASTRA's projection magnitude
        projections = projections * self.scale_factor
        
        if squeeze_output:
            projections = projections.squeeze(-1)
        
        return projections
    
    def __call__(self, volume: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        return self.forward(volume, batch_size=batch_size)


def build_pytorch_projector(geometry, device=None, batch_size=8):
    """Factory function to create a differentiable PyTorch projector.
    
    Parameters
    ----------
    geometry : mumott.Geometry
        mumott Geometry object defining the projection geometry
    device : torch.device, optional
        Compute device (default: CUDA if available)
    batch_size : int
        Number of projections to process at once (default: 8)
        
    Returns
    -------
    projector : callable
        Function that performs differentiable projection.
        Input: volume tensor (X, Y, Z) or (X, Y, Z, C)
        Output: projection tensor (N, J, K) or (N, J, K, C)
        Supports torch.autograd for gradient computation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    proj = PyTorchProjectorOptimized(geometry, device)
    
    def projector_fn(volume):
        return proj.forward(volume, batch_size=batch_size)
    
    return projector_fn


class PyTorchProjectorModule(torch.nn.Module):
    """PyTorch Module wrapper for the projector.
    
    This allows using the projector in nn.Sequential and other PyTorch constructs.
    Since grid_sample is differentiable, gradients flow through automatically.
    """
    
    def __init__(self, geometry, device=None, batch_size=8):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.projector = PyTorchProjectorOptimized(geometry, device)
        self.batch_size = batch_size
        
    def forward(self, volume):
        return self.projector.forward(volume, batch_size=self.batch_size)
