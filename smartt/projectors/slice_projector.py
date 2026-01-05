"""
Slice projector for spherical harmonic coefficients.

This module implements projection of spherical harmonic coefficients onto
orthogonal slices in reciprocal space, corresponding to different projection
angles in tomographic reconstruction.
"""

import numpy as np
import torch
from typing import Optional, Tuple
from scipy.special import lpmv
from e3nn import o3


def _compute_rotation_worker(proj_vec: torch.Tensor, ell_max: int) -> torch.Tensor:
    """
    Worker function for parallel rotation matrix computation.
    Must be at module level for pickling.
    
    Parameters
    ----------
    proj_vec : torch.Tensor
        Projection direction vector (3,)
    ell_max : int
        Maximum spherical harmonic degree
    
    Returns
    -------
    wigner_D : torch.Tensor
        Block-diagonal Wigner D-matrix
    """
    # Compute rotation matrix to align proj_vec with z-axis
    vec = proj_vec / torch.norm(proj_vec)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=vec.dtype)
    
    # Dot product with z-axis
    cos_theta = torch.dot(vec, ez)
    eps = 1e-7
    
    # vec already aligned with +z
    if cos_theta >= 1.0 - eps:
        R = torch.eye(3, dtype=vec.dtype)
    # vec aligned with -z (180° rotation)
    elif cos_theta <= -1.0 + eps:
        R = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=vec.dtype)
    else:
        # General case: use Rodrigues formula
        axis = torch.cross(vec, ez)
        axis = axis / torch.norm(axis)
        
        sin_theta = torch.sqrt(1 - cos_theta**2)
        
        # Skew-symmetric matrix K
        kx, ky, kz = axis[0], axis[1], axis[2]
        K = torch.stack([
            torch.stack([torch.tensor(0.0, dtype=vec.dtype), -kz, ky]),
            torch.stack([kz, torch.tensor(0.0, dtype=vec.dtype), -kx]),
            torch.stack([-ky, kx, torch.tensor(0.0, dtype=vec.dtype)])
        ])
        
        I = torch.eye(3, dtype=vec.dtype)
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        R = I + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)
    
    # Convert to Euler angles
    alpha, beta, gamma = o3.matrix_to_angles(R)
    
    # Build block-diagonal Wigner D-matrix for even-ℓ only
    D_blocks = []
    for ell in range(0, ell_max + 1, 2):  # Even ℓ only
        D_ell = o3.wigner_D(ell, alpha, beta, gamma)
        D_blocks.append(D_ell)
    
    # Combine into block-diagonal matrix
    wigner_D = torch.block_diag(*D_blocks)
    
    return wigner_D


def _worker_process_batch(start_idx: int, end_idx: int, proj_vecs: torch.Tensor, 
                          ell_max: int, result_queue) -> None:
    """
    Worker process that computes a batch of rotation matrices.
    Must be at module level for pickling.
    
    Parameters
    ----------
    start_idx : int
        Start index in projection vectors
    end_idx : int
        End index (exclusive)
    proj_vecs : torch.Tensor
        All projection vectors
    ell_max : int
        Maximum spherical harmonic degree
    result_queue : multiprocessing.Queue
        Queue to put results
    """
    results = []
    for idx in range(start_idx, end_idx):
        proj_vec = proj_vecs[idx]
        # Compute rotation matrix
        wigner_D = _compute_rotation_worker(proj_vec, ell_max)
        results.append((idx, wigner_D))
    result_queue.put(results)
    """
    Worker function for parallel rotation matrix computation.
    Must be at module level for pickling.
    
    Parameters
    ----------
    proj_vec : torch.Tensor
        Projection direction vector (3,)
    ell_max : int
        Maximum spherical harmonic degree
    
    Returns
    -------
    wigner_D : torch.Tensor
        Block-diagonal Wigner D-matrix
    """
    # Compute rotation matrix to align proj_vec with z-axis
    vec = proj_vec / torch.norm(proj_vec)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=vec.dtype)
    
    # Dot product with z-axis
    cos_theta = torch.dot(vec, ez)
    eps = 1e-7
    
    # vec already aligned with +z
    if cos_theta >= 1.0 - eps:
        R = torch.eye(3, dtype=vec.dtype)
    # vec aligned with -z (180° rotation)
    elif cos_theta <= -1.0 + eps:
        R = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=vec.dtype)
    else:
        # General case: use Rodrigues formula
        axis = torch.cross(vec, ez)
        axis = axis / torch.norm(axis)
        
        sin_theta = torch.sqrt(1 - cos_theta**2)
        
        # Skew-symmetric matrix K
        kx, ky, kz = axis[0], axis[1], axis[2]
        K = torch.stack([
            torch.stack([torch.tensor(0.0, dtype=vec.dtype), -kz, ky]),
            torch.stack([kz, torch.tensor(0.0, dtype=vec.dtype), -kx]),
            torch.stack([-ky, kx, torch.tensor(0.0, dtype=vec.dtype)])
        ])
        
        I = torch.eye(3, dtype=vec.dtype)
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        R = I + sin_theta * K + (1 - cos_theta) * torch.matmul(K, K)
    
    # Convert to Euler angles
    alpha, beta, gamma = o3.matrix_to_angles(R)
    
    # Build block-diagonal Wigner D-matrix for even-ℓ only
    D_blocks = []
    for ell in range(0, ell_max + 1, 2):  # Even ℓ only
        D_ell = o3.wigner_D(ell, alpha, beta, gamma)
        D_blocks.append(D_ell)
    
    # Combine into block-diagonal matrix
    wigner_D = torch.block_diag(*D_blocks)
    
    return wigner_D


class SphericalHarmonicSliceProjector:
    """
    Project spherical harmonic coefficients onto circular slices.
    
    For each projection angle, computes an orthogonal slice through the
    spherical function by:
    1. Rotating coefficients so the slice becomes the equator (θ=π/2)
    2. Evaluating only terms where ℓ+m is even (others vanish at equator)
    3. Returning a Fourier series (circular harmonics) for each slice
    
    Parameters
    ----------
    ell_max : int
        Maximum spherical harmonic degree (must be even)
    geometry : mumott.Geometry
        Geometry object containing projection angles
    device : torch.device, optional
        Device for torch tensors
    use_rotation : bool, optional
        If True, properly rotate SH coefficients using Wigner D-matrices.
        If False, skip rotation (faster but less accurate). Default: False.
    """
    
    def __init__(
        self, 
        ell_max: int, 
        geometry, 
        device: Optional[torch.device] = None,
        use_rotation: bool = False
    ):
        if ell_max % 2 != 0:
            raise ValueError(f"ell_max must be even, got {ell_max}")
        
        self.ell_max = ell_max
        self.geometry = geometry
        self.device = device if device is not None else torch.device('cpu')
        self.use_rotation = use_rotation
        
        # Build index arrays for even-ℓ only spherical harmonics
        self.ell_indices, self.m_indices = self._build_even_ell_indices()
        self.num_coeffs = len(self.ell_indices)
        
        # Build irreps string for e3nn (even-ℓ only)
        self.irreps = self._build_irreps()
        
        # Compute projection vectors from geometry
        self.projection_vectors = self._compute_projection_vectors()
        self.num_projections = len(self.projection_vectors)
        
        # Pre-compute P_ℓ^m(0) for the equator restriction
        self.legendre_at_zero = self._compute_legendre_at_zero()
        
        # Pre-compute rotation matrices if using rotation
        if self.use_rotation:
            self.rotation_matrices = self._precompute_rotations()
        
    def _build_irreps(self) -> o3.Irreps:
        """Build e3nn Irreps string for even-ℓ only."""
        irrep_list = []
        for ell in range(0, self.ell_max + 1, 2):
            parity = 'e' if ell % 2 == 0 else 'o'
            irrep_list.append(f"1x{ell}{parity}")
        return o3.Irreps(" + ".join(irrep_list))
    
    def _build_even_ell_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build (ℓ, m) index arrays for even-ℓ only basis."""
        ell_list = []
        m_list = []
        
        for ell in range(0, self.ell_max + 1, 2):  # Only even ℓ
            for m in range(-ell, ell + 1):
                ell_list.append(ell)
                m_list.append(m)
        
        return np.array(ell_list), np.array(m_list)
    
    def _compute_projection_vectors(self) -> torch.Tensor:
        """
        Compute unit projection vectors from geometry angles.
        
        Returns unit vectors pointing in the projection direction for each
        angle. These define the normals to the slicing planes.
        
        Returns
        -------
        vectors : torch.Tensor
            Shape (num_projections, 3) unit vectors
        """
        # Extract angles from geometry
        rotation_angles_rad = self.geometry.inner_angles[:]
        tilt_angles_rad = np.pi/2 - np.array(self.geometry.outer_angles)[:]
        
        # Convert to Cartesian coordinates on unit sphere
        x = np.sin(tilt_angles_rad) * np.cos(rotation_angles_rad)
        y = np.sin(tilt_angles_rad) * np.sin(rotation_angles_rad)
        z = np.cos(tilt_angles_rad)
        
        vectors = torch.tensor(
            np.stack([x, y, z], axis=-1),
            dtype=torch.float32,
            device=self.device
        )
        
        # Normalize to ensure unit vectors
        vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
        
        return vectors
    
    def _compute_legendre_at_zero(self) -> np.ndarray:
        """
        Pre-compute associated Legendre functions P_ℓ^m(0) with normalization.
        
        These values determine which spherical harmonic terms survive
        when restricted to the equator (θ=π/2).
        
        Returns the normalized values as they appear in spherical harmonics:
        Y_ℓ^m(θ, φ) = N_ℓ^m * P_ℓ^|m|(cos θ) * exp(i m φ)
        where N_ℓ^m = sqrt((2ℓ+1)/(4π) * (ℓ-|m|)!/(ℓ+|m|)!)
        
        Returns
        -------
        legendre_values : np.ndarray
            Shape (num_coeffs,) normalized values of P_ℓ^|m|(0)
        """
        from scipy.special import factorial
        
        legendre_values = np.zeros(self.num_coeffs)
        
        for i, (ell, m) in enumerate(zip(self.ell_indices, self.m_indices)):
            abs_m = abs(m)
            # P_ℓ^m(0) = 0 if ℓ+m is odd
            if (ell + abs_m) % 2 == 0:
                # Compute unnormalized P_ℓ^m(0)
                P_lm = lpmv(abs_m, ell, 0.0)
                
                # Apply spherical harmonic normalization
                # N_ℓ^m = sqrt((2ℓ+1)/(4π) * (ℓ-|m|)!/(ℓ+|m|)!)
                normalization = np.sqrt(
                    (2 * ell + 1) / (4 * np.pi) * 
                    factorial(ell - abs_m) / factorial(ell + abs_m)
                )
                
                legendre_values[i] = P_lm * normalization
            else:
                legendre_values[i] = 0.0
        
        return legendre_values
    
    def _rotation_matrix_to_align_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation matrix that maps vec to +z axis.
        
        Uses Rodrigues' rotation formula. If vec is already along ±z,
        handles degenerate cases appropriately.
        
        Parameters
        ----------
        vec : torch.Tensor
            Shape (3,) unit vector to align with z-axis
        
        Returns
        -------
        R : torch.Tensor
            Shape (3, 3) rotation matrix
        """
        device = vec.device
        dtype = vec.dtype
        
        ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        
        # Normalize input
        vec = vec / torch.norm(vec)
        
        # Dot product with z-axis
        cos_theta = torch.dot(vec, ez)
        
        eps = 1e-7
        
        # vec already aligned with +z
        if cos_theta >= 1.0 - eps:
            return torch.eye(3, device=device, dtype=dtype)
        
        # vec aligned with -z (180° rotation)
        if cos_theta <= -1.0 + eps:
            return torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ], device=device, dtype=dtype)
        
        # General case: use Rodrigues formula
        axis = torch.cross(vec, ez)
        axis = axis / torch.norm(axis)
        
        sin_theta = torch.sqrt(1 - cos_theta**2)
        
        # Skew-symmetric matrix K
        kx, ky, kz = axis[0], axis[1], axis[2]
        K = torch.stack([
            torch.stack([torch.tensor(0.0, device=device, dtype=dtype), -kz, ky]),
            torch.stack([kz, torch.tensor(0.0, device=device, dtype=dtype), -kx]),
            torch.stack([-ky, kx, torch.tensor(0.0, device=device, dtype=dtype)])
        ])
        
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + sin_theta * K + (1.0 - cos_theta) * (K @ K)
        
        return R
    
    def _precompute_rotations(self, use_parallel=True, num_workers=None) -> list:
        """
        Pre-compute Wigner D-matrices for all projection angles.
        
        Parameters
        ----------
        use_parallel : bool
            If True, use multiprocessing to compute rotations in parallel
        num_workers : int, optional
            Number of worker processes. If None, uses CPU count.
        
        Returns
        -------
        rotation_matrices : list
            List of rotation matrices that can rotate SH coefficients (45x45 matrices)
        """
        if use_parallel and self.num_projections > 1:
            return self._precompute_rotations_parallel(num_workers)
        else:
            return self._precompute_rotations_serial()
    
    def _precompute_rotations_serial(self) -> list:
        """Compute rotation matrices serially (single-threaded)."""
        rotation_matrices = []
        
        for proj_vec in self.projection_vectors:
            wigner_D = self._compute_single_rotation(proj_vec)
            rotation_matrices.append(wigner_D.to(self.device))
        
        return rotation_matrices
    
    def _precompute_rotations_parallel(self, num_workers=None) -> list:
        """Compute rotation matrices in parallel using multiprocessing."""
        import torch.multiprocessing as mp
        import os
        
        if num_workers is None:
            num_workers = min(os.cpu_count(), self.num_projections)
        
        print(f"Computing {self.num_projections} rotation matrices using {num_workers} workers...")
        
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        mp_context = mp.get_context('spawn')
        
        # Create a queue for results
        result_queue = mp_context.Queue()
        
        # Divide work into chunks
        chunk_size = max(1, self.num_projections // num_workers)
        processes = []
        proj_vecs_cpu = self.projection_vectors.cpu()
        
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, self.num_projections) if i < num_workers - 1 else self.num_projections
            
            if start_idx >= self.num_projections:
                break
            
            p = mp_context.Process(
                target=_worker_process_batch,
                args=(start_idx, end_idx, proj_vecs_cpu, self.ell_max, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        rotation_matrices = [None] * self.num_projections
        for _ in range(len(processes)):
            results = result_queue.get()
            for idx, wigner_D in results:
                rotation_matrices[idx] = wigner_D.to(self.device)
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        print(f"✓ Computed {len(rotation_matrices)} rotation matrices")
        return rotation_matrices
    
    def _compute_single_rotation(self, proj_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute a single Wigner D-matrix for a projection vector.
        
        Parameters
        ----------
        proj_vec : torch.Tensor
            Projection direction vector (3,)
        
        Returns
        -------
        wigner_D : torch.Tensor
            Block-diagonal Wigner D-matrix (45x45 for ell_max=8)
        """
        # Get rotation matrix (3x3)
        R = self._rotation_matrix_to_align_vector(proj_vec)
        
        # Convert to Euler angles
        R_cpu = R.cpu() if R.is_cuda else R
        alpha, beta, gamma = o3.matrix_to_angles(R_cpu)
        
        # Build block-diagonal Wigner D-matrix for even-ℓ only
        D_blocks = []
        for ell in range(0, self.ell_max + 1, 2):  # Even ℓ only
            # Wigner D-matrix for this ℓ
            D_ell = o3.wigner_D(ell, alpha, beta, gamma)
            D_blocks.append(D_ell)
        
        # Combine into block-diagonal matrix
        wigner_D = torch.block_diag(*D_blocks)
        
        return wigner_D
    
    def _rotate_sh_coefficients(
        self, 
        coeffs: torch.Tensor, 
        proj_idx: int
    ) -> torch.Tensor:
        """
        Rotate spherical harmonic coefficients using Wigner D-matrices.
        
        Parameters
        ----------
        coeffs : torch.Tensor
            Shape (..., num_coeffs) spherical harmonic coefficients
        proj_idx : int
            Index of projection (to use pre-computed rotation)
        
        Returns
        -------
        rotated_coeffs : torch.Tensor
            Shape (..., num_coeffs) rotated coefficients
        """
        if not self.use_rotation:
            # Skip rotation for speed
            return coeffs
        
        # Get pre-computed Wigner D-matrix
        D = self.rotation_matrices[proj_idx]
        
        # Apply rotation: rotated = D @ coeffs
        # Broadcasting: (num_coeffs, num_coeffs) @ (..., num_coeffs) -> (..., num_coeffs)
        input_shape = coeffs.shape
        coeffs_flat = coeffs.reshape(-1, self.num_coeffs)  # (batch, num_coeffs)
        
        rotated_flat = torch.matmul(coeffs_flat, D.T)  # (batch, num_coeffs)
        
        rotated_coeffs = rotated_flat.reshape(input_shape)
        
        return rotated_coeffs
    
    def project_to_slices(
        self,
        coeffs: torch.Tensor,
        output_type: str = 'coefficients',
        phi_samples: Optional[int] = 360
    ) -> torch.Tensor:
        """
        Project spherical harmonic coefficients to circular slices.
        
        For SAXS tensor tomography: input is organized by projection, where
        coeffs[i] corresponds to projection_vectors[i].
        
        Parameters
        ----------
        coeffs : torch.Tensor
            Spherical harmonic coefficients with shape:
            - (num_projections, X, Y, num_coeffs) for projections of a volume
            - (num_projections, num_coeffs) for projections of a single point
            - (X, Y, num_coeffs) for a single projection slice (will use first projection vector)
            - (num_coeffs,) for a single point (will use first projection vector)
        output_type : str
            Type of output to return:
            - 'coefficients': Returns ~25 Fourier/harmonic coefficients per slice (compact)
            - 'samples': Returns evaluated function at phi_samples angles (for visualization)
        phi_samples : int
            Number of azimuthal samples (only used if output_type='samples')
        
        Returns
        -------
        slices : torch.Tensor
            Projected slices with shape:
            - If output_type='coefficients':
              - (num_projections, X, Y, num_surviving_coeffs) for volume projections
              - (num_projections, num_surviving_coeffs) for point projections
            - If output_type='samples':
              - (num_projections, X, Y, phi_samples) for volume projections
              - (num_projections, phi_samples) for point projections
        """
        if output_type not in ['coefficients', 'samples']:
            raise ValueError(f"output_type must be 'coefficients' or 'samples', got '{output_type}'")
        
        # Ensure coeffs is on the correct device
        coeffs = coeffs.to(self.device)
        
        # Determine input format
        if coeffs.shape[0] == self.num_projections:
            # Input is (num_projections, ..., num_coeffs)
            # This is the standard SAXS-TT format
            has_projection_dim = True
            spatial_shape = coeffs.shape[1:-1]  # Everything between num_proj and num_coeffs
        else:
            # Input is (..., num_coeffs) - single projection
            # Will use first projection vector
            has_projection_dim = False
            spatial_shape = coeffs.shape[:-1]
            # Add projection dimension
            coeffs = coeffs.unsqueeze(0)  # (1, ..., num_coeffs)
        
        # Flatten spatial dimensions: (num_proj, ..., num_coeffs) -> (num_proj, batch, num_coeffs)
        num_proj = coeffs.shape[0]
        if len(spatial_shape) > 0:
            batch_size = int(np.prod(spatial_shape))
            coeffs_flat = coeffs.reshape(num_proj, batch_size, -1)
            is_volume = True
        else:
            # Single point per projection
            coeffs_flat = coeffs.reshape(num_proj, 1, -1)
            is_volume = False
            spatial_shape = ()
        
        if output_type == 'coefficients':
            # Return compact harmonic coefficients representation
            return self._project_to_harmonic_coefficients_matched(
                coeffs_flat, is_volume, spatial_shape, has_projection_dim
            )
        else:
            # Return evaluated samples at phi angles
            return self._project_to_phi_samples_matched(
                coeffs_flat, is_volume, spatial_shape, phi_samples, has_projection_dim
            )
    
    def _project_to_harmonic_coefficients(
        self,
        coeffs: torch.Tensor,
        coeffs_flat: torch.Tensor,
        is_volume: bool,
        input_shape: tuple
    ) -> torch.Tensor:
        """
        Project to compact harmonic coefficient representation.
        
        Returns ~25 Fourier coefficients per slice (only surviving even ℓ+m terms).
        """
        # Mask for surviving terms (even ℓ+m)
        ell_plus_m = self.ell_indices + self.m_indices
        surviving_mask = (ell_plus_m % 2 == 0)
        num_surviving = int(surviving_mask.sum())
        
        # Convert to tensor
        surviving_mask_tensor = torch.from_numpy(surviving_mask).to(self.device)
        legendre_tensor = torch.tensor(
            self.legendre_at_zero,
            device=self.device,
            dtype=coeffs.dtype
        )
        
        # Apply P_ℓ^m(0) restriction: c_ℓm → c_ℓm * P_ℓ^m(0)
        # This zeros out odd ℓ+m terms
        restricted_coeffs = coeffs_flat * legendre_tensor.unsqueeze(0)  # (batch, num_coeffs)
        
        if self.use_rotation:
            # Apply rotation for each projection
            result_list = []
            for proj_idx in range(self.num_projections):
                rotated_coeffs = self._rotate_sh_coefficients(restricted_coeffs, proj_idx)
                # Extract only surviving coefficients
                surviving_coeffs = rotated_coeffs[:, surviving_mask_tensor]
                result_list.append(surviving_coeffs)
            
            # Stack: (num_projections, batch, num_surviving)
            result = torch.stack(result_list, dim=0)
        else:
            # Without rotation: all projections have same coefficients
            # Extract only surviving coefficients
            surviving_coeffs = restricted_coeffs[:, surviving_mask_tensor]  # (batch, num_surviving)
            
            # Repeat for all projections
            result = surviving_coeffs.unsqueeze(0).expand(self.num_projections, -1, -1)
        
        # Reshape back to volume if needed
        if is_volume:
            result = result.reshape(self.num_projections, *input_shape, num_surviving)
        else:
            result = result.squeeze(1)  # (num_projections, num_surviving)
        
        return result
    
    def _project_to_harmonic_coefficients_matched(
        self,
        coeffs_flat: torch.Tensor,
        is_volume: bool,
        spatial_shape: tuple,
        has_projection_dim: bool
    ) -> torch.Tensor:
        """
        Project to compact harmonic coefficients with matched projections.
        
        Each projection's coefficients are evaluated in its corresponding direction.
        
        Parameters
        ----------
        coeffs_flat : torch.Tensor
            Shape (num_proj, batch, num_coeffs)
        """
        # Mask for surviving terms (even ℓ+m)
        ell_plus_m = self.ell_indices + self.m_indices
        surviving_mask = (ell_plus_m % 2 == 0)
        num_surviving = int(surviving_mask.sum())
        
        # Convert to tensor
        surviving_mask_tensor = torch.from_numpy(surviving_mask).to(self.device)
        legendre_tensor = torch.tensor(
            self.legendre_at_zero,
            device=self.device,
            dtype=coeffs_flat.dtype
        )
        
        # Apply P_ℓ^m(0) restriction: c_ℓm → c_ℓm * P_ℓ^m(0)
        # Broadcasting: (num_proj, batch, num_coeffs) * (num_coeffs,)
        restricted_coeffs = coeffs_flat * legendre_tensor.unsqueeze(0).unsqueeze(0)
        
        if self.use_rotation:
            # Apply rotation for each projection to its corresponding direction
            result_list = []
            for proj_idx in range(coeffs_flat.shape[0]):
                rotated_coeffs = self._rotate_sh_coefficients(
                    restricted_coeffs[proj_idx], proj_idx
                )
                # Extract only surviving coefficients
                surviving_coeffs = rotated_coeffs[:, surviving_mask_tensor]
                result_list.append(surviving_coeffs)
            
            # Stack: (num_proj, batch, num_surviving)
            result = torch.stack(result_list, dim=0)
        else:
            # Without rotation: just extract surviving coefficients
            # (num_proj, batch, num_coeffs) -> (num_proj, batch, num_surviving)
            result = restricted_coeffs[:, :, surviving_mask_tensor]
        
        # Reshape back to spatial dimensions
        if is_volume:
            result = result.reshape(coeffs_flat.shape[0], *spatial_shape, num_surviving)
        else:
            result = result.squeeze(1)  # (num_proj, num_surviving)
        
        # Remove projection dim if input didn't have it
        if not has_projection_dim:
            result = result.squeeze(0)
        
        return result
    
    def _project_to_phi_samples_matched(
        self,
        coeffs_flat: torch.Tensor,
        is_volume: bool,
        spatial_shape: tuple,
        phi_samples: int,
        has_projection_dim: bool
    ) -> torch.Tensor:
        """
        Project to phi samples with matched projections.
        
        Each projection's coefficients are evaluated in its corresponding direction.
        
        Parameters
        ----------
        coeffs_flat : torch.Tensor
            Shape (num_proj, batch, num_coeffs)
        """
        # Step 1: Get compact harmonic coefficients for each projection
        harmonic_coeffs = self._project_to_harmonic_coefficients_matched(
            coeffs_flat, is_volume=True, spatial_shape=(coeffs_flat.shape[1],), 
            has_projection_dim=True
        )
        # harmonic_coeffs shape: (num_proj, batch, num_surviving)
        
        # Step 2: Evaluate at phi angles
        phi = torch.linspace(0, 2*np.pi, phi_samples, device=self.device)
        
        # Use the optimized evaluation
        slices = self._evaluate_harmonic_coefficients(harmonic_coeffs, phi)
        # slices shape: (num_proj, batch, phi_samples)
        
        # Reshape back to spatial dimensions
        if is_volume:
            slices = slices.reshape(coeffs_flat.shape[0], *spatial_shape, phi_samples)
        else:
            slices = slices.squeeze(1)  # (num_proj, phi_samples)
        
        # Remove projection dim if input didn't have it
        if not has_projection_dim:
            slices = slices.squeeze(0)
        
        return slices
    
    def _project_to_phi_samples(
        self,
        coeffs_flat: torch.Tensor,
        is_volume: bool,
        input_shape: tuple,
        phi_samples: int
    ) -> torch.Tensor:
        """
        Project to evaluated samples at phi angles (for visualization).
        
        GPU-optimized: First computes harmonic coefficients, then samples from them.
        
        Returns function values at phi_samples angles around each slice.
        """
        # Step 1: Get compact harmonic coefficients
        # We need to pass the original coeffs too for compatibility
        if is_volume:
            coeffs_original = coeffs_flat.reshape(*input_shape, -1)
        else:
            coeffs_original = coeffs_flat.squeeze(0)
        
        harmonic_coeffs = self._project_to_harmonic_coefficients(
            coeffs_original, coeffs_flat, is_volume, input_shape
        )
        
        # harmonic_coeffs shape: (num_projections, batch, num_surviving) or (num_projections, num_surviving)
        
        # Step 2: Evaluate harmonic coefficients at phi sample points
        phi = torch.linspace(0, np.pi, phi_samples, device=self.device)
        
        # Reshape to (num_projections, batch, num_surviving) if needed
        if not is_volume:
            harmonic_coeffs = harmonic_coeffs.unsqueeze(1)  # (num_proj, 1, num_surviving)
        else:
            # Flatten spatial dimensions: (num_proj, X, Y, Z, num_surviving) -> (num_proj, batch, num_surviving)
            batch_size = int(np.prod(input_shape))
            harmonic_coeffs = harmonic_coeffs.reshape(self.num_projections, batch_size, -1)
        
        # Step 3: Evaluate the surviving harmonic coefficients at phi angles
        slices = self._evaluate_harmonic_coefficients(harmonic_coeffs, phi)
        
        # Reshape back to original spatial dimensions
        if is_volume:
            slices = slices.reshape(self.num_projections, *input_shape, phi_samples)
        else:
            slices = slices.squeeze(1)  # Remove batch dimension for single point
        
        return slices
    
    def _evaluate_harmonic_coefficients(
        self,
        harmonic_coeffs: torch.Tensor,
        phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate compact harmonic coefficients at phi sample points.
        
        This takes the output of _project_to_harmonic_coefficients (the ~25 surviving
        Fourier coefficients) and evaluates them at specific φ angles.
        
        GPU-optimized vectorized implementation.
        
        Parameters
        ----------
        harmonic_coeffs : torch.Tensor
            Shape (num_projections, batch, num_surviving) compact harmonic coefficients
        phi : torch.Tensor
            Shape (phi_samples,) azimuthal angles
        
        Returns
        -------
        values : torch.Tensor
            Shape (num_projections, batch, phi_samples) function values
        """
        # Initialize cached values if needed
        if not hasattr(self, '_surviving_m'):
            ell_plus_m = self.ell_indices + self.m_indices
            self._surviving_m = torch.tensor(
                self.m_indices[ell_plus_m % 2 == 0],
                device=self.device,
                dtype=torch.long
            )
        
        # Compute angular basis functions
        # Shape: (num_surviving, phi_samples)
        m_values = self._surviving_m.unsqueeze(1)  # (num_surviving, 1)
        phi_grid = phi.unsqueeze(0)  # (1, phi_samples)
        
        # Angular part: cos(mφ) for m≥0, sin(|m|φ) for m<0
        # Match dtype to harmonic_coeffs
        angular_basis = torch.where(
            m_values >= 0,
            torch.cos(m_values.float() * phi_grid),
            torch.sin(torch.abs(m_values.float()) * phi_grid)
        ).to(harmonic_coeffs.dtype)  # (num_surviving, phi_samples)
        
        # Evaluate: (num_proj, batch, num_surviving) @ (num_surviving, phi_samples)
        # -> (num_proj, batch, phi_samples)
        values = torch.matmul(harmonic_coeffs, angular_basis)
        
        return values
    
    def forward(
        self,
        coeffs: torch.Tensor,
        output_type: str = 'coefficients',
        phi_samples: Optional[int] = 360
    ) -> torch.Tensor:
        """
        Forward projection (same as project_to_slices).
        
        For compatibility with PyTorch projector interface.
        
        Parameters
        ----------
        coeffs : torch.Tensor
            Spherical harmonic coefficients
        output_type : str
            'coefficients' for compact representation (~25 coeffs per slice) or
            'samples' for evaluated function at phi angles
        phi_samples : int
            Number of azimuthal samples (only used if output_type='samples')
        """
        return self.project_to_slices(coeffs, output_type=output_type, phi_samples=phi_samples)
