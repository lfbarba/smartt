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
    
    def _precompute_rotations(self) -> list:
        """
        Pre-compute Wigner D-matrices for all projection angles.
        
        Returns
        -------
        rotation_matrices : list
            List of e3nn Wigner D-matrices for each projection
        """
        rotation_matrices = []
        
        for proj_vec in self.projection_vectors:
            # Get rotation matrix
            R = self._rotation_matrix_to_align_vector(proj_vec)
            
            # Convert to e3nn Wigner D-matrix representation
            # Note: e3nn uses a different convention, so we may need to transpose
            D_matrix = o3.matrix_to_angles(R.cpu().numpy())
            
            # Create Wigner D-matrix for this rotation
            wigner_D = o3.wigner_D(self.irreps, *D_matrix)
            
            rotation_matrices.append(wigner_D.to(self.device))
        
        return rotation_matrices
    
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
        Project spherical harmonic volume to circular slices.
        
        Parameters
        ----------
        coeffs : torch.Tensor
            Spherical harmonic coefficients with shape:
            - (X, Y, Z, num_coeffs) for a volume of voxels, or
            - (num_coeffs,) for a single point
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
              - (num_projections, num_surviving_coeffs) for single point (~25 coeffs)
              - (num_projections, X, Y, Z, num_surviving_coeffs) for volume
            - If output_type='samples':
              - (num_projections, phi_samples) for single point
              - (num_projections, X, Y, Z, phi_samples) for volume
        """
        if output_type not in ['coefficients', 'samples']:
            raise ValueError(f"output_type must be 'coefficients' or 'samples', got '{output_type}'")
        
        # Ensure coeffs is on the correct device
        coeffs = coeffs.to(self.device)
        
        # Determine input shape
        input_shape = coeffs.shape[:-1]  # Everything except num_coeffs
        is_volume = len(input_shape) > 0
        
        if is_volume:
            # Reshape to (batch, num_coeffs) for easier processing
            batch_size = int(np.prod(input_shape))
            coeffs_flat = coeffs.reshape(batch_size, -1)
        else:
            coeffs_flat = coeffs.unsqueeze(0)
            batch_size = 1
        
        if output_type == 'coefficients':
            # Return compact harmonic coefficients representation
            return self._project_to_harmonic_coefficients(coeffs, coeffs_flat, is_volume, input_shape)
        else:
            # Return evaluated samples at phi angles
            return self._project_to_phi_samples(coeffs_flat, is_volume, input_shape, phi_samples)
    
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
    
    def _project_to_phi_samples(
        self,
        coeffs_flat: torch.Tensor,
        is_volume: bool,
        input_shape: tuple,
        phi_samples: int
    ) -> torch.Tensor:
        """
        Project to evaluated samples at phi angles (for visualization).
        
        Returns function values at phi_samples angles around each slice.
        """
        # Create phi samples
        phi = torch.linspace(0, 2*np.pi, phi_samples, device=self.device)
        
        # Evaluate at equator for all projections
        slice_values_list = []
        
        for proj_idx in range(self.num_projections):
            # Rotate coefficients (if enabled)
            rotated_coeffs = self._rotate_sh_coefficients(coeffs_flat, proj_idx)
            
            # Evaluate at equator (θ=π/2) for all φ
            slice_values = self._evaluate_equator_slice(rotated_coeffs, phi)
            
            slice_values_list.append(slice_values)
        
        # Stack all projections
        slices = torch.stack(slice_values_list, dim=0)  # (num_proj, batch, phi_samples)
        
        # Reshape back to original spatial dimensions
        if is_volume:
            slices = slices.reshape(self.num_projections, *input_shape, phi_samples)
        else:
            slices = slices.squeeze(1)  # Remove batch dimension for single point
        
        return slices
    
    def _evaluate_equator_slice(
        self,
        coeffs: torch.Tensor,
        phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate spherical function on equator (θ=π/2).
        
        Parameters
        ----------
        coeffs : torch.Tensor
            Shape (batch, num_coeffs) spherical harmonic coefficients
        phi : torch.Tensor
            Shape (phi_samples,) azimuthal angles
        
        Returns
        -------
        values : torch.Tensor
            Shape (batch, phi_samples) function values on the equator
        """
        batch_size = coeffs.shape[0]
        phi_samples = phi.shape[0]
        
        # Initialize output
        values = torch.zeros(
            batch_size, phi_samples,
            device=self.device,
            dtype=coeffs.dtype
        )
        
        # Convert Legendre values to tensor
        legendre_tensor = torch.tensor(
            self.legendre_at_zero,
            device=self.device,
            dtype=coeffs.dtype
        )
        
        # Compute for each coefficient
        for i, (ell, m) in enumerate(zip(self.ell_indices, self.m_indices)):
            abs_m = abs(m)
            
            # Skip terms where P_ℓ^m(0) = 0 (ℓ+m odd)
            if (ell + abs_m) % 2 != 0:
                continue
            
            # Get coefficient values
            c = coeffs[:, i]  # (batch,)
            
            # Get P_ℓ^m(0) value
            P_lm_0 = legendre_tensor[i]
            
            # Compute angular part: e^(imφ) = cos(mφ) + i·sin(mφ)
            # For real coefficients, we use real/imaginary parts
            if m >= 0:
                # Positive m: use cos(mφ)
                angular = torch.cos(m * phi)  # (phi_samples,)
            else:
                # Negative m: use sin(|m|φ)
                angular = torch.sin(abs_m * phi)
            
            # The Legendre values already include spherical harmonic normalization
            # No additional normalization needed here
            
            # Add contribution: c * P_ℓ^m(0) * angular_part
            # Broadcasting: (batch, 1) * (phi_samples,) -> (batch, phi_samples)
            values += (c.unsqueeze(1) * P_lm_0 * angular)
        
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
