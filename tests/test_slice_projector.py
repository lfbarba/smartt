"""
Test script for SphericalHarmonicSliceProjector.

This tests the basic functionality of projecting SH coefficients to slices.
"""

import sys
sys.path.insert(0, '/myhome/smartt')

import torch
import numpy as np
from smartt.projectors import SphericalHarmonicSliceProjector
from mumott.data_handling import DataContainer

def test_slice_projector():
    """Test basic slice projector functionality."""
    
    print("=" * 80)
    print("Testing SphericalHarmonicSliceProjector")
    print("=" * 80)
    
    # Load geometry
    print("\n1. Loading geometry...")
    dc = DataContainer('/myhome/data/smartt/shared/frogbone/dataset_qbin_0009.h5')
    geometry = dc.geometry
    print(f"   Geometry loaded: {len(geometry)} projections")
    
    # Create projector
    print("\n2. Creating slice projector...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    projector = SphericalHarmonicSliceProjector(
        ell_max=8,
        geometry=geometry,
        device=device,
        use_rotation=False  # Start with rotation disabled
    )
    
    print(f"   ✓ Projector created")
    print(f"     - ell_max: {projector.ell_max}")
    print(f"     - num_coeffs: {projector.num_coeffs}")
    print(f"     - num_projections: {projector.num_projections}")
    print(f"     - projection_vectors shape: {projector.projection_vectors.shape}")
    
    # Test with single point
    print("\n3. Testing with single point...")
    test_coeffs = torch.randn(45, device=device)
    
    slices = projector.project_to_slices(test_coeffs, phi_samples=180)
    
    print(f"   ✓ Single point test passed")
    print(f"     - Input shape: {test_coeffs.shape}")
    print(f"     - Output shape: {slices.shape}")
    print(f"     - Expected: ({projector.num_projections}, 180)")
    print(f"     - Output stats: min={slices.min():.3f}, max={slices.max():.3f}, mean={slices.mean():.3f}")
    
    # Test with volume
    print("\n4. Testing with small volume...")
    test_volume = torch.randn(5, 5, 5, 45, device=device)
    
    slices_vol = projector.project_to_slices(test_volume, phi_samples=90)
    
    print(f"   ✓ Volume test passed")
    print(f"     - Input shape: {test_volume.shape}")
    print(f"     - Output shape: {slices_vol.shape}")
    print(f"     - Expected: ({projector.num_projections}, 5, 5, 5, 90)")
    print(f"     - Output stats: min={slices_vol.min():.3f}, max={slices_vol.max():.3f}, mean={slices_vol.mean():.3f}")
    
    # Test forward method
    print("\n5. Testing forward() method...")
    slices_fwd = projector.forward(test_coeffs, phi_samples=180)
    
    assert torch.allclose(slices, slices_fwd), "forward() should match project_to_slices()"
    print(f"   ✓ forward() method works correctly")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    
    return projector


if __name__ == "__main__":
    projector = test_slice_projector()
