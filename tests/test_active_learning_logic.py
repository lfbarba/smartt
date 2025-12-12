"""
Test script to verify active learning logic without running full pipeline.
"""

import numpy as np


def get_num_coeffs(ell_max: int) -> int:
    """Compute number of spherical harmonic coefficients for even ℓ only."""
    num_coeffs = 0
    for ell in range(0, ell_max + 1, 2):
        num_coeffs += 2 * ell + 1
    return num_coeffs


def test_coefficient_calculation():
    """Test coefficient calculation for even ℓ only."""
    print("Testing coefficient calculation...")
    
    test_cases = {
        0: 1,    # ℓ=0: 1 coeff
        2: 6,    # ℓ=0,2: 1+5 = 6
        4: 15,   # ℓ=0,2,4: 1+5+9 = 15
        6: 28,   # ℓ=0,2,4,6: 1+5+9+13 = 28
        8: 45,   # ℓ=0,2,4,6,8: 1+5+9+13+17 = 45
    }
    
    for ell_max, expected in test_cases.items():
        result = get_num_coeffs(ell_max)
        print(f"  ell_max={ell_max}: {result} coeffs (expected {expected})")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ Coefficient calculation test passed!\n")


def test_angle_selection():
    """Test angle selection logic."""
    print("Testing angle selection...")
    
    # Simulate variability for 240 angles
    np.random.seed(42)
    total_angles = 240
    variability_per_angle = np.random.rand(total_angles)
    
    # Current indices (20 angles)
    current_indices = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 
                                50, 55, 60, 65, 70, 75, 80, 85, 90, 95])
    
    # Select top 2 angles not in current set
    b = 2
    available_mask = np.ones(total_angles, dtype=bool)
    available_mask[current_indices] = False
    
    available_variability = variability_per_angle.copy()
    available_variability[~available_mask] = -np.inf
    
    selected_indices = np.argsort(available_variability)[-b:][::-1]
    
    print(f"Current indices: {current_indices}")
    print(f"Selected top {b} angles: {selected_indices}")
    print(f"Selected variability: {variability_per_angle[selected_indices]}")
    print(f"Max variability in available: {np.max(variability_per_angle[available_mask])}")
    print(f"Min variability in available: {np.min(variability_per_angle[available_mask])}")
    
    # Verify selected angles not in current
    assert not np.any(np.isin(selected_indices, current_indices)), "Selected angles should not be in current set"
    
    print("✓ Angle selection test passed!\n")


def test_metrics():
    """Test metrics computation."""
    print("Testing metrics...")
    
    # Create dummy reconstructions
    np.random.seed(42)
    volume_shape = (64, 64, 64, 45)
    
    reconstruction = np.random.rand(*volume_shape)
    ground_truth = np.random.rand(*volume_shape)
    
    # MSE
    mse = np.mean((reconstruction - ground_truth) ** 2)
    print(f"MSE: {mse:.6f}")
    
    # MAE  
    mae = np.mean(np.abs(reconstruction - ground_truth))
    print(f"MAE: {mae:.6f}")
    
    print("✓ Metrics test passed!\n")


def test_iteration_logic():
    """Test active learning iteration logic."""
    print("Testing iteration logic...")
    
    total_projections = 240
    initial_size = 20
    num_iterations = 5
    b = 2
    
    # Simulate indices growth
    current_indices = np.arange(initial_size)
    indices_history = [current_indices.copy()]
    
    for i in range(num_iterations):
        # Simulate selecting b new angles
        available = np.setdiff1d(np.arange(total_projections), current_indices)
        selected = np.random.choice(available, size=b, replace=False)
        
        current_indices = np.sort(np.concatenate([current_indices, selected]))
        indices_history.append(current_indices.copy())
        
        print(f"Iteration {i+1}: {len(current_indices)} projections (added {selected})")
    
    expected_final_size = initial_size + num_iterations * b
    assert len(current_indices) == expected_final_size, f"Expected {expected_final_size}, got {len(current_indices)}"
    
    print(f"✓ Iteration logic test passed! Final size: {len(current_indices)}\n")


def test_subsampling():
    """Test subsampling logic."""
    print("Testing subsampling...")
    
    current_indices = np.arange(20)
    subsample_fraction = 0.8
    num_experiments = 5
    
    num_subsamples = int(subsample_fraction * len(current_indices))
    
    for exp_idx in range(num_experiments):
        subsample_indices = np.random.choice(
            current_indices,
            size=num_subsamples,
            replace=False
        )
        print(f"Experiment {exp_idx + 1}: sampled {len(subsample_indices)}/{len(current_indices)} projections")
        assert len(subsample_indices) == num_subsamples
        assert np.all(np.isin(subsample_indices, current_indices))
    
    print(f"✓ Subsampling test passed!\n")


if __name__ == "__main__":
    print("=" * 80)
    print("ACTIVE LEARNING LOGIC TESTS")
    print("=" * 80 + "\n")
    
    test_coefficient_calculation()
    test_angle_selection()
    test_metrics()
    test_iteration_logic()
    test_subsampling()
    
    print("=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
