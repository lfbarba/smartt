#!/usr/bin/env python3
"""
Test script to verify the smartt library structure and imports.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import smartt
        print("✓ smartt package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import smartt: {e}")
        return False
    
    try:
        from smartt import generate_reconstruction_dataset
        print("✓ generate_reconstruction_dataset imported")
    except ImportError as e:
        print(f"✗ Failed to import generate_reconstruction_dataset: {e}")
        return False
    
    try:
        from smartt import ReconstructionDataset
        print("✓ ReconstructionDataset imported")
    except ImportError as e:
        print(f"✗ Failed to import ReconstructionDataset: {e}")
        return False
    
    try:
        from smartt.dataset import ReconstructionDatasetSubset
        print("✓ ReconstructionDatasetSubset imported")
    except ImportError as e:
        print(f"✗ Failed to import ReconstructionDatasetSubset: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_module_attributes():
    """Test that modules have expected attributes."""
    print("\nTesting module attributes...")
    
    import smartt
    
    expected_attrs = ['generate_reconstruction_dataset', 'ReconstructionDataset', '__version__']
    for attr in expected_attrs:
        if hasattr(smartt, attr):
            print(f"✓ smartt.{attr} exists")
        else:
            print(f"✗ smartt.{attr} missing")
            return False
    
    print(f"\nPackage version: {smartt.__version__}")
    return True


def test_dataset_class():
    """Test that dataset class has expected methods."""
    print("\nTesting ReconstructionDataset class...")
    
    from smartt import ReconstructionDataset
    
    expected_methods = ['__init__', '__len__', '__getitem__', 'get_metadata', 'get_projection_indices']
    for method in expected_methods:
        if hasattr(ReconstructionDataset, method):
            print(f"✓ ReconstructionDataset.{method} exists")
        else:
            print(f"✗ ReconstructionDataset.{method} missing")
            return False
    
    return True


def test_function_signatures():
    """Test that functions have expected signatures."""
    print("\nTesting function signatures...")
    
    from smartt import generate_reconstruction_dataset
    import inspect
    
    sig = inspect.signature(generate_reconstruction_dataset)
    params = list(sig.parameters.keys())
    
    expected_params = ['data_path', 'output_path', 'ell_max', 'num_projections', 
                      'num_iterations', 'maxiter', 'regularization_weight', 'seed', 'verbose']
    
    for param in expected_params:
        if param in params:
            print(f"✓ generate_reconstruction_dataset has parameter: {param}")
        else:
            print(f"✗ generate_reconstruction_dataset missing parameter: {param}")
            return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("SMARTT LIBRARY - STRUCTURE VERIFICATION")
    print("=" * 70)
    print()
    
    tests = [
        test_imports,
        test_module_attributes,
        test_dataset_class,
        test_function_signatures,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            results.append(False)
        print()
    
    print("=" * 70)
    if all(results):
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
