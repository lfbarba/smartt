#!/usr/bin/env python
"""
Test script to validate the updated dataset module.

This script performs basic validation of the new features without requiring
actual data files.
"""

import tempfile
import numpy as np
import h5py
from pathlib import Path
import torch
from torch.utils.data import DataLoader


def create_mock_hdf5_file(file_path: Path, num_projections: int = 100):
    """Create a mock HDF5 file for testing."""
    with h5py.File(file_path, 'w') as f:
        # Create mock data
        f.create_dataset('projections', data=np.zeros((num_projections, 10, 10)))
        f.attrs['num_projections'] = num_projections
    print(f"Created mock HDF5 file: {file_path}")


def test_multi_file_processing():
    """Test processing multiple HDF5 files."""
    print("\n" + "="*70)
    print("Test 1: Multi-file Processing")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock input directory with multiple files
        input_dir = tmpdir / "input"
        input_dir.mkdir()
        
        print(f"\nCreating mock input files in {input_dir}")
        for i in range(3):
            create_mock_hdf5_file(input_dir / f"scan_{i}.h5", num_projections=50)
        
        # List files found
        from smartt.data_processing import _get_h5_files
        files = _get_h5_files(input_dir)
        print(f"\nFound {len(files)} HDF5 files:")
        for file_path, file_id in files:
            print(f"  - {file_id}: {file_path.name}")
        
        assert len(files) == 3, f"Expected 3 files, found {len(files)}"
        print("\n✅ Multi-file detection works!")


def test_dataset_structure():
    """Test the new dataset structure."""
    print("\n" + "="*70)
    print("Test 2: Dataset Structure")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "test_dataset.h5"
        
        # Create a mock dataset file with new structure
        print(f"\nCreating mock dataset file: {output_file}")
        
        num_files = 2
        num_iterations = 5
        total_samples = num_files * num_iterations
        volume_shape = (32, 32, 32)
        num_coeffs = 81  # ell_max=8
        
        with h5py.File(output_file, 'w') as f:
            # Create datasets
            f.create_dataset(
                'reconstructions',
                data=np.random.randn(total_samples, *volume_shape, num_coeffs).astype(np.float32)
            )
            f.create_dataset(
                'ground_truths',
                data=np.random.randn(num_files, *volume_shape, num_coeffs).astype(np.float32)
            )
            
            # File identifiers
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(
                'file_identifiers',
                data=[f'scan_{i}' for i in range(num_files)],
                dtype=dt
            )
            
            # Mapping
            mapping = np.repeat(np.arange(num_files), num_iterations)
            f.create_dataset('reconstruction_to_gt_index', data=mapping.astype(np.int32))
            
            # Projection indices
            f.create_dataset(
                'projection_indices',
                data=np.random.randint(0, 100, size=(total_samples, 60))
            )
            
            # Metadata
            f.create_dataset('num_projections', data=60)
            f.create_dataset('ell_max', data=8)
            f.create_dataset('num_iterations', data=num_iterations)
            
            # Attributes
            f.attrs['num_coefficients'] = num_coeffs
            f.attrs['volume_shape'] = volume_shape
            f.attrs['num_files'] = num_files
        
        print("✅ Mock dataset created successfully!")
        
        # Test loading with ReconstructionDataset
        from smartt.dataset import ReconstructionDataset
        
        print("\nLoading dataset with ground truth...")
        dataset = ReconstructionDataset(output_file, return_ground_truth=True)
        
        print(f"\nDataset properties:")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Number of ground truths: {dataset.num_ground_truths}")
        print(f"  - Volume shape: {dataset.volume_shape}")
        print(f"  - Number of coefficients: {dataset.num_coefficients}")
        print(f"  - Has ground truth: {dataset.has_ground_truth}")
        print(f"  - File identifiers: {dataset.file_identifiers}")
        
        assert len(dataset) == total_samples, f"Expected {total_samples} samples"
        assert dataset.num_ground_truths == num_files, f"Expected {num_files} ground truths"
        assert dataset.has_ground_truth == True, "Should have ground truth"
        
        print("\n✅ Dataset loading works!")
        
        # Test getting a sample
        print("\nTesting sample retrieval...")
        sparse, gt = dataset[0]
        
        print(f"  - Sparse shape: {sparse.shape}")
        print(f"  - Ground truth shape: {gt.shape}")
        print(f"  - Sparse dtype: {sparse.dtype}")
        print(f"  - Ground truth dtype: {gt.dtype}")
        
        expected_shape = (*volume_shape, num_coeffs)
        assert tuple(sparse.shape) == expected_shape, f"Expected shape {expected_shape}"
        assert tuple(gt.shape) == expected_shape, f"Expected shape {expected_shape}"
        assert isinstance(sparse, torch.Tensor), "Should be torch.Tensor"
        assert isinstance(gt, torch.Tensor), "Should be torch.Tensor"
        
        print("\n✅ Sample retrieval works!")
        
        # Test file identifier retrieval
        print("\nTesting file identifier retrieval...")
        file_id = dataset.get_file_identifier(0)
        print(f"  - File identifier for sample 0: {file_id}")
        assert file_id in ['scan_0', 'scan_1'], f"Unexpected file identifier: {file_id}"
        
        print("\n✅ File identifier retrieval works!")
        
        # Test ground truth retrieval
        print("\nTesting ground truth retrieval...")
        gt_direct = dataset.get_ground_truth(0)
        print(f"  - Direct GT shape: {gt_direct.shape}")
        assert tuple(gt_direct.shape) == expected_shape, f"Expected shape {expected_shape}"
        
        print("\n✅ Ground truth retrieval works!")


def test_dataloader():
    """Test DataLoader integration."""
    print("\n" + "="*70)
    print("Test 3: DataLoader Integration")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "test_dataset.h5"
        
        # Create a mock dataset
        num_samples = 20
        volume_shape = (16, 16, 16)
        num_coeffs = 81
        num_files = 2
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset(
                'reconstructions',
                data=np.random.randn(num_samples, *volume_shape, num_coeffs).astype(np.float32)
            )
            f.create_dataset(
                'ground_truths',
                data=np.random.randn(num_files, *volume_shape, num_coeffs).astype(np.float32)
            )
            
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('file_identifiers', data=[f'scan_{i}' for i in range(num_files)], dtype=dt)
            f.create_dataset('reconstruction_to_gt_index', data=np.repeat(np.arange(num_files), num_samples // num_files).astype(np.int32))
            f.create_dataset('projection_indices', data=np.random.randint(0, 100, size=(num_samples, 60)))
            f.create_dataset('num_projections', data=60)
            f.create_dataset('ell_max', data=8)
            
            f.attrs['num_coefficients'] = num_coeffs
            f.attrs['volume_shape'] = volume_shape
        
        from smartt.dataset import ReconstructionDataset
        
        print("\nCreating dataset and dataloader...")
        dataset = ReconstructionDataset(output_file, return_ground_truth=True)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Number of batches: {len(dataloader)}")
        
        # Test first batch
        print("\nTesting first batch...")
        for sparse_batch, gt_batch in dataloader:
            print(f"  - Sparse batch shape: {sparse_batch.shape}")
            print(f"  - GT batch shape: {gt_batch.shape}")
            
            assert sparse_batch.shape[0] <= 4, "Batch size should be <= 4"
            assert sparse_batch.shape[1:] == (*volume_shape, num_coeffs), "Wrong sample shape"
            assert gt_batch.shape == sparse_batch.shape, "GT should match sparse shape"
            
            break
        
        print("\n✅ DataLoader integration works!")


def test_backward_compatibility():
    """Test backward compatibility mode."""
    print("\n" + "="*70)
    print("Test 4: Backward Compatibility")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "test_dataset.h5"
        
        # Create a mock dataset
        num_samples = 10
        volume_shape = (16, 16, 16)
        num_coeffs = 81
        num_files = 2
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset(
                'reconstructions',
                data=np.random.randn(num_samples, *volume_shape, num_coeffs).astype(np.float32)
            )
            f.create_dataset(
                'ground_truths',
                data=np.random.randn(num_files, *volume_shape, num_coeffs).astype(np.float32)
            )
            
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('file_identifiers', data=[f'scan_{i}' for i in range(num_files)], dtype=dt)
            f.create_dataset('reconstruction_to_gt_index', data=np.repeat(np.arange(num_files), num_samples // num_files).astype(np.int32))
            f.create_dataset('projection_indices', data=np.random.randint(0, 100, size=(num_samples, 60)))
            f.create_dataset('num_projections', data=60)
            f.create_dataset('ell_max', data=8)
            
            f.attrs['num_coefficients'] = num_coeffs
            f.attrs['volume_shape'] = volume_shape
        
        from smartt.dataset import ReconstructionDataset
        
        print("\nLoading dataset WITHOUT ground truth (backward compatible mode)...")
        dataset = ReconstructionDataset(output_file, return_ground_truth=False)
        
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Has ground truth: {dataset.has_ground_truth}")
        
        # Test getting a sample (should return only sparse)
        print("\nTesting sample retrieval...")
        result = dataset[0]
        
        print(f"  - Result type: {type(result)}")
        print(f"  - Result shape: {result.shape}")
        
        assert isinstance(result, torch.Tensor), "Should return single tensor"
        assert len(result.shape) == 4, "Should be 4D tensor"
        
        print("\n✅ Backward compatibility works!")


def test_metadata():
    """Test metadata handling."""
    print("\n" + "="*70)
    print("Test 5: Metadata Handling")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "test_dataset.h5"
        
        # Create a mock dataset with metadata
        num_samples = 10
        volume_shape = (16, 16, 16)
        num_coeffs = 81
        num_files = 2
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset(
                'reconstructions',
                data=np.random.randn(num_samples, *volume_shape, num_coeffs).astype(np.float32)
            )
            f.create_dataset(
                'ground_truths',
                data=np.random.randn(num_files, *volume_shape, num_coeffs).astype(np.float32)
            )
            
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('file_identifiers', data=[f'scan_{i}' for i in range(num_files)], dtype=dt)
            f.create_dataset('reconstruction_to_gt_index', data=np.repeat(np.arange(num_files), num_samples // num_files).astype(np.int32))
            f.create_dataset('projection_indices', data=np.random.randint(0, 100, size=(num_samples, 60)))
            f.create_dataset('num_projections', data=60)
            f.create_dataset('ell_max', data=8)
            f.create_dataset('num_iterations', data=5)
            f.create_dataset('maxiter', data=20)
            f.create_dataset('regularization_weight', data=1.0)
            
            f.attrs['num_coefficients'] = num_coeffs
            f.attrs['volume_shape'] = volume_shape
            f.attrs['num_files'] = num_files
            f.attrs['data_path'] = '/path/to/data'
        
        from smartt.dataset import ReconstructionDataset
        
        print("\nLoading dataset and retrieving metadata...")
        dataset = ReconstructionDataset(output_file, return_metadata=True)
        
        metadata = dataset.get_metadata()
        print(f"\nDataset-level metadata:")
        for key, value in metadata.items():
            if key != 'file_identifiers':
                print(f"  - {key}: {value}")
        
        # Test getting sample with metadata
        print("\nTesting sample with metadata...")
        (sparse, gt), item_metadata = dataset[0]
        
        print(f"\nSample metadata:")
        for key, value in item_metadata.items():
            if key != 'projection_indices':
                print(f"  - {key}: {value}")
        
        assert 'index' in item_metadata, "Should have index in metadata"
        assert 'file_identifier' in item_metadata, "Should have file_identifier in metadata"
        
        print("\n✅ Metadata handling works!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("SMARTT DATASET MODULE VALIDATION")
    print("="*70)
    
    try:
        test_multi_file_processing()
        test_dataset_structure()
        test_dataloader()
        test_backward_compatibility()
        test_metadata()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe updated dataset module is working correctly.")
        print("You can now use it with real data files.")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
