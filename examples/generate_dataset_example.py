#!/usr/bin/env python
"""
Example script demonstrating the new dataset generation capabilities.

This script shows how to:
1. Generate datasets from a single HDF5 file
2. Generate datasets from a directory of HDF5 files
3. Load and use the dataset with ground truth pairs
"""

from pathlib import Path
from smartt.data_processing import generate_reconstruction_dataset
from smartt.dataset import ReconstructionDataset
from torch.utils.data import DataLoader


def example_single_file():
    """Example: Generate dataset from a single HDF5 file."""
    print("=" * 70)
    print("Example 1: Single HDF5 File")
    print("=" * 70)
    
    # Generate dataset from single file
    generate_reconstruction_dataset(
        data_path='path/to/your/data.h5',  # Replace with actual path
        output_path='output/training_data_single.h5',
        ell_max=8,
        num_projections=60,
        num_iterations=10,  # Small number for demo
        maxiter=20,
        regularization_weight=1.0,
        seed=42,
        verbose=True
    )
    
    print("\nDataset generated successfully!")


def example_multiple_files():
    """Example: Generate dataset from a directory of HDF5 files."""
    print("\n" + "=" * 70)
    print("Example 2: Directory of HDF5 Files")
    print("=" * 70)
    
    # Generate dataset from directory
    generate_reconstruction_dataset(
        data_path='path/to/your/data_directory/',  # Replace with actual path
        output_path='output/training_data_multi.h5',
        ell_max=8,
        num_projections=60,
        num_iterations=10,  # 10 iterations per file
        maxiter=20,
        regularization_weight=1.0,
        seed=42,
        verbose=True
    )
    
    print("\nDataset generated successfully!")


def example_load_and_use():
    """Example: Load and use the generated dataset."""
    print("\n" + "=" * 70)
    print("Example 3: Loading and Using the Dataset")
    print("=" * 70)
    
    # Load dataset with ground truth pairs
    dataset = ReconstructionDataset(
        'output/training_data_multi.h5',
        return_ground_truth=True,  # Returns (sparse, ground_truth) pairs
        load_in_memory=False  # Load from disk as needed
    )
    
    print(f"\nDataset loaded:")
    print(dataset)
    
    # Print metadata
    print("\nDataset metadata:")
    metadata = dataset.get_metadata()
    for key, value in metadata.items():
        if key != 'file_identifiers':  # Skip long list
            print(f"  {key}: {value}")
    
    if dataset.file_identifiers:
        print(f"\nSource files ({len(dataset.file_identifiers)}):")
        for i, file_id in enumerate(dataset.file_identifiers):
            print(f"  {i}: {file_id}")
    
    # Get a single sample
    print("\nLoading first sample...")
    sparse, ground_truth = dataset[0]
    print(f"  Sparse reconstruction shape: {sparse.shape}")
    print(f"  Ground truth shape: {ground_truth.shape}")
    print(f"  File identifier: {dataset.get_file_identifier(0)}")
    print(f"  Projection indices: {dataset.get_projection_indices(0)}")
    
    # Use with DataLoader
    print("\nCreating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Increase for parallel loading
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Get first batch
    for sparse_batch, gt_batch in dataloader:
        print(f"\nFirst batch:")
        print(f"  Sparse batch shape: {sparse_batch.shape}")
        print(f"  Ground truth batch shape: {gt_batch.shape}")
        break


def example_without_ground_truth():
    """Example: Use dataset without ground truth (backward compatible)."""
    print("\n" + "=" * 70)
    print("Example 4: Loading Without Ground Truth (Backward Compatible)")
    print("=" * 70)
    
    # Load dataset without ground truth
    dataset = ReconstructionDataset(
        'output/training_data_multi.h5',
        return_ground_truth=False,  # Only returns sparse reconstructions
        load_in_memory=False
    )
    
    print(f"\nDataset loaded (sparse only):")
    print(dataset)
    
    # Get a single sample (only sparse)
    print("\nLoading first sample...")
    sparse = dataset[0]
    print(f"  Sparse reconstruction shape: {sparse.shape}")
    
    # Use with DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in dataloader:
        print(f"\nFirst batch:")
        print(f"  Batch shape: {batch.shape}")
        break


def example_with_metadata():
    """Example: Load dataset with metadata."""
    print("\n" + "=" * 70)
    print("Example 5: Loading With Metadata")
    print("=" * 70)
    
    # Load dataset with metadata
    dataset = ReconstructionDataset(
        'output/training_data_multi.h5',
        return_ground_truth=True,
        return_metadata=True,
        load_in_memory=False
    )
    
    print(f"\nDataset loaded:")
    print(dataset)
    
    # Get a single sample with metadata
    print("\nLoading first sample with metadata...")
    (sparse, ground_truth), metadata = dataset[0]
    
    print(f"  Sparse shape: {sparse.shape}")
    print(f"  Ground truth shape: {ground_truth.shape}")
    print(f"  Metadata:")
    for key, value in metadata.items():
        if key != 'projection_indices':  # Skip long array
            print(f"    {key}: {value}")


if __name__ == "__main__":
    print("SmartTT Dataset Generation Examples")
    print("=" * 70)
    print("\nNOTE: Update the file paths in this script before running!")
    print("\nUncomment the examples you want to run:")
    print()
    
    # Uncomment the examples you want to run:
    
    # Example 1: Generate from single file
    # example_single_file()
    
    # Example 2: Generate from directory
    # example_multiple_files()
    
    # Example 3: Load and use with ground truth
    # example_load_and_use()
    
    # Example 4: Load without ground truth (backward compatible)
    # example_without_ground_truth()
    
    # Example 5: Load with metadata
    # example_with_metadata()
    
    print("\nDone! Uncomment examples above to run them.")
