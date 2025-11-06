"""
Example usage of the smartt library.

This script demonstrates how to:
1. Generate a reconstruction dataset from projection data
2. Load and use the dataset with PyTorch
3. Create data loaders for training
"""

import sys
from pathlib import Path

# Add parent directory to path to import smartt
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from smartt import generate_reconstruction_dataset, ReconstructionDataset


def example_generate_dataset():
    """Example: Generate a reconstruction dataset."""
    print("=" * 70)
    print("EXAMPLE 1: Generating Reconstruction Dataset")
    print("=" * 70)
    
    # Input data path (adjust to your data location)
    data_path = project_root / "sici" / "trabecular_bone_9.h5"
    
    # Output path
    output_path = project_root / "data" / "training_reconstructions.h5"
    
    # Check if input data exists
    if not data_path.exists():
        print(f"Input data not found at: {data_path}")
        print("Please adjust the data_path in the script.")
        return None
    
    print(f"Input: {data_path}")
    print(f"Output: {output_path}")
    print()
    
    # Generate dataset with a small number of iterations for demonstration
    generate_reconstruction_dataset(
        data_path=data_path,
        output_path=output_path,
        ell_max=8,
        num_projections=60,
        num_iterations=5,  # Small number for quick demonstration
        maxiter=20,
        regularization_weight=1.0,
        seed=42,  # For reproducibility
        verbose=True
    )
    
    print("\nDataset generation complete!")
    return output_path


def example_load_and_explore_dataset(hdf5_path):
    """Example: Load and explore a reconstruction dataset."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Loading and Exploring Dataset")
    print("=" * 70)
    
    if not hdf5_path.exists():
        print(f"Dataset not found at: {hdf5_path}")
        return None
    
    # Create dataset
    dataset = ReconstructionDataset(hdf5_path, load_in_memory=False)
    
    # Print dataset information
    print(f"\n{dataset}")
    
    # Get metadata
    print("\nDataset Metadata:")
    metadata = dataset.get_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Get a single sample
    print("\n--- Single Sample ---")
    reconstruction = dataset[0]
    print(f"Shape: {reconstruction.shape}")
    print(f"Dtype: {reconstruction.dtype}")
    print(f"Min value: {reconstruction.min():.6f}")
    print(f"Max value: {reconstruction.max():.6f}")
    print(f"Mean value: {reconstruction.mean():.6f}")
    
    # Get projection indices if available
    indices = dataset.get_projection_indices(0)
    if indices is not None:
        print(f"\nProjection indices used: {indices}")
    
    return dataset


def example_dataloader(dataset):
    """Example: Create and use a PyTorch DataLoader."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Using PyTorch DataLoader")
    print("=" * 70)
    
    # Create a DataLoader
    batch_size = 2
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for compatibility, increase for performance
    )
    
    print(f"\nDataLoader created with batch_size={batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Iterate through first few batches
    print("\n--- First 2 Batches ---")
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        print(f"Batch {i + 1}:")
        print(f"  Shape: {batch.shape}")
        print(f"  Dtype: {batch.dtype}")
        print(f"  Device: {batch.device}")
        print(f"  Memory: {batch.element_size() * batch.nelement() / 1024 / 1024:.2f} MB")


def example_with_transforms(hdf5_path):
    """Example: Using the dataset with custom transforms."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Transforms")
    print("=" * 70)
    
    if not hdf5_path.exists():
        print(f"Dataset not found at: {hdf5_path}")
        return
    
    # Define a simple transform (normalize to [0, 1])
    def normalize_transform(tensor):
        """Normalize each coefficient channel independently."""
        for i in range(tensor.shape[-1]):
            channel = tensor[..., i]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                tensor[..., i] = (channel - min_val) / (max_val - min_val)
        return tensor
    
    # Create dataset with transform
    dataset = ReconstructionDataset(
        hdf5_path,
        transform=normalize_transform,
        load_in_memory=False
    )
    
    print(f"Created dataset with normalization transform")
    
    # Get a sample
    reconstruction = dataset[0]
    print(f"\nTransformed sample:")
    print(f"  Shape: {reconstruction.shape}")
    print(f"  Min value: {reconstruction.min():.6f}")
    print(f"  Max value: {reconstruction.max():.6f}")
    print(f"  Mean value: {reconstruction.mean():.6f}")


def example_with_metadata(hdf5_path):
    """Example: Using the dataset with metadata return."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Returning Metadata")
    print("=" * 70)
    
    if not hdf5_path.exists():
        print(f"Dataset not found at: {hdf5_path}")
        return
    
    # Create dataset that returns metadata
    dataset = ReconstructionDataset(
        hdf5_path,
        return_metadata=True
    )
    
    print(f"Created dataset with return_metadata=True")
    
    # Get a sample with metadata
    reconstruction, metadata = dataset[0]
    print(f"\nSample with metadata:")
    print(f"  Reconstruction shape: {reconstruction.shape}")
    print(f"  Metadata: {metadata}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("SMARTT LIBRARY - USAGE EXAMPLES")
    print("=" * 70)
    
    # Example 1: Generate dataset (commented out by default to avoid long runtime)
    # Uncomment the following line to actually generate the dataset
    # output_path = example_generate_dataset()
    
    # For demonstration, use existing data if available
    output_path = project_root / "data" / "training_reconstructions.h5"
    
    if not output_path.exists():
        print(f"\nNo dataset found at {output_path}")
        print("To generate one, uncomment the example_generate_dataset() call in main()")
        print("\nYou can also run data generation separately:")
        print(f"  python -m smartt.data_processing <input.h5> <output.h5> [options]")
        return
    
    # Example 2: Load and explore
    dataset = example_load_and_explore_dataset(output_path)
    
    if dataset is None:
        return
    
    # Example 3: DataLoader
    example_dataloader(dataset)
    
    # Example 4: Transforms
    example_with_transforms(output_path)
    
    # Example 5: Metadata
    example_with_metadata(output_path)
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
