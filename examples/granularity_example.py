"""
Example: Using the granularity feature in ReconstructionDataset

This example demonstrates how to use the granularity parameter to control
whether you get full 4D volumes or individual coefficient vectors.
"""

from smartt import ReconstructionDataset
from torch.utils.data import DataLoader
import torch


def example_coarse_granularity():
    """Example using coarse granularity (full volumes)."""
    print("=" * 60)
    print("Example 1: Coarse Granularity (Full Volumes)")
    print("=" * 60)
    
    # Load dataset with coarse granularity
    dataset = ReconstructionDataset(
        'training_data.h5',
        granularity='coarse',  # Returns full 4D volumes
        return_ground_truth=True,
        load_in_memory=False
    )
    
    print(f"\nDataset info:")
    print(f"  Length: {len(dataset)} volumes")
    print(f"  Volume shape: {dataset.volume_shape}")
    print(f"  Coefficients: {dataset.num_coefficients}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Get a batch
    sparse_batch, gt_batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  Sparse: {sparse_batch.shape}")  # (2, H, W, D, 45)
    print(f"  Ground truth: {gt_batch.shape}")
    
    return dataset


def example_fine_granularity():
    """Example using fine granularity (individual voxels)."""
    print("\n" + "=" * 60)
    print("Example 2: Fine Granularity (Individual Voxels)")
    print("=" * 60)
    
    # Load dataset with fine granularity
    dataset = ReconstructionDataset(
        'training_data.h5',
        granularity='fine',  # Returns individual coefficient vectors
        return_ground_truth=True,
        load_in_memory=False  # Can also use True for faster access
    )
    
    print(f"\nDataset info:")
    print(f"  Length: {len(dataset)} voxels")
    print(f"  Vector dimension: {dataset.num_coefficients}")
    
    # Create dataloader with larger batch size (we have many more samples now)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Get a batch
    sparse_batch, gt_batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  Sparse: {sparse_batch.shape}")  # (256, 45)
    print(f"  Ground truth: {gt_batch.shape}")
    
    return dataset


def example_with_metadata():
    """Example using fine granularity with metadata."""
    print("\n" + "=" * 60)
    print("Example 3: Fine Granularity with Metadata")
    print("=" * 60)
    
    # Load dataset with metadata
    dataset = ReconstructionDataset(
        'training_data.h5',
        granularity='fine',
        return_ground_truth=True,
        return_metadata=True  # Get voxel coordinates and sample info
    )
    
    # Get a single item
    (sparse, gt), metadata = dataset[10000]
    
    print(f"\nSample 10000:")
    print(f"  Coefficient vector shape: {sparse.shape}")
    print(f"  From volume: {metadata['sample_index']}")
    print(f"  Voxel coordinates: {metadata['voxel_coords']}")
    print(f"  Linear voxel index: {metadata['voxel_index']}")
    
    return dataset


def example_training_loop():
    """Example training loop with fine granularity."""
    print("\n" + "=" * 60)
    print("Example 4: Training Loop with Fine Granularity")
    print("=" * 60)
    
    # Setup
    dataset = ReconstructionDataset(
        'training_data.h5',
        granularity='fine',
        return_ground_truth=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=512,  # Large batch size for individual vectors
        shuffle=True,
        num_workers=2
    )
    
    # Simple model example (replace with your actual model)
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = SimpleModel(
        input_dim=dataset.num_coefficients,
        hidden_dim=128,
        output_dim=dataset.num_coefficients
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\nTraining setup:")
    print(f"  Dataset size: {len(dataset):,} voxels")
    print(f"  Batch size: 512")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # Training loop (just one epoch for demo)
    model.train()
    total_loss = 0
    
    print(f"\nTraining (first 10 batches):")
    for i, (sparse, gt) in enumerate(dataloader):
        if i >= 10:  # Just show first 10 batches for demo
            break
        
        # Forward pass
        output = model(sparse)
        loss = criterion(output, gt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/10: loss = {loss.item():.6f}")
    
    avg_loss = total_loss / min(10, len(dataloader))
    print(f"\nAverage loss (first 10 batches): {avg_loss:.6f}")


if __name__ == "__main__":
    print("\nGranularity Feature Examples")
    print("=" * 60)
    print("\nThese examples demonstrate the two granularity modes:")
    print("  - coarse: Full 4D volumes (H, W, D, num_coefficients)")
    print("  - fine: Individual coefficient vectors (num_coefficients,)")
    print()
    
    # Note: Update the path to your actual HDF5 file
    print("Note: Update 'training_data.h5' to your actual file path in the code")
    print()
    
    # Run examples (comment out if file doesn't exist)
    # example_coarse_granularity()
    # example_fine_granularity()
    # example_with_metadata()
    # example_training_loop()
    
    print("\n" + "=" * 60)
    print("To run these examples:")
    print("1. Update the HDF5 file path in each function")
    print("2. Uncomment the example functions at the bottom")
    print("3. Run: python granularity_example.py")
    print("=" * 60)
