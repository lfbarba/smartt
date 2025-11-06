# smartt - Smart Tensor Tomography Library

A Python library for processing tensor tomography projection data and generating training datasets for machine learning applications. The `smartt` library automates the workflow of sampling projection subsets, performing mumott reconstructions, and creating PyTorch-compatible datasets.

## Features

- **Automated Data Generation**: Randomly sample projection subsets and generate spherical harmonic reconstructions
- **PyTorch Integration**: Native PyTorch Dataset implementation for seamless ML pipeline integration
- **Flexible Configuration**: Customize spherical harmonic degree, number of projections, regularization, and more
- **Memory Efficient**: Options for in-memory or on-disk data loading
- **Reproducible**: Support for random seeds to ensure reproducibility
- **Extensible**: Easy to add custom transforms and subset selections

## Installation

1. Ensure you have the required dependencies:
```bash
pip install numpy torch h5py mumott
```

2. Add the smartt library to your Python path, or install in development mode:
```bash
cd /path/to/smartTT
pip install -e .
```

## Quick Start

### 1. Generate a Reconstruction Dataset

```python
from smartt import generate_reconstruction_dataset

# Generate dataset from projection data
generate_reconstruction_dataset(
    data_path='trabecular_bone_9.h5',
    output_path='training_data.h5',
    ell_max=8,                    # Spherical harmonic degree (45 coefficients)
    num_projections=60,           # Projections per reconstruction
    num_iterations=100,           # Number of different samplings
    maxiter=20,                   # LBFGS optimization iterations
    regularization_weight=1.0,    # Laplacian regularization
    seed=42,                      # Random seed for reproducibility
    verbose=True
)
```

**Command-line interface:**
```bash
python -m smartt.data_processing input.h5 output.h5 \
    --ell-max 8 \
    --num-projections 60 \
    --num-iterations 100 \
    --seed 42
```

### 2. Load and Use the Dataset

```python
from smartt import ReconstructionDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ReconstructionDataset('training_data.h5')

# Inspect dataset
print(f"Dataset size: {len(dataset)}")
print(f"Volume shape: {dataset.volume_shape}")
print(f"Number of coefficients: {dataset.num_coefficients}")

# Get a single sample
reconstruction = dataset[0]  # Shape: (H, W, D, num_coeffs)

# Create DataLoader for training
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# Training loop
for batch in dataloader:
    # batch shape: (batch_size, H, W, D, num_coeffs)
    # Your training code here
    pass
```

## API Reference

### Data Processing

#### `generate_reconstruction_dataset`

Generate a dataset of tensor tomography reconstructions from random projection subsets.

**Parameters:**
- `data_path` (str or Path): Path to input HDF5 file with projection data
- `output_path` (str or Path): Path for output HDF5 file
- `ell_max` (int, default=8): Maximum spherical harmonic degree
- `num_projections` (int, default=60): Number of projections per reconstruction
- `num_iterations` (int, default=100): Number of different random samplings
- `maxiter` (int, default=20): LBFGS optimizer max iterations
- `regularization_weight` (float, default=1.0): Laplacian regularization weight
- `seed` (int, optional): Random seed for reproducibility
- `verbose` (bool, default=True): Print progress information

**Output HDF5 Structure:**
```
reconstructions/          # Shape: (num_iterations, H, W, D, num_coeffs)
projection_indices/       # Shape: (num_iterations, num_projections)
num_projections           # Scalar
ell_max                   # Scalar
num_iterations            # Scalar
maxiter                   # Scalar
regularization_weight     # Scalar
```

### Dataset Classes

#### `ReconstructionDataset`

PyTorch Dataset for tensor tomography reconstructions.

**Parameters:**
- `hdf5_path` (str or Path): Path to HDF5 file
- `transform` (callable, optional): Transform to apply to each tensor
- `load_in_memory` (bool, default=False): Load all data into memory at initialization
- `return_metadata` (bool, default=False): Return (tensor, metadata) tuples

**Attributes:**
- `num_samples`: Total number of reconstructions
- `volume_shape`: Shape of 3D volume (H, W, D)
- `num_coefficients`: Number of spherical harmonic coefficients
- `ell_max`: Maximum spherical harmonic degree
- `num_projections`: Number of projections used per reconstruction

**Methods:**
- `__len__()`: Returns number of samples
- `__getitem__(idx)`: Returns reconstruction tensor at index
- `get_metadata()`: Returns dataset-level metadata dictionary
- `get_projection_indices(idx)`: Returns projection indices for sample

#### `ReconstructionDatasetSubset`

A subset dataset with specific coefficient channels.

**Parameters:**
- `base_dataset` (ReconstructionDataset): Base dataset to create subset from
- `coefficient_indices` (array-like): Indices of coefficients to include

**Example:**
```python
# Only use first 9 coefficients (ell ≤ 2)
subset = ReconstructionDatasetSubset(dataset, coefficient_indices=range(9))
```

## Advanced Usage

### Custom Transforms

```python
def normalize_transform(tensor):
    """Normalize each coefficient channel to [0, 1]."""
    for i in range(tensor.shape[-1]):
        channel = tensor[..., i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            tensor[..., i] = (channel - min_val) / (max_val - min_val)
    return tensor

dataset = ReconstructionDataset(
    'training_data.h5',
    transform=normalize_transform
)
```

### Memory Management

For large datasets, use on-disk loading:
```python
# Memory efficient - reads from disk on each access
dataset = ReconstructionDataset('large_data.h5', load_in_memory=False)
```

For smaller datasets or faster training:
```python
# Loads all data into RAM at initialization
dataset = ReconstructionDataset('small_data.h5', load_in_memory=True)
```

### Accessing Metadata

```python
# Get dataset-level metadata
dataset = ReconstructionDataset('training_data.h5')
metadata = dataset.get_metadata()
print(f"Source data: {metadata['source_data_path']}")
print(f"Regularization: {metadata['regularization_weight']}")

# Get per-sample metadata
dataset = ReconstructionDataset('training_data.h5', return_metadata=True)
reconstruction, meta = dataset[0]
print(f"Projection indices: {meta['projection_indices']}")
```

### Coefficient Subsets

Work with specific spherical harmonic orders:
```python
from smartt.dataset import ReconstructionDatasetSubset

# Full dataset
full_dataset = ReconstructionDataset('training_data.h5')

# Only use coefficients for ell ≤ 4 (first 25 coefficients)
low_order_dataset = ReconstructionDatasetSubset(
    full_dataset,
    coefficient_indices=range(25)
)
```

## Spherical Harmonic Coefficients

The number of coefficients for a given `ell_max` is `(ell_max + 1)²`:

| ell_max | Coefficients | Description |
|---------|-------------|-------------|
| 0 | 1 | Isotropic |
| 2 | 9 | Low order |
| 4 | 25 | Medium order |
| 6 | 49 | Higher order |
| 8 | 81 | Full detail |

## Examples

See `examples/usage_example.py` for comprehensive examples including:
- Dataset generation
- Loading and exploration
- PyTorch DataLoader usage
- Custom transforms
- Metadata access

Run the examples:
```bash
python examples/usage_example.py
```

## Workflow Overview

The typical workflow using smartt:

1. **Data Generation** (one-time, computationally intensive):
   ```
   Raw projections → Random sampling → mumott optimization → HDF5 dataset
   ```

2. **Training** (repeated, fast):
   ```
   HDF5 dataset → ReconstructionDataset → DataLoader → ML model
   ```

## Requirements

- Python ≥ 3.7
- NumPy
- PyTorch
- h5py
- mumott (tensor tomography reconstruction library)

## File Structure

```
smartt/
├── __init__.py           # Package initialization
├── data_processing.py    # Dataset generation functions
└── dataset.py            # PyTorch Dataset classes

examples/
└── usage_example.py      # Comprehensive usage examples
```

## Tips and Best Practices

1. **Start Small**: Begin with a small `num_iterations` (e.g., 5-10) to test your pipeline
2. **Use Seeds**: Always use a seed for reproducibility in research
3. **Monitor Progress**: Keep `verbose=True` during data generation
4. **Batch Size**: Adjust DataLoader batch size based on available memory and tensor size
5. **Parallel Workers**: Use `num_workers > 0` in DataLoader for faster training
6. **Data Inspection**: Always inspect the first few samples before training

## Troubleshooting

### Memory Issues
- Use `load_in_memory=False` for large datasets
- Reduce batch size in DataLoader
- Use fewer workers in DataLoader

### Slow Data Loading
- Set `load_in_memory=True` if dataset fits in RAM
- Increase `num_workers` in DataLoader
- Consider using SSD storage for HDF5 files

### Reproducibility
- Always set `seed` parameter
- Use the same PyTorch and NumPy versions
- Document your hardware (especially CUDA version)

## Citation

If you use this library in your research, please cite the associated paper and the mumott library.

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

[Your contact information]
