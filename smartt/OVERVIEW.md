# smartt Library Overview

## Project Structure

```
smartTT/
├── smartt/                      # Main library package
│   ├── __init__.py             # Package initialization and exports
│   ├── data_processing.py      # Dataset generation functions
│   ├── dataset.py              # PyTorch Dataset classes
│   └── README.md               # Comprehensive documentation
├── examples/
│   └── usage_example.py        # Complete usage examples
├── tests/
│   └── test_smartt_structure.py # Library structure tests
├── setup.py                     # Package installation script
└── requirements.txt            # Dependencies
```

## Quick Reference

### 1. Generate Training Data

**Python API:**
```python
from smartt import generate_reconstruction_dataset

generate_reconstruction_dataset(
    data_path='input.h5',
    output_path='training.h5',
    ell_max=8,
    num_projections=60,
    num_iterations=100,
    seed=42
)
```

**Command Line:**
```bash
python -m smartt.data_processing input.h5 output.h5 \
    --ell-max 8 \
    --num-projections 60 \
    --num-iterations 100 \
    --seed 42
```

### 2. Load and Train

```python
from smartt import ReconstructionDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ReconstructionDataset('training.h5')

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Training loop
for batch in dataloader:
    # batch.shape: (4, H, W, D, num_coeffs)
    # Your model training here
    pass
```

## Key Features

### Data Processing
- **Random Sampling**: Automatically samples random projection subsets
- **Mumott Integration**: Uses mumott for spherical harmonic reconstructions
- **Parallel Ready**: Supports CUDA acceleration when available
- **Reproducible**: Seed support for deterministic results
- **Progress Tracking**: Verbose output for monitoring long runs

### Dataset Class
- **PyTorch Compatible**: Native `torch.utils.data.Dataset` implementation
- **Memory Modes**: Choose between memory-efficient or fast in-memory loading
- **Flexible**: Support for custom transforms
- **Metadata Access**: Full access to reconstruction parameters and indices
- **Subset Support**: Work with specific coefficient channels

## Data Flow

```
┌─────────────────┐
│ Raw Projections │  trabecular_bone_9.h5
│   (247 images)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Random Sample  │  Select num_projections indices
│   (e.g., 60)    │  Repeat num_iterations times
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mumott Optimize │  Spherical harmonics (ell_max=8)
│  (LBFGS + Lap)  │  Results: (H, W, D, 81 coeffs)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HDF5 Dataset   │  training.h5
│ (100 samples)   │  Shape: (100, H, W, D, 81)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ReconstructionDS│  PyTorch Dataset
│   + DataLoader  │  → ML Training
└─────────────────┘
```

## Common Use Cases

### 1. Generate Small Test Dataset
```python
# Quick test with 5 iterations
generate_reconstruction_dataset(
    'input.h5', 'test.h5',
    num_iterations=5,
    verbose=True
)
```

### 2. Generate Large Training Dataset
```python
# Production dataset
generate_reconstruction_dataset(
    'input.h5', 'train.h5',
    ell_max=8,
    num_projections=60,
    num_iterations=1000,
    seed=42
)
```

### 3. Load with Normalization
```python
def normalize(tensor):
    for i in range(tensor.shape[-1]):
        ch = tensor[..., i]
        tensor[..., i] = (ch - ch.mean()) / ch.std()
    return tensor

dataset = ReconstructionDataset('train.h5', transform=normalize)
```

### 4. Work with Low-Order Coefficients Only
```python
from smartt.dataset import ReconstructionDatasetSubset

full_ds = ReconstructionDataset('train.h5')
# Only ell ≤ 2 (first 9 coefficients)
low_order_ds = ReconstructionDatasetSubset(full_ds, range(9))
```

### 5. Training Loop with Metadata
```python
dataset = ReconstructionDataset('train.h5', return_metadata=True)

for reconstruction, meta in dataset:
    print(f"Sample {meta['index']}")
    print(f"Used indices: {meta['projection_indices']}")
    # Train your model
```

## Configuration Guidelines

### Number of Projections
- **Few (20-40)**: Fast, lower quality reconstructions
- **Medium (50-70)**: Good balance for ML training
- **Many (80+)**: High quality, slower generation

### Spherical Harmonic Order (ell_max)
| ell_max | Coefficients | Use Case |
|---------|-------------|----------|
| 0 | 1 | Isotropic only |
| 2 | 9 | Fast prototyping |
| 4 | 25 | Low-order features |
| 6 | 49 | Medium detail |
| 8 | 81 | Full detail (recommended) |

### Number of Iterations
- **Development**: 5-10 samples for testing
- **Small Dataset**: 50-100 samples
- **Training Set**: 500-1000 samples
- **Large Dataset**: 5000+ samples

## Performance Tips

### Data Generation
1. Use CUDA if available (automatic detection)
2. Start with small `num_iterations` to test
3. Monitor disk space (each sample ~several MB)
4. Consider batch generation for very large datasets

### Dataset Loading
1. Use `load_in_memory=False` for large datasets
2. Use `load_in_memory=True` if dataset fits in RAM
3. Adjust DataLoader `num_workers` for parallel loading
4. Balance `batch_size` with available GPU memory

### Storage
- Typical sizes:
  - 100 samples ≈ 500 MB - 2 GB
  - 1000 samples ≈ 5 GB - 20 GB
- HDF5 compression is enabled by default
- Use SSD storage for faster I/O

## Verification

Run the structure test:
```bash
python tests/test_smartt_structure.py
```

Expected output: `✓ ALL TESTS PASSED`

## Installation

From the smartTT directory:
```bash
pip install -e .
```

This installs smartt in development mode with all dependencies.

## Getting Help

1. **README**: See `smartt/README.md` for full documentation
2. **Examples**: Run `python examples/usage_example.py`
3. **Docstrings**: All functions have comprehensive docstrings
4. **Tests**: Check `tests/` for usage patterns

## Troubleshooting

### Import Errors
```python
# Ensure smartTT is in your Python path
import sys
sys.path.insert(0, '/path/to/smartTT')
import smartt
```

### Memory Issues
- Use `load_in_memory=False`
- Reduce DataLoader `batch_size`
- Reduce number of DataLoader `num_workers`

### CUDA Issues
- Library auto-detects CUDA availability
- Falls back to CPU if CUDA unavailable
- Check: `torch.cuda.is_available()`

### File Not Found
- Use absolute paths when possible
- Check that input data exists before generating
- Ensure output directory has write permissions

## Next Steps

1. **Test Installation**: Run structure test
2. **Review Examples**: Check `examples/usage_example.py`
3. **Generate Small Dataset**: Test with 5-10 iterations
4. **Verify Output**: Inspect HDF5 file structure
5. **Create Training Pipeline**: Build your ML model
6. **Scale Up**: Generate full training dataset

## Support

For issues specific to:
- **smartt library**: Check this documentation
- **mumott**: See mumott documentation
- **PyTorch**: See PyTorch documentation
- **HDF5**: See h5py documentation
