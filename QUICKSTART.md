# smartt - Quick Start Guide

## What is smartt?

A Python library that automates the generation of machine learning training datasets from tensor tomography projection data. It randomly samples projection subsets, performs mumott reconstructions, and packages results as PyTorch-compatible datasets.

## Installation

```bash
cd /path/to/smartTT
pip install -e .
```

## Verify Installation

```bash
python tests/test_smartt_structure.py
# Should show: ✓ ALL TESTS PASSED
```

## Basic Usage

### 1. Generate Training Data

```python
from smartt import generate_reconstruction_dataset

# Generate 100 reconstructions from random 60-projection subsets
generate_reconstruction_dataset(
    data_path='sici/trabecular_bone_9.h5',
    output_path='data/training.h5',
    ell_max=8,              # 81 spherical harmonic coefficients
    num_projections=60,     # Projections per reconstruction
    num_iterations=100,     # Generate 100 samples
    seed=42                 # For reproducibility
)
```

Or via command line:
```bash
python -m smartt.data_processing \
    sici/trabecular_bone_9.h5 \
    data/training.h5 \
    --ell-max 8 \
    --num-projections 60 \
    --num-iterations 100 \
    --seed 42
```

### 2. Load and Use in PyTorch

```python
from smartt import ReconstructionDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ReconstructionDataset('data/training.h5')

print(f"Dataset size: {len(dataset)}")
print(f"Sample shape: {dataset[0].shape}")  # (H, W, D, 81)

# Create DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Use in training
for batch in loader:
    # batch.shape: (4, H, W, D, 81)
    # Your training code here
    pass
```

## What Gets Generated?

The pipeline creates an HDF5 file containing:
- **Reconstructions**: 4D tensors (height, width, depth, coefficients)
- **Indices**: Which projections were used for each reconstruction
- **Metadata**: All generation parameters for reproducibility

## Examples

See complete examples:
```bash
python examples/usage_example.py
```

## Documentation

- **Full API Reference**: `smartt/README.md`
- **Quick Reference**: `smartt/OVERVIEW.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`

## File Structure

```
smartTT/
├── smartt/                      # Main library
│   ├── __init__.py
│   ├── data_processing.py       # Dataset generation
│   ├── dataset.py               # PyTorch Dataset
│   ├── README.md                # Full documentation
│   └── OVERVIEW.md              # Quick reference
├── examples/
│   └── usage_example.py         # Usage examples
├── tests/
│   └── test_smartt_structure.py # Tests
└── setup.py                     # Installation
```

## Common Workflows

### Quick Test (5 samples, ~5-10 minutes)
```python
generate_reconstruction_dataset(
    'input.h5', 'test.h5',
    num_iterations=5
)
```

### Development Set (50 samples)
```python
generate_reconstruction_dataset(
    'input.h5', 'dev.h5',
    num_iterations=50,
    seed=42
)
```

### Full Training Set (1000 samples)
```python
generate_reconstruction_dataset(
    'input.h5', 'train.h5',
    num_iterations=1000,
    num_projections=60,
    seed=42
)
```

## Key Features

✓ Random projection sampling  
✓ Automated mumott reconstruction  
✓ PyTorch Dataset integration  
✓ Memory-efficient HDF5 storage  
✓ GPU acceleration (automatic)  
✓ Reproducible (seed support)  
✓ Progress tracking  
✓ Command-line interface  

## Performance Tips

- Start with small `num_iterations` (5-10) to test
- Use CUDA-capable GPU for faster generation
- Adjust `num_projections` for quality/speed tradeoff
- Use `load_in_memory=False` for large datasets
- Set `seed` for reproducible research

## Troubleshooting

**Import errors**: Ensure smartTT is in Python path or installed
**Memory issues**: Use `load_in_memory=False` in ReconstructionDataset
**Slow generation**: Check CUDA availability, reduce num_projections
**File not found**: Use absolute paths, verify input file exists

## Support

For detailed documentation, see:
- `smartt/README.md` - Complete API reference
- `smartt/OVERVIEW.md` - Common use cases
- `examples/usage_example.py` - Working examples

## What's Next?

1. ✓ Install and test the library
2. Generate a small test dataset
3. Inspect the output
4. Build your ML pipeline
5. Scale up to full training set

---

**Ready to use!** 🚀
