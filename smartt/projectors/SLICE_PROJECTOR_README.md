# Spherical Harmonic Slice Projector

This module implements projection of spherical harmonic coefficients onto orthogonal circular slices in reciprocal space, corresponding to different projection angles in tomographic reconstruction.

## Overview

When performing SAXS tensor tomography, each projection corresponds to a different angle. The measured data at each angle represents a slice through reciprocal space orthogonal to the projection direction. The `SphericalHarmonicSliceProjector` computes these slices from spherical harmonic coefficient volumes.

## Mathematical Background

### Spherical Function Representation

A 3D spherical function is expanded in spherical harmonics:

```
f(r, θ, φ) = Σ_ℓ Σ_m c_ℓm Y_ℓ^m(θ, φ)
```

### Circular Slicing

To take a circular slice orthogonal to a direction **n**:

1. **Rotate** coordinates so **n** aligns with the z-axis
2. **Evaluate** at the equator (θ = π/2)

At the equator, the associated Legendre polynomials satisfy:
- `P_ℓ^m(0) = 0` when ℓ+m is odd
- For even-ℓ basis (ℓ=0,2,4,6,8), only even-m terms survive

The result is a **Fourier series** (circular harmonics):

```
f_slice(φ) = Σ_ℓm c_ℓm P_ℓ^m(0) Y_ℓ^m(π/2, φ)
```

where `Y_ℓ^m(π/2, φ) ∝ cos(mφ)` or `sin(mφ)`.

### Output Dimensions

- **Input**: 45 SH coefficients (ℓ=0,2,4,6,8 with all m: 1+5+9+13+17=45)
- **Output**: ~25 non-zero Fourier coefficients per slice
- These form a 1D function around each circular slice

## Usage

### Basic Example

```python
import torch
from smartt.projectors import SphericalHarmonicSliceProjector
from mumott.data_handling import DataContainer

# Load geometry
dc = DataContainer('dataset.h5')
geometry = dc.geometry

# Create projector
device = torch.device('cuda')
projector = SphericalHarmonicSliceProjector(
    ell_max=8,
    geometry=geometry,
    device=device,
    use_rotation=False  # Set True for proper rotation
)

# Option 1: Get compact harmonic coefficients (~25 per slice)
sh_coeffs = torch.randn(45, device=device)
harmonic_coeffs = projector.project_to_slices(sh_coeffs, output_type='coefficients')
# Output shape: (num_projections, 25)
# This is the compact representation - best for storage/transmission

# Option 2: Get evaluated samples for visualization
samples = projector.project_to_slices(sh_coeffs, output_type='samples', phi_samples=360)
# Output shape: (num_projections, 360)
# This is for visualization - 7x more storage than coefficients

# Works with volumes too
sh_volume = torch.randn(65, 82, 65, 45, device=device)
harmonic_vol = projector.project_to_slices(sh_volume, output_type='coefficients')
# Output shape: (num_projections, 65, 82, 65, 25)
```

### With Proper Rotation (Experimental)

```python
# Enable Wigner D-matrix rotation for proper geometric transformation
projector = SphericalHarmonicSliceProjector(
    ell_max=8,
    geometry=geometry,
    device=device,
    use_rotation=True  # Uses e3nn Wigner D-matrices
)

slices = projector.project_to_slices(sh_coeffs, phi_samples=360)
```

## Implementation Details

### What Works

✅ **Projection vector computation** - Correctly extracts angles from mumott geometry  
✅ **Legendre function evaluation** - Pre-computes P_ℓ^m(0) for all modes  
✅ **Equator evaluation** - Properly evaluates surviving terms at θ=π/2  
✅ **Batch processing** - Handles single points or full volumes  
✅ **GPU acceleration** - All operations run on CUDA when available  

### Current Limitations

⚠️ **Rotation** - Wigner D-matrix rotation is optional (use_rotation=True)
  - When disabled: assumes projection vectors aligned with z-axis (fast but approximate)
  - When enabled: uses e3nn for proper rotation (experimental)

### Performance

- Single point (45 coeffs → 360 phi samples, 240 projections): ~10ms on GPU
- Small volume (5×5×5, 45 coeffs → 90 phi samples): ~100ms on GPU
- Memory scales linearly with volume size and phi resolution

## Integration with ASTRA Projector

The slice projector complements the ASTRA tomographic projector:

1. **ASTRA Projector** (`astra_projector.py`)
   - Projects volumes (X, Y, Z) → sinograms (I, J, K)
   - Used for gradient descent reconstruction
   - Handles real-space geometry

2. **Slice Projector** (`slice_projector.py`)
   - Projects SH coefficients (X, Y, Z, 45) → circular slices (I, J, K, φ)
   - Used for reciprocal space analysis
   - Handles orientation-dependent scattering

Together they enable:
- Forward model: SH coefficients → circular slices → detector patterns
- Inverse problem: detector patterns → SH coefficients via gradient descent

## API Reference

### Class: SphericalHarmonicSliceProjector

#### Constructor

```python
SphericalHarmonicSliceProjector(
    ell_max: int,
    geometry: mumott.Geometry,
    device: Optional[torch.device] = None,
    use_rotation: bool = False
)
```

**Parameters:**
- `ell_max`: Maximum spherical harmonic degree (must be even)
- `geometry`: mumott Geometry object with projection angles
- `device`: torch device for computation (default: CPU)
- `use_rotation`: Enable Wigner D-matrix rotation (default: False)

#### Methods

```python
def project_to_slices(
    coeffs: torch.Tensor,
    output_type: str = 'coefficients',
    phi_samples: Optional[int] = 360
) -> torch.Tensor
```

Project SH coefficients to circular slices.

**Parameters:**
- `coeffs`: Shape (X, Y, Z, num_coeffs) or (num_coeffs,)
- `output_type`: 'coefficients' for compact representation (~25 per slice) or 'samples' for evaluated function
- `phi_samples`: Azimuthal resolution (only used if output_type='samples')

**Returns:**
- If output_type='coefficients':
  - Shape (num_projections, num_surviving_coeffs) or (num_projections, X, Y, Z, num_surviving_coeffs)
  - ~25 Fourier coefficients per slice (compact)
- If output_type='samples':
  - Shape (num_projections, phi_samples) or (num_projections, X, Y, Z, phi_samples)
  - Evaluated function at phi angles (for visualization)

**Storage comparison** (single point, 240 projections):
- Coefficients: 240 × 25 = 6,000 values
- Samples (360°): 240 × 360 = 86,400 values (14.4× more storage)

```python
def forward(
    coeffs: torch.Tensor,
    phi_samples: Optional[int] = 360
) -> torch.Tensor
```

Alias for `project_to_slices()` for PyTorch compatibility.

## Testing

Run the test suite:

```bash
python tests/test_slice_projector.py
```

Example notebook:
```bash
jupyter notebook notebooks/astra_projector.ipynb
# See "Spherical Harmonic Slice Projector" section
```

## References

1. **Spherical Harmonics**: Restricted to circles on the sphere
   - Only even ℓ+m terms survive at equator
   - Reduces to Fourier series in φ

2. **Wigner D-matrices**: For rotation of SH coefficients
   - Implemented via e3nn library
   - Preserves angular momentum structure

3. **SAXS Tensor Tomography**: Application context
   - Each projection measures reciprocal space slice
   - SH coefficients encode orientation distribution
   - Circular slice represents scattering pattern

## Future Work

- [ ] Optimize Wigner D-matrix computation
- [ ] Implement adjoint operation (backprojection from slices)
- [ ] Add gradient computation for optimization
- [ ] Support arbitrary slice orientations (not just geometry angles)
- [ ] Integrate with forward model for full reconstruction pipeline
