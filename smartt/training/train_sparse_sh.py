"""
Training script for SH2SHNet on tensor tomography reconstructions.

This script trains a rotation-equivariant neural network to refine sparse
spherical harmonic reconstructions using the PyTorchExperiment framework.
"""
import os
import random
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split
from diffusers.optimization import get_cosine_schedule_with_warmup
import lovely_tensors as lt
lt.monkey_patch()  # Enable lovely_tensors for better tensor printing

# Import from smartt package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from smartt.dataset import ReconstructionDataset
from smartt.models.sh2shnet import SH2SHNet

# Import PyTorchExperiment framework
try:
    from pytorch_base.experiment import PyTorchExperiment
    from pytorch_base.base_loss import BaseLoss
except ImportError:
    raise ImportError(
        "pytorch_base not found. Please install it or ensure it's in your Python path."
    )


def create_mumott_to_e3nn_mapping(lmax: int = 8):
    """
    Create a mapping from mumott's even-ℓ only representation to e3nn's full representation.
    
    Mumott uses only even ℓ values: ℓ = 0, 2, 4, 6, 8
    For each ℓ, it includes all m values: -ℓ, ..., +ℓ
    
    This gives 45 coefficients for lmax=8:
    - ℓ=0: 1 coeff (m=0)
    - ℓ=2: 5 coeffs (m=-2,-1,0,1,2)
    - ℓ=4: 9 coeffs (m=-4,...,4)
    - ℓ=6: 13 coeffs (m=-6,...,6)
    - ℓ=8: 17 coeffs (m=-8,...,8)
    Total: 45
    
    e3nn uses all ℓ values: ℓ = 0, 1, 2, 3, 4, 5, 6, 7, 8
    This gives 81 coefficients: (8+1)^2 = 81
    
    Returns
    -------
    mumott_to_e3nn : list
        List of e3nn indices corresponding to each mumott coefficient.
        For odd ℓ coefficients not in mumott, we'll set them to zero.
    """
    mumott_to_e3nn = []
    e3nn_idx = 0
    mumott_idx = 0
    
    for l in range(lmax + 1):
        if l % 2 == 0:  # Even ℓ - these are in mumott
            for m in range(-l, l + 1):
                mumott_to_e3nn.append(e3nn_idx)
                e3nn_idx += 1
        else:  # Odd ℓ - skip these in e3nn indexing
            e3nn_idx += (2 * l + 1)
    
    return mumott_to_e3nn


# Cache for index mappings (device-specific)
_INDEX_CACHE = {}


def _get_padding_indices(lmax: int, device: torch.device) -> torch.Tensor:
    """
    Get cached indices for padding mumott (45-dim) to e3nn (81-dim).
    Returns a tensor of shape (45,) where each value is the target index in e3nn space.
    """
    cache_key = (lmax, device)
    if cache_key not in _INDEX_CACHE:
        indices = []
        mumott_idx = 0
        e3nn_idx = 0
        
        for l in range(lmax + 1):
            num_m = 2 * l + 1
            if l % 2 == 0:  # Even ℓ
                indices.extend(range(e3nn_idx, e3nn_idx + num_m))
                mumott_idx += num_m
            e3nn_idx += num_m
        
        _INDEX_CACHE[cache_key] = torch.tensor(indices, dtype=torch.long, device=device)
    
    return _INDEX_CACHE[cache_key]


def _get_extraction_indices(lmax: int, device: torch.device) -> torch.Tensor:
    """
    Get cached indices for extracting even-ℓ from e3nn (81-dim) to mumott (45-dim).
    Returns a tensor of shape (45,) where each value is the source index in e3nn space.
    """
    cache_key = (f'extract_{lmax}', device)
    if cache_key not in _INDEX_CACHE:
        indices = []
        e3nn_idx = 0
        
        for l in range(lmax + 1):
            num_m = 2 * l + 1
            if l % 2 == 0:  # Even ℓ
                indices.extend(range(e3nn_idx, e3nn_idx + num_m))
            e3nn_idx += num_m
        
        _INDEX_CACHE[cache_key] = torch.tensor(indices, dtype=torch.long, device=device)
    
    return _INDEX_CACHE[cache_key]


def pad_mumott_to_e3nn(mumott_coeffs: torch.Tensor, lmax: int = 8) -> torch.Tensor:
    """
    Pad mumott's 45-dimensional even-ℓ coefficients to e3nn's 81-dimensional representation.
    
    GPU-optimized version using precomputed index tensors for vectorized operations.
    
    Parameters
    ----------
    mumott_coeffs : torch.Tensor
        Tensor of shape (..., 45) with mumott coefficients.
    lmax : int
        Maximum ℓ value (default: 8).
    
    Returns
    -------
    e3nn_coeffs : torch.Tensor
        Tensor of shape (..., 81) with padded coefficients.
        Odd-ℓ coefficients are set to zero.
    """
    original_shape = mumott_coeffs.shape
    batch_shape = original_shape[:-1]
    
    # Flatten batch dimensions
    mumott_flat = mumott_coeffs.reshape(-1, 45)
    
    # Create output tensor (zeros for odd-ℓ coefficients)
    e3nn_flat = torch.zeros(
        mumott_flat.shape[0], 81, 
        device=mumott_coeffs.device, 
        dtype=mumott_coeffs.dtype
    )
    
    # Get precomputed indices and use index_put for vectorized assignment
    indices = _get_padding_indices(lmax, mumott_coeffs.device)
    e3nn_flat.index_copy_(1, indices, mumott_flat)
    
    # Reshape back to original batch shape
    return e3nn_flat.reshape(*batch_shape, 81)


def extract_even_l_from_e3nn(e3nn_coeffs: torch.Tensor, lmax: int = 8) -> torch.Tensor:
    """
    Extract even-ℓ coefficients from e3nn's 81-dimensional representation to mumott's 45-dim.
    
    GPU-optimized version using precomputed index tensors for vectorized operations.
    
    Parameters
    ----------
    e3nn_coeffs : torch.Tensor
        Tensor of shape (..., 81) with e3nn coefficients.
    lmax : int
        Maximum ℓ value (default: 8).
    
    Returns
    -------
    mumott_coeffs : torch.Tensor
        Tensor of shape (..., 45) with only even-ℓ coefficients.
    """
    original_shape = e3nn_coeffs.shape
    batch_shape = original_shape[:-1]
    
    # Flatten batch dimensions
    e3nn_flat = e3nn_coeffs.reshape(-1, 81)
    
    # Get precomputed indices and use index_select for vectorized extraction
    indices = _get_extraction_indices(lmax, e3nn_coeffs.device)
    mumott_flat = e3nn_flat.index_select(1, indices)
    
    # Reshape back to original batch shape
    return mumott_flat.reshape(*batch_shape, 45)


class SphericalHarmonicsLoss(BaseLoss):
    """
    Loss function for spherical harmonics reconstruction.
    
    This loss computes MSE between predicted and ground truth SH coefficients.
    The dataset returns 4D tensors of shape (H, W, D, 45) where 45 corresponds
    to mumott's even-ℓ only representation (ℓ = 0, 2, 4, 6, 8).
    
    The model (SH2SHNet) uses e3nn's full representation with 81 coefficients
    (all ℓ from 0 to 8). We pad the input to 81 dims, run the model, then
    extract the even-ℓ coefficients for loss computation.
    """
    def __init__(self, lmax: int = 8):
        stats_names = ["loss"]
        super().__init__(stats_names)
        self.mse = nn.MSELoss()
        self.lmax = lmax

    def compute_loss(self, instance, model: SH2SHNet):
        """
        Compute loss for a batch.
        
        Parameters
        ----------
        instance : tuple
            Tuple of (sparse_reconstruction, ground_truth) from ReconstructionDataset.
            Each has shape (batch_size, H, W, D, 45) where 45 is mumott's even-ℓ coeffs.
        model : SH2SHNet
            The spherical harmonics network (expects 81-dim e3nn format).
        
        Returns
        -------
        loss : torch.Tensor
            Scalar loss value.
        stats : dict
            Dictionary with loss statistics.
        """
        sparse_recon, ground_truth = instance
        
        device = next(model.parameters()).device
        sparse_recon = sparse_recon.float().to(device)
        ground_truth = ground_truth.float().to(device)
        
        # Pad to e3nn format: (B*H*W*D, 45) -> (B*H*W*D, 81)
        sparse_e3nn = pad_mumott_to_e3nn(sparse_recon, lmax=self.lmax)
        
        # Forward pass through model (in e3nn space)
        model.zero_grad()
        pred_e3nn = model(sparse_e3nn)
        
        # Extract even-ℓ coefficients: (B*H*W*D, 81) -> (B*H*W*D, 45)
        pred_flat = extract_even_l_from_e3nn(pred_e3nn, lmax=self.lmax)
        
        # Compute MSE loss on the even-ℓ coefficients
        loss = self.mse(pred_flat, ground_truth)
        
        return loss, {"loss": loss.item()}


def load_model(model, model_path):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Model loaded from checkpoint {model_path}")


def build_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Build train and test datasets from HDF5 file.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    Returns
    -------
    train_ds : ReconstructionDataset
        Training dataset.
    test_ds : ReconstructionDataset
        Test dataset.
    """
    # Load full dataset
    full_dataset = ReconstructionDataset(
        hdf5_path=args.data_path,
        return_ground_truth=True,
        load_in_memory=args.load_in_memory,
        return_metadata=False,
        granularity='fine'
    )
    
    print(f"\nDataset loaded:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Volume shape: {full_dataset.volume_shape}")
    print(f"  Num coefficients: {full_dataset.num_coefficients}")
    print(f"  ell_max: {full_dataset.ell_max}")
    print(f"  Num projections (sparse): {full_dataset.num_projections}")
    
    # Split into train and test
    total_size = len(full_dataset)
    train_size = int(args.train_split * total_size)
    test_size = total_size - train_size
    
    # Use random_split for reproducible splits
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, test_ds = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=generator
    )
    
    print(f"\nSplit into:")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    
    return train_ds, test_ds


def create_model(args) -> nn.Module:
    """
    Create SH2SHNet model.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    Returns
    -------
    model : SH2SHNet
        The spherical harmonics network.
    """
    model = SH2SHNet(
        lmax_in=args.lmax_in,
        lmax_out=args.lmax_out,
        lmax_hidden=args.lmax_hidden,
        mul_non_scalar=args.mul_non_scalar,
        scalars_hidden=args.scalars_hidden,
        n_blocks=args.n_blocks
    )
    
    print(f"\nModel created:")
    print(f"  lmax_in: {args.lmax_in}")
    print(f"  lmax_out: {args.lmax_out}")
    print(f"  lmax_hidden: {args.lmax_hidden}")
    print(f"  mul_non_scalar: {args.mul_non_scalar}")
    print(f"  scalars_hidden: {args.scalars_hidden}")
    print(f"  n_blocks: {args.n_blocks}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train SH2SHNet on sparse spherical harmonic reconstructions"
    )
    
    # Dataset / paths
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True,
        help='Path to HDF5 file with reconstruction data'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.95,
        help='Fraction of data to use for training (default: 0.95)'
    )
    parser.add_argument(
        '--load_in_memory',
        action='store_true',
        help='Load all data into memory (faster but uses more RAM)'
    )
    
    # Training hyperparams
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-3,
        help='Learning rate (default: 3e-3)'
    )
    parser.add_argument(
        '--scheduler_milestones',
        type=str,
        default='[20,40]',
        help='Learning rate scheduler milestones (default: [20,40])'
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.5,
        help='Learning rate decay factor (default: 0.5)'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help='Number of warmup steps for cosine scheduler (default: 500)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='sh2shnet_sparse',
        help='Experiment name (default: sh2shnet_sparse)'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=torch.get_num_threads(),
        help='Number of dataloader workers (default: number of CPU threads available)'
    )
    
    # Model architecture
    parser.add_argument(
        '--lmax_in',
        type=int,
        default=8,
        help='Maximum l for input SH (default: 8)'
    )
    parser.add_argument(
        '--lmax_out',
        type=int,
        default=8,
        help='Maximum l for output SH (default: 8)'
    )
    parser.add_argument(
        '--lmax_hidden',
        type=int,
        default=12,
        help='Maximum l for hidden layers (default: 12, allows bandwidth expansion)'
    )
    parser.add_argument(
        '--mul_non_scalar',
        type=int,
        default=4,
        help='Multiplicity for non-scalar irreps (default: 4)'
    )
    parser.add_argument(
        '--scalars_hidden',
        type=int,
        default=16,
        help='Number of scalar features per block (default: 16)'
    )
    parser.add_argument(
        '--n_blocks',
        type=int,
        default=3,
        help='Number of gated equivariant blocks (default: 3)'
    )
    
    # Checkpoint
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Process scheduler milestones
    milestones = args.scheduler_milestones.replace('[','').replace(']','').replace(' ','')
    milestones = [int(x) for x in milestones.split(',') if x]
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Build datasets
    train_ds, test_ds = build_datasets(args)
    
    # Create model
    model = create_model(args)
    
    # Load checkpoint if provided
    checkpoint_path = f"checkpoints/{args.exp_name}.pt"
    if args.checkpoint_path != "":
        checkpoint_path = args.checkpoint_path
        try:
            load_model(model, checkpoint_path)
        except Exception as e:
            print(f"Could not load model from {checkpoint_path}: {e}")
            print("Initializing randomly...")
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model.to(device)
    
    # Create loss function
    loss_fn = SphericalHarmonicsLoss(lmax=args.lmax_in)
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create experiment
    exp = PyTorchExperiment(
        args=vars(args),
        train_dataset=train_ds,
        test_dataset=test_ds,
        batch_size=args.batch_size,
        model=model,
        loss_fn=loss_fn,
        checkpoint_path=checkpoint_path,
        experiment_name=args.exp_name,
        with_wandb=args.wandb,
        num_workers=args.num_workers if args.load_in_memory == False else 0,
        seed=args.seed,
        save_always=True,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Create learning rate scheduler
    total_steps = len(exp.train_loader) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    exp.train(
        args.epochs, 
        optimizer, 
        milestones=milestones, 
        gamma=args.lr_decay, 
        scheduler=lr_scheduler
    )
    
    print(f"\nTraining complete! Model saved to {checkpoint_path}")


if __name__ == '__main__':
    main()

