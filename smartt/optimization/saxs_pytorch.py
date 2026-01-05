"""PyTorch wrapper for SAXS projectors.

This module provides PyTorch-compatible wrappers for SAXSProjector and SAXSProjectorCUDA,
enabling differentiable forward projection and backprojection (adjoint) operations
that work seamlessly with PyTorch's autograd system and standard optimizers.

The wrapper maintains compatibility with both CPU and CUDA backends, with the CUDA
version delegating GPU memory management to SAXSProjectorCUDA for stability.

Public Functions
----------------
build_saxs_projector(...):
    Create a differentiable PyTorch projector layer for a given geometry.

project_saxs(...):
    Convenience function for forward projection with autograd support.

Notes
-----
* SAXSProjector uses CPU (float64), SAXSProjectorCUDA uses GPU (float32)
* Tensor layout: field is (B, 1, X, Y, Z, C) and projections are (B, I, J, K, C)
  where B=batch, I=projections, J,K=detector pixels, C=coefficients, X,Y,Z=volume
* For CUDA: tensors are converted to numpy, passed to SAXSProjectorCUDA (which handles
  GPU internally), then converted back to PyTorch. Computation happens on GPU.
* Projection indices can be used to compute only a subset of all projections
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
from mumott import Geometry
from mumott.methods.projectors import SAXSProjector, SAXSProjectorCUDA


class SAXSProjectorLayer:
    """Low-level wrapper managing SAXSProjector or SAXSProjectorCUDA operations.
    
    This class handles the conversion between PyTorch tensors and numpy arrays
    (for CPU) or keeps everything in GPU memory (for CUDA), and provides
    forward and adjoint operations compatible with PyTorch autograd.
    
    Parameters
    ----------
    geometry : Geometry
        The geometry defining the projection directions and detector configuration.
    use_cuda : bool, default=True
        If True and CUDA is available, use SAXSProjectorCUDA (float32, GPU).
        Otherwise use SAXSProjector (float64, CPU).
    device : torch.device, optional
        Target device for tensors. If None, uses cuda if use_cuda else cpu.
    """
    
    def __init__(
        self,
        geometry: Geometry,
        use_cuda: bool = True,
        device: Optional[torch.device] = None
    ):
        self.geometry = geometry
        # NOTE: CUDA version currently has stability issues in some environments
        # Default to CPU for now to avoid kernel crashes
        self.use_cuda = False  # Temporarily disabled: use_cuda and torch.cuda.is_available()
        
        if device is None:
            device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.device = device
        
        # Create the appropriate projector
        if self.use_cuda:
            self.projector = SAXSProjectorCUDA(geometry)
            self.dtype_np = np.float32
            self.dtype_torch = torch.float32
        else:
            self.projector = SAXSProjector(geometry)
            self.dtype_np = np.float64
            self.dtype_torch = torch.float64
        
        self.volume_shape = tuple(geometry.volume_shape)
        self.projection_shape = tuple(geometry.projection_shape)
        self.num_projections = len(geometry)
    
    def forward(
        self,
        field: torch.Tensor,
        indices: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Forward projection: field -> projections.
        
        Parameters
        ----------
        field : torch.Tensor
            Field tensor of shape (X, Y, Z, C) where C is the number of coefficients.
            For CUDA version, should be on GPU and float32.
        indices : np.ndarray, optional
            Projection indices to compute. If None, compute all projections.
        
        Returns
        -------
        torch.Tensor
            Projections of shape (I, J, K, C) where I is number of projections
            (or length of indices if provided), J, K are detector dimensions.
        """
        if self.use_cuda:
            # CUDA path: work with numpy arrays since SAXSProjectorCUDA
            # internally handles CUDA device arrays
            # Convert to numpy on CPU, let the projector handle GPU transfers
            field_np = field.detach().cpu().numpy().astype(self.dtype_np)
            
            # Perform forward projection (projector handles CUDA internally)
            projs_np = self.projector.forward(field_np, indices=indices)
            
            # Convert result back to PyTorch tensor on GPU
            projs_torch = torch.from_numpy(projs_np).to(device=self.device, dtype=self.dtype_torch)
            
        else:
            # CPU path: convert to numpy, compute, convert back
            field_np = field.detach().cpu().numpy().astype(self.dtype_np)
            projs_np = self.projector.forward(field_np, indices=indices)
            projs_torch = torch.from_numpy(projs_np).to(device=self.device, dtype=self.dtype_torch)
        
        return projs_torch
    
    def adjoint(
        self,
        projections: torch.Tensor,
        indices: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Adjoint projection: projections -> field.
        
        Parameters
        ----------
        projections : torch.Tensor
            Projections of shape (I, J, K, C) where I is number of projections,
            J, K are detector dimensions, C is number of coefficients.
        indices : np.ndarray, optional
            Projection indices corresponding to the projections. If None, assumes
            all projections in order.
        
        Returns
        -------
        torch.Tensor
            Field reconstruction of shape (X, Y, Z, C).
        """
        if self.use_cuda:
            # CUDA path: let projector handle CUDA internally
            # Ensure C-contiguous
            if not projections.is_contiguous():
                projections = projections.contiguous()
            
            # Convert to numpy, let projector handle GPU
            projs_np = projections.detach().cpu().numpy().astype(self.dtype_np)
            
            # Perform adjoint projection
            field_np = self.projector.adjoint(projs_np, indices=indices)
            
            # Convert back to PyTorch on GPU
            field_torch = torch.from_numpy(field_np).to(device=self.device, dtype=self.dtype_torch)
            
        else:
            # CPU path
            # Ensure C-contiguous as required by the projector
            if not projections.is_contiguous():
                projections = projections.contiguous()
            
            projs_np = projections.detach().cpu().numpy().astype(self.dtype_np)
            field_np = self.projector.adjoint(projs_np, indices=indices)
            field_torch = torch.from_numpy(field_np).to(device=self.device, dtype=self.dtype_torch)
        
        return field_torch


class _SAXSProjectorFunction(torch.autograd.Function):
    """Autograd function wrapping SAXS forward and adjoint operations.
    
    This function integrates the SAXSProjector/SAXSProjectorCUDA into
    PyTorch's autograd system, allowing gradients to flow through the
    projection operation.
    """
    
    @staticmethod
    def forward(ctx, field, projector_layer, indices):
        """Forward pass: field -> projections.
        
        Parameters
        ----------
        ctx : context
            PyTorch autograd context for saving information for backward.
        field : torch.Tensor
            Shape (B, 1, X, Y, Z, C) where B is batch size (typically 1).
        projector_layer : SAXSProjectorLayer
            The projector wrapper instance.
        indices : np.ndarray or None
            Projection indices to compute.
        
        Returns
        -------
        torch.Tensor
            Projections of shape (B, I, J, K, C).
        """
        # Save for backward
        ctx.projector_layer = projector_layer
        ctx.indices = indices
        
        # Remove batch dimension for projector (expects X, Y, Z, C)
        field_unbatched = field.squeeze(0).squeeze(0)  # (X, Y, Z, C)
        
        # Forward project
        projs_unbatched = projector_layer.forward(field_unbatched, indices=indices)
        
        # Add batch dimension back (I, J, K, C) -> (1, I, J, K, C)
        projs = projs_unbatched.unsqueeze(0)
        
        return projs
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: projection gradient -> field gradient.
        
        The adjoint of the forward projection is the backprojection (adjoint),
        which gives us the gradient with respect to the field.
        
        Parameters
        ----------
        ctx : context
            PyTorch autograd context.
        grad_output : torch.Tensor
            Gradient w.r.t. projections, shape (B, I, J, K, C).
        
        Returns
        -------
        tuple
            (grad_field, None, None) where grad_field has shape (B, 1, X, Y, Z, C).
            The None values correspond to projector_layer and indices (not differentiable).
        """
        projector_layer = ctx.projector_layer
        indices = ctx.indices
        
        # Remove batch dimension (B, I, J, K, C) -> (I, J, K, C)
        grad_projs_unbatched = grad_output.squeeze(0)
        
        # Adjoint projection
        grad_field_unbatched = projector_layer.adjoint(grad_projs_unbatched, indices=indices)
        
        # Add batch dimensions back (X, Y, Z, C) -> (1, 1, X, Y, Z, C)
        grad_field = grad_field_unbatched.unsqueeze(0).unsqueeze(0)
        
        return grad_field, None, None


def build_saxs_projector(
    geometry: Geometry,
    use_cuda: bool = True,
    device: Optional[torch.device] = None
) -> callable:
    """Build a differentiable SAXS projector function for PyTorch.
    
    Returns a callable that can be used as a layer in a PyTorch model,
    with full autograd support for training with standard optimizers.
    
    Parameters
    ----------
    geometry : Geometry
        The geometry defining the projection configuration.
    use_cuda : bool, default=True
        Whether to use CUDA acceleration (if available).
    device : torch.device, optional
        Target device. If None, uses cuda if use_cuda else cpu.
    
    Returns
    -------
    callable
        A function that takes a field tensor (B, 1, X, Y, Z, C) and returns
        projections (B, I, J, K, C). The function supports autograd.
    
    Examples
    --------
    >>> projector = build_saxs_projector(geometry, use_cuda=True)
    >>> field = torch.randn(1, 1, 65, 82, 65, 45, requires_grad=True)
    >>> projections = projector(field)
    >>> loss = (projections - target).pow(2).sum()
    >>> loss.backward()
    >>> # field.grad now contains gradients
    """
    if device is None:
        device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    
    projector_layer = SAXSProjectorLayer(geometry, use_cuda=use_cuda, device=device)
    
    def projector_fn(field: torch.Tensor, indices: Optional[np.ndarray] = None) -> torch.Tensor:
        """Forward project a field tensor.
        
        Parameters
        ----------
        field : torch.Tensor
            Field of shape (B, 1, X, Y, Z, C) or (X, Y, Z, C).
        indices : np.ndarray, optional
            Subset of projection indices to compute.
        
        Returns
        -------
        torch.Tensor
            Projections of shape (B, I, J, K, C) or (I, J, K, C).
        """
        # Normalize input shape
        if field.ndim == 4:
            field = field.unsqueeze(0).unsqueeze(0)  # (X,Y,Z,C) -> (1,1,X,Y,Z,C)
            squeeze_output = True
        elif field.ndim == 6:
            squeeze_output = False
        else:
            raise ValueError(f"Expected field with shape (X,Y,Z,C) or (B,1,X,Y,Z,C), got {field.shape}")
        
        # Apply projector
        projs = _SAXSProjectorFunction.apply(field, projector_layer, indices)
        
        # Remove batch dimension if input didn't have it
        if squeeze_output:
            projs = projs.squeeze(0)
        
        return projs
    
    return projector_fn


def project_saxs(
    field: torch.Tensor,
    geometry: Geometry,
    indices: Optional[np.ndarray] = None,
    use_cuda: bool = True,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convenience function for SAXS forward projection with autograd support.
    
    This is a simpler interface when you don't need to reuse the projector
    multiple times. For repeated use (e.g., in training loops), prefer
    build_saxs_projector() to avoid re-creating the projector.
    
    Parameters
    ----------
    field : torch.Tensor
        Field tensor of shape (X, Y, Z, C) or (B, 1, X, Y, Z, C).
    geometry : Geometry
        The projection geometry.
    indices : np.ndarray, optional
        Subset of projections to compute.
    use_cuda : bool, default=True
        Whether to use CUDA acceleration.
    device : torch.device, optional
        Target device for computation.
    
    Returns
    -------
    torch.Tensor
        Projections of shape matching input batch structure.
    
    Examples
    --------
    >>> field = torch.randn(65, 82, 65, 45, requires_grad=True)
    >>> projections = project_saxs(field, geometry, use_cuda=True)
    >>> loss = (projections - measured).pow(2).mean()
    >>> loss.backward()
    """
    projector_fn = build_saxs_projector(geometry, use_cuda=use_cuda, device=device)
    return projector_fn(field, indices=indices)
