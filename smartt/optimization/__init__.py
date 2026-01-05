"""PyTorch optimization utilities for smartt.

This module provides PyTorch-compatible wrappers for SAXS projectors,
enabling the use of standard PyTorch optimizers instead of LBFGS.
"""

from .saxs_pytorch import (
    build_saxs_projector,
    project_saxs,
    SAXSProjectorLayer
)

__all__ = [
    'build_saxs_projector',
    'project_saxs',
    'SAXSProjectorLayer',
]
