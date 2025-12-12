"""
smartt - Smart Tensor Tomography Library

A library for processing tensor tomography projection data and generating
training datasets for machine learning applications.
"""

from .data_processing import generate_reconstruction_dataset
from .dataset import ReconstructionDataset

# Optimization submodule is available via smartt.optimization
# from . import optimization

__version__ = "0.1.0"
# __all__ = ["generate_reconstruction_dataset", "ReconstructionDataset", "optimization"]
