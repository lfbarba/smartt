from mumott import DataContainer
import numpy as np
import os
import sys
from scipy.optimize import Bounds
from mumott import DataContainer
from mumott.methods.basis_sets import GaussianKernels
from mumott.methods.projectors import SAXSProjector
from mumott.methods.residual_calculators import GradientResidualCalculator
from mumott.optimization.loss_functions import SquaredLoss
from mumott.optimization.optimizers import LBFGS
import pickle
import h5py

q_index = int(sys.argv[1])

# Load data
data_container = DataContainer(f'frogbone/dataset_qbin_{q_index:04d}.h5', nonfinite_replacement_value = 0)
with h5py.File(f'frogbone/dataset_qbin_{q_index:04d}.h5', 'r') as file:
    q =  float(file['q'][...])

# Parmeters of the reconstruction
basis_set = GaussianKernels(grid_scale=3)
projector = SAXSProjector(data_container.geometry)
functional = GradientResidualCalculator(data_container=data_container,
                                basis_set=basis_set,
                                projector=projector)
loss_function = SquaredLoss(functional, use_weights = True)
bounds = Bounds(lb = 0)
optimizer = LBFGS(loss_function, maxiter=20, maxfun=20, bounds = bounds)

# Run
results = optimizer.optimize()

# Save output
with open(f'recon/qbin_{q_index:04d}.npy','wb') as fid:
    pickle.dump({'recon':results['x'], 'q':q}, fid)
    
