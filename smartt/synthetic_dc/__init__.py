"""Build mumott ``DataContainer`` datasets from a ground-truth SH volume.

This is the shared machinery behind the ``SyntheticDataContainers`` and
``NpyToDataContainer`` notebooks.  Both start from a ``(X, Y, Z, n_coeffs)``
spherical-harmonics ground-truth tensor, forward-project it over a
Fibonacci-hemisphere set of directions, wrap the result in a ``DataContainer``
saved to HDF5, and persist the ground truth itself in the ``save_recon``
npy+JSON-sidecar format so benchmark notebooks can reload it with ``load_recon``.

The only difference between the two notebooks is where the ground truth comes
from — ``SyntheticDataContainers`` *builds* it by binning a composite volume,
whereas ``NpyToDataContainer`` *loads* it directly from an ``.npy`` file.  This
module owns everything that happens after the ground truth exists.
"""

from .from_gt import build_datacontainer_from_gt

__all__ = ["build_datacontainer_from_gt"]
