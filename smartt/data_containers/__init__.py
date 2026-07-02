"""SAXS-TT benchmark dataset registry.

Usage
-----
>>> from smartt.data_containers import get_dataset
>>> ds = get_dataset("b411")
>>> train_dc, held_dc = split_holdout(ds.get_main_dc())

Adding a new dataset
--------------------
1. Create ``smartt/data_containers/mydata.py`` with a subclass of
   :class:`~smartt.data_containers.base.SmarttDataContainer`.
2. Add an entry to :data:`REGISTRY` below.
"""

from .base import SmarttDataContainer
from .b411 import B411DataContainer
from .frogbone import FrogboneDataContainer
from .zenodo import ZenodoDataContainer
from .fiber_synthetic import FiberSyntheticDataContainer
from .synthetic_b411 import SyntheticB411DataContainer

REGISTRY: dict = {
    "b411":     B411DataContainer,
    "zenodo":   ZenodoDataContainer,
    "frogbone": FrogboneDataContainer,
    "fiber-synthetic": FiberSyntheticDataContainer,
    "synthetic-b411": SyntheticB411DataContainer,
}


def get_dataset(name: str) -> SmarttDataContainer:
    """Return an instance of the dataset class registered under *name*."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}.  Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]()


__all__ = [
    "SmarttDataContainer",
    "B411DataContainer",
    "ZenodoDataContainer",
    "FrogboneDataContainer",
    "FiberSyntheticDataContainer",
    "SyntheticB411DataContainer",
    "REGISTRY",
    "get_dataset",
]
