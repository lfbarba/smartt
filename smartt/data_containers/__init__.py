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
from .fiber_synthetic_full import FiberSyntheticFullDataContainer
from .synthetic_b411 import SyntheticB411DataContainer
from .nielsen_synthetic import (
    NielsenMDataContainer,
    NielsenTDataContainer,
    NielsenMammothDataContainer,
)
from .cf_peek import CfPeekDataContainer
from .auditory_ossicle import AuditoryOssicleDataContainer
from .cf_carolina import CfCarolinaDataContainer
from .plastic_plasmonics import PlasticPlasmonicsDataContainer

REGISTRY: dict = {
    "b411":     B411DataContainer,
    "zenodo":   ZenodoDataContainer,
    "frogbone": FrogboneDataContainer,
    "fiber-synthetic": FiberSyntheticDataContainer,
    "fiber-synthetic-full": FiberSyntheticFullDataContainer,
    "synthetic-b411": SyntheticB411DataContainer,
    "nielsen-m": NielsenMDataContainer,
    "nielsen-t": NielsenTDataContainer,
    "nielsen-mammoth": NielsenMammothDataContainer,
    "cf-peek":  CfPeekDataContainer,
    "auditory-ossicle": AuditoryOssicleDataContainer,
    "cf-carolina": CfCarolinaDataContainer,
    "plastic-plasmonics": PlasticPlasmonicsDataContainer,
}


def get_dataset(name: str) -> SmarttDataContainer:
    """Return an instance of the dataset class registered under *name*.

    ``"cf-carolina"`` and ``"plastic-plasmonics"`` take their default q
    selector (see their classes). To pick a specific q, construct the class
    directly instead: ``CfCarolinaDataContainer(qbin=70)`` or
    ``PlasticPlasmonicsDataContainer(q=0.362)``.
    """
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
    "FiberSyntheticFullDataContainer",
    "SyntheticB411DataContainer",
    "NielsenMDataContainer",
    "NielsenTDataContainer",
    "NielsenMammothDataContainer",
    "CfPeekDataContainer",
    "AuditoryOssicleDataContainer",
    "CfCarolinaDataContainer",
    "PlasticPlasmonicsDataContainer",
    "REGISTRY",
    "get_dataset",
]
