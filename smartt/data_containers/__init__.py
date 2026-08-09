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
from .nielsen_synthetic import (
    NielsenMDataContainer,
    NielsenTDataContainer,
    NielsenMammothDataContainer,
)
from .cf_peek import CfPeekDataContainer
from .auditory_ossicle import AuditoryOssicleDataContainer
from .cf_carolina import CfCarolinaDataContainer
from .plastic_plasmonics import PlasticPlasmonicsDataContainer
from .steel_wire_waxs import SteelWireWaxsDataContainer
from .c4 import C4DataContainer
from .c5 import C5DataContainer
from .px_chameleon import PxChameleonDataContainer

REGISTRY: dict = {
    "b411":     B411DataContainer,
    "zenodo":   ZenodoDataContainer,
    "frogbone": FrogboneDataContainer,
    "fiber-synthetic": FiberSyntheticDataContainer,
    "fiber-synthetic-full": FiberSyntheticFullDataContainer,
    "nielsen-m": NielsenMDataContainer,
    "nielsen-t": NielsenTDataContainer,
    "nielsen-mammoth": NielsenMammothDataContainer,
    "cf-peek":  CfPeekDataContainer,
    "auditory-ossicle": AuditoryOssicleDataContainer,
    "cf-carolina": CfCarolinaDataContainer,
    "plastic-plasmonics": PlasticPlasmonicsDataContainer,
    "steel-wire-waxs": SteelWireWaxsDataContainer,
    "c4":       C4DataContainer,
    "c5":       C5DataContainer,
    "px-chameleon": PxChameleonDataContainer,
}


def get_dataset(name: str, **kwargs) -> SmarttDataContainer:
    """Return an instance of the dataset class registered under *name*.

    ``**kwargs`` are forwarded to the class constructor, e.g.
    ``get_dataset("c4", q=100)`` or ``get_dataset("frogbone", qbin=40)`` —
    equivalent to constructing the class directly (``C4DataContainer(q=100)``),
    just addressable by registry name (what CLI scripts like
    ``reconstruct_job.py``/``orchestrate_benchmark.py`` do).

    ``"cf-carolina"`` and ``"plastic-plasmonics"`` take their default q
    selector (see their classes); pass ``qbin=70`` / ``q=0.362`` to pick
    another. Likewise ``"steel-wire-waxs"`` takes its default peak; pass
    ``peak="111"`` to pick a different one. ``"c4"``, ``"c5"`` and
    ``"px-chameleon"`` each take their default q index (15); pass ``q=100``
    to pick another, or see e.g. ``C4DataContainer.list_qshells()`` for all
    available indices.
    ``"frogbone"`` takes its default qbin (9, back-compat); pass ``qbin=40``
    to pick another, or see ``FrogboneDataContainer.list_qshells()`` for all
    79 available q-bins.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}.  Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**kwargs)


__all__ = [
    "SmarttDataContainer",
    "B411DataContainer",
    "ZenodoDataContainer",
    "FrogboneDataContainer",
    "FiberSyntheticDataContainer",
    "FiberSyntheticFullDataContainer",
    "NielsenMDataContainer",
    "NielsenTDataContainer",
    "NielsenMammothDataContainer",
    "CfPeekDataContainer",
    "AuditoryOssicleDataContainer",
    "CfCarolinaDataContainer",
    "PlasticPlasmonicsDataContainer",
    "SteelWireWaxsDataContainer",
    "C4DataContainer",
    "C5DataContainer",
    "PxChameleonDataContainer",
    "REGISTRY",
    "get_dataset",
]
