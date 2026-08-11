"""Base class for SAXS-TT benchmark datasets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class SmarttDataContainer(ABC):
    """Abstract base for all benchmark datasets.

    Subclasses hard-code the data paths and the recipe to build each DC
    (main, remount, combined).  The orchestrator and worker scripts address
    datasets by name via the :func:`~smartt.data_containers.get_dataset`
    registry, so adding a new dataset only requires writing a new subclass.
    """

    name: str          # unique slug used in job names and cache keys
    has_remount: bool = False
    has_combined: bool = False
    is_waxs: bool = False   # True if geometry.two_theta is nonzero (Ewald curvature)

    # ------------------------------------------------------------------
    # Data access — override as needed
    # ------------------------------------------------------------------

    @abstractmethod
    def get_main_dc(self):
        """Primary training DataContainer (always available)."""
        ...

    def get_remount_dc(self):
        """Remount DataContainer, or ``None`` if not available."""
        return None

    def get_combined_dc(self):
        """Combined (both mounts) DataContainer, or ``None``."""
        return None

    @abstractmethod
    def get_cache_dir(self) -> Path:
        """Directory for cached reconstructions (created on demand)."""
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_dc(self, dc_type: str):
        """Return the DataContainer for *dc_type* (``'main'``, ``'remount'``, or ``'combined'``)."""
        if dc_type == "main":
            return self.get_main_dc()
        if dc_type == "remount":
            return self.get_remount_dc()
        if dc_type == "combined":
            return self.get_combined_dc()
        raise ValueError(
            f"Unknown dc_type {dc_type!r}.  Choose from 'main', 'remount', 'combined'."
        )

    def available_dc_types(self) -> List[str]:
        """List of dc_type strings available for this dataset."""
        types = ["main"]
        if self.has_remount:
            types.append("remount")
        if self.has_combined:
            types.append("combined")
        return types
