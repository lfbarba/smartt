"""Preprocessing utilities for SAXS-TT volumes: spherical crop.

Public API
----------
make_sphere_mask(d)
spherical_crop(reconstruction, cube_size=None)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def make_sphere_mask(d: int) -> np.ndarray:
    """Boolean (d, d, d) mask; True inside the sphere inscribed in a d×d×d cube.

    Centre at ((d-1)/2, …), radius d/2 — the sphere just touches all six faces.
    """
    c = (d - 1) / 2.0
    coords = np.arange(d, dtype=np.float64) - c
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    return (X ** 2 + Y ** 2 + Z ** 2) <= (d / 2.0) ** 2


def spherical_crop(
    reconstruction: torch.Tensor,
    cube_size: Optional[int] = None,
) -> Tuple[torch.Tensor, np.ndarray, int]:
    """Center-crop and spherically mask a (K, X, Y, Z) reconstruction tensor.

    Parameters
    ----------
    reconstruction : (K, X, Y, Z) float tensor — raw reconstruction volumes.
    cube_size      : output side length (must be a multiple of 8 and ≤ min(X,Y,Z)).
                     If None, computed as ``(min(X, Y, Z) // 8) * 8``.

    Returns
    -------
    cropped     : (K, d, d, d) float tensor — cubic, exterior voxels zeroed.
    sphere_mask : (d, d, d) bool ndarray — True inside the inscribed sphere.
    d           : int — cube side length used.
    """
    K, X, Y, Z = reconstruction.shape

    if cube_size is None:
        d = (min(X, Y, Z) // 8) * 8
    else:
        d = int(cube_size)

    if d <= 0:
        raise ValueError(
            f"cube_size={d} is non-positive (volume shape {X},{Y},{Z}). "
            "Smallest dimension must be ≥ 8."
        )
    if d > min(X, Y, Z):
        raise ValueError(
            f"cube_size={d} exceeds the smallest volume dimension {min(X,Y,Z)}."
        )

    x0 = (X - d) // 2
    y0 = (Y - d) // 2
    z0 = (Z - d) // 2
    cropped = reconstruction[:, x0:x0 + d, y0:y0 + d, z0:z0 + d].contiguous()

    sphere_mask = make_sphere_mask(d)
    mask_t = torch.from_numpy(sphere_mask).to(cropped.device)
    cropped = cropped * mask_t.unsqueeze(0)

    return cropped, sphere_mask, d
