from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from e3nn import o3


def _default_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cpu")


def _default_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is not None:
        return dtype
    return torch.get_default_dtype()


@dataclass(frozen=True)
class SphericalHarmonicLobes:
    """Container with geometry data for rendering spherical harmonic lobes."""

    l: int
    m: int
    theta: torch.Tensor
    phi: torch.Tensor
    radius: torch.Tensor
    values: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor


@torch.no_grad()
def generate_spherical_harmonic_lobes(
    l: int,
    m: int,
    *,
    base_radius: float = 1.0,
    amplitude: float = 0.4,
    theta_resolution: int = 120,
    phi_resolution: int = 240,
    normalization: str = "component",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> SphericalHarmonicLobes:
    """Generate geometry for visualising real spherical harmonics as lobed surfaces.

    Parameters
    ----------
    l:
        Degree of the spherical harmonic (ℓ ≥ 0).
    m:
        Order of the spherical harmonic (|m| ≤ ℓ).
    base_radius:
        Baseline sphere radius onto which lobes are superimposed.
    amplitude:
        Amount by which the spherical harmonic values modulate the radius.
    theta_resolution / phi_resolution:
        Number of samples along polar (θ) and azimuthal (φ) directions.
    normalization:
        Spherical harmonic normalization used by :func:`e3nn.o3.spherical_harmonics`.
    device, dtype:
        Optional overrides for the generated tensors. Defaults fall back to the
        current PyTorch settings.

    Returns
    -------
    SphericalHarmonicLobes
        Dataclass carrying the radial modulation, sampled values and Cartesian
        coordinates suitable for 3D plotting libraries such as Matplotlib.
    """

    if l < 0:
        raise ValueError("l must be non-negative.")
    if abs(m) > l:
        raise ValueError("|m| must be <= l.")
    if theta_resolution < 2 or phi_resolution < 3:
        raise ValueError("theta_resolution >= 2 and phi_resolution >= 3 are required.")

    dev = _default_device(device)
    dt = _default_dtype(dtype)

    theta = torch.linspace(0.0, math.pi, theta_resolution, device=dev, dtype=dt)
    phi = torch.linspace(0.0, 2.0 * math.pi, phi_resolution, device=dev, dtype=dt)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

    directions = torch.stack(
        (
            torch.sin(theta_grid) * torch.cos(phi_grid),
            torch.sin(theta_grid) * torch.sin(phi_grid),
            torch.cos(theta_grid),
        ),
        dim=-1,
    ).reshape(-1, 3)

    harmonics = o3.spherical_harmonics(
        [l],
        directions,
        normalize=True,
        normalization=normalization,
    )
    values = harmonics[:, m + l].reshape(theta_grid.shape)

    radius = base_radius + amplitude * values
    x = radius * torch.sin(theta_grid) * torch.cos(phi_grid)
    y = radius * torch.sin(theta_grid) * torch.sin(phi_grid)
    z = radius * torch.cos(theta_grid)

    return SphericalHarmonicLobes(
        l=l,
        m=m,
        theta=theta_grid,
        phi=phi_grid,
        radius=radius,
        values=values,
        x=x,
        y=y,
        z=z,
    )
