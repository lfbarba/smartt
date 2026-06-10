"""Implicit neural field for SAXS tensor tomography (``SaxsNafField``).

A coordinate network that maps each voxel position to the ``C`` real
spherical-harmonic coefficients (even ℓ up to ``ell_max``) of that voxel's
reciprocal-space map.  Sampled on the full voxel grid it produces an
``(X, Y, Z, C)`` coefficient field that is fed, unchanged, through the existing
linear forward model (``build_mumott_projector`` + ``forward_quadrature``).

Design (see project memory ``project_saxs_naf_design``):

* **Isotropic coordinate normalisation** — cubic voxels preserved; the longest
  axis spans ``[0, 1]`` and shorter axes are centred sub-intervals, so each hash
  level is isotropic in real space.
* **Single linear head → C**, with ``c00`` passed through softplus
  (non-negative mean intensity) and every channel divided by a fixed per-ℓ
  scale so the 45 outputs have commensurate gradients.
* **Cold start** — final layer zero-initialised so the initial field is
  ``c00 = softplus(bias) ≈ data mean`` and all anisotropy zero.
* **Coarse-to-fine masks** — :meth:`forward` accepts a spatial ``level_weights``
  vector (faded into the encoder) and an angular ``ell_mask`` vector (faded onto
  the output channels).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hash_encoding import MultiResolutionHashEncoding


def _generate_lm_list(ell_max: int) -> List[Tuple[int, int]]:
    """(ℓ, m) pairs for even ℓ, ordered as the forward operator expects."""
    lm = []
    for ell in range(0, ell_max + 1, 2):
        for m in range(-ell, ell + 1):
            lm.append((ell, m))
    return lm


def _softplus_inv(y: float) -> float:
    """Inverse softplus: x such that softplus(x) = y (y > 0)."""
    y = max(float(y), 1e-6)
    # Numerically stable log(exp(y) - 1).
    return float(np.log(np.expm1(y))) if y < 20 else y


class SaxsNafField(nn.Module):
    """Coordinate MLP emitting per-voxel SH coefficients.

    Parameters
    ----------
    volume_shape : ``(X, Y, Z)`` voxel grid (taken from ``geometry.volume_shape``).
    ell_max : Maximum even SH degree (8 ⇒ 45 coefficients).
    n_levels, n_features_per_level, base_resolution, max_resolution, table_size :
        Hash-encoding hyper-parameters.  ``max_resolution`` defaults to
        ``max(volume_shape)`` (grid Nyquist) and ``table_size`` to
        ``max_resolution ** 3`` (dense at these sizes).
    hidden_dim, n_hidden_layers : MLP trunk.
    c00_init : Initial mean-intensity value for the ``c00`` channel (cold start
        bias).  If ``None`` the caller should set it from the data.
    per_l_scale_power : Per-ℓ output scale is ``1 / (ℓ + 1) ** power``.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        ell_max: int = 8,
        n_levels: int = 8,
        n_features_per_level: int = 4,
        base_resolution: int = 8,
        max_resolution: Optional[int] = None,
        table_size: Optional[int] = None,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        c00_init: float = 1.0,
        per_l_scale_power: float = 1.0,
    ):
        super().__init__()
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.ell_max = ell_max
        self.lm_list = _generate_lm_list(ell_max)
        self.num_coeffs = len(self.lm_list)

        if max_resolution is None:
            max_resolution = max(self.volume_shape)

        self.encoding = MultiResolutionHashEncoding(
            n_dims=3,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            table_size=table_size,
            include_input=True,
        )

        # MLP trunk.
        layers: List[nn.Module] = []
        in_dim = self.encoding.output_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.num_coeffs)

        # Cold start: zero final layer so raw output = 0 everywhere.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # c00 bias chosen so softplus(bias) = c00_init.
        self.head.bias.data[0] = _softplus_inv(c00_init)

        # Per-ℓ output scale (fixed buffer), shape (C,).
        ells = torch.tensor([l for l, _ in self.lm_list], dtype=torch.float32)
        per_l_scale = 1.0 / (ells + 1.0) ** per_l_scale_power
        self.register_buffer("per_l_scale", per_l_scale)

        # ℓ(ℓ+1) weights for the angular regulariser (Laplace-Beltrami).
        lap = torch.tensor(
            [l * (l + 1) for l, _ in self.lm_list], dtype=torch.float32
        )
        self.register_buffer("laplace_beltrami", lap)

        # Precompute normalised, isotropic grid coordinates in [0, 1].
        self.register_buffer("grid_coords", self._build_grid_coords())

    def _build_grid_coords(self) -> torch.Tensor:
        """``(X*Y*Z, 3)`` coords in ``[0, 1]``, isotropic (cubic voxels)."""
        X, Y, Z = self.volume_shape
        axes = [torch.arange(n, dtype=torch.float32) for n in (X, Y, Z)]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        idx = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (N, 3)
        # Centre each axis, divide by the SAME extent (longest axis), recentre to 0.5.
        centres = torch.tensor(
            [(n - 1) / 2.0 for n in self.volume_shape], dtype=torch.float32
        )
        extent = float(max(self.volume_shape) - 1) if max(self.volume_shape) > 1 else 1.0
        coords = (idx - centres) / extent + 0.5
        return coords

    def set_c00_init(self, value: float) -> None:
        """Reset the cold-start ``c00`` bias from a data-derived mean."""
        self.head.bias.data[0] = _softplus_inv(value)

    def forward(
        self,
        level_weights: Optional[torch.Tensor] = None,
        ell_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample the field on the full grid → ``(X, Y, Z, C)`` coefficients.

        Parameters
        ----------
        level_weights : optional ``(n_levels,)`` spatial-annealing weights.
        ell_mask : optional ``(C,)`` angular-annealing visibility in ``[0, 1]``.
        """
        feats = self.encoding(self.grid_coords, level_weights=level_weights)
        raw = self.head(self.trunk(feats))               # (N, C)
        scaled = raw * self.per_l_scale                  # commensurate channels

        # c00 ≥ 0 via softplus; higher orders linear.
        c00 = F.softplus(scaled[..., :1])
        coeffs = torch.cat([c00, scaled[..., 1:]], dim=-1)

        if ell_mask is not None:
            coeffs = coeffs * ell_mask

        return coeffs.reshape(*self.volume_shape, self.num_coeffs)

    def sh_regularization(self, coeffs: torch.Tensor) -> torch.Tensor:
        """ℓ(ℓ+1)-weighted energy of the coefficient field (scalar)."""
        return (coeffs**2 * self.laplace_beltrami).sum()

    def tv_regularization(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Squared total variation across all spatial axes and SH channels.

        Penalises finite differences along X, Y, Z summed over all C channels.
        Uses squared (Tikhonov-style) differences for everywhere-differentiability.
        """
        tv = (coeffs[1:] - coeffs[:-1]).pow(2).sum()   # X differences
        tv += (coeffs[:, 1:] - coeffs[:, :-1]).pow(2).sum()  # Y
        tv += (coeffs[:, :, 1:] - coeffs[:, :, :-1]).pow(2).sum()  # Z
        return tv
