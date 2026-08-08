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
    hidden_dim, n_hidden_layers : MLP trunk.  Defaults favour grid capacity over
        trunk capacity (``hidden_dim=64``, ``n_hidden_layers=2``): the trunk's
        parameters are shared by every voxel, so an oversized trunk relative to
        the grid tends to compress genuinely different per-voxel hash features
        into a handful of directions the trunk finds useful — see
        ``head_init_std`` below for why that matters.
    c00_init : Initial mean-intensity value for the ``c00`` channel (cold start
        bias).  If ``None`` the caller should set it from the data.
    per_l_scale_power : Per-ℓ output scale is ``1 / (ℓ + 1) ** power``.
    head_init_std : Std of the ``head.weight`` Gaussian init (``0.0`` reproduces
        the old zero-init).  Zero-init starts the ``C×hidden`` readout matrix at
        *exactly* rank 1 (every row's first gradient step is the same outer
        product with ``trunk_out``), which is a direct, structural cause of the
        "same RSM shape everywhere, only rescaled" failure mode: even with
        arbitrarily rich per-voxel encoder features, ``coeffs(x) = W·trunk_out(x)``
        collapses to ``a(x)·v`` for a shared shape vector ``v`` if ``W`` stays
        low-rank. A small random init keeps the cold start (init anisotropy is
        still ≈0, since ``std`` is small and ``trunk_out`` starts near the raw
        coordinate scale) while letting every output row receive an
        independent gradient direction from step 0.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        ell_max: int = 8,
        n_levels: int = 8,
        n_features_per_level: int = 8,
        base_resolution: int = 8,
        max_resolution: Optional[int] = None,
        table_size: Optional[int] = None,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        c00_init: float = 1.0,
        per_l_scale_power: float = 1.0,
        head_init_std: float = 1e-3,
        n_qshells: int = 1,
        q_n_levels: int = 6,
        q_n_features_per_level: int = 4,
        q_base_resolution: int = 4,
    ):
        super().__init__()
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.ell_max = ell_max
        self.lm_list = _generate_lm_list(ell_max)
        self.num_coeffs = len(self.lm_list)
        self.n_qshells = int(n_qshells)

        # Constructor arguments captured verbatim (max_resolution/table_size may
        # be None here; the same defaulting logic re-runs identically on reload),
        # so a saved checkpoint can rebuild an identical architecture.  See
        # :func:`smartt.saxs_naf.cache.save_model`.
        self._config = dict(
            volume_shape=self.volume_shape,
            ell_max=ell_max,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            table_size=table_size,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            c00_init=c00_init,
            per_l_scale_power=per_l_scale_power,
            head_init_std=head_init_std,
            n_qshells=self.n_qshells,
            q_n_levels=q_n_levels,
            q_n_features_per_level=q_n_features_per_level,
            q_base_resolution=q_base_resolution,
        )

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

        # Optional q-shell conditioning: a SEPARATE, small 1-D hash encoding
        # over the (normalised) q coordinate, concatenated to the spatial
        # encoding's features before the trunk. A separate encoder — rather
        # than folding q into the same isotropic (x,y,z) hash grid as a 4th
        # axis — avoids forcing q to share the same per-level resolution
        # schedule as space (physically meaningless: q has ~79 distinct
        # shells at most, nowhere near spatial resolutions of 60-140) and
        # keeps every code path for n_qshells=1 (the default, and every
        # dataset used before this feature existed) IDENTICAL to the
        # pre-existing model: self.q_encoding is None, so forward_coords below
        # takes the exact same branch it always did, with the exact same
        # trunk input dimension. See project memory ``project_frogbone_3drsm``.
        self.q_encoding = None
        if self.n_qshells > 1:
            self.q_encoding = MultiResolutionHashEncoding(
                n_dims=1,
                n_levels=q_n_levels,
                n_features_per_level=q_n_features_per_level,
                base_resolution=q_base_resolution,
                max_resolution=max(self.n_qshells, q_base_resolution),
                include_input=True,
            )

        # MLP trunk.
        layers: List[nn.Module] = []
        in_dim = self.encoding.output_dim
        if self.q_encoding is not None:
            in_dim += self.q_encoding.output_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.num_coeffs)

        # Cold start: near-zero raw output (small random weight, not exactly
        # zero — see head_init_std docstring for why zero-init is avoided).
        nn.init.normal_(self.head.weight, mean=0.0, std=head_init_std)
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

        # Global output rescale (buffer ⇒ persists through state_dict save/load).
        # Training against a normalised target (see ``normalize_target`` in
        # ``saxs_naf_reconstruction``) leaves this at 1.0 throughout; it is set
        # once, post-training, so the field natively emits physical-unit
        # coefficients for any later caller (super-resolution, checkpoint reload).
        self.register_buffer("output_scale", torch.tensor(1.0))

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

    def get_config(self) -> dict:
        """Constructor kwargs needed to rebuild an identical (untrained) field."""
        return dict(self._config)

    def set_c00_init(self, value: float) -> None:
        """Reset the cold-start ``c00`` bias from a data-derived mean."""
        self.head.bias.data[0] = _softplus_inv(value)

    def set_output_scale(self, value: float) -> None:
        """Set the global post-hoc output rescale (see ``output_scale``)."""
        self.output_scale.fill_(float(value))

    def forward_coords(
        self,
        coords: torch.Tensor,
        q_coord: Optional[torch.Tensor] = None,
        level_weights: Optional[torch.Tensor] = None,
        ell_mask: Optional[torch.Tensor] = None,
        q_level_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the field at arbitrary ``(..., 3)`` coords in ``[0, 1]``.

        Returns ``(..., C)`` SH coefficients (c₀₀ passed through softplus).  This
        is the coordinate-driven core shared by :meth:`forward` (native grid),
        :meth:`forward_at_q` (native grid × q-shells), and
        :meth:`sample_super_resolution` (denser grid).

        ``q_coord`` — ``(..., 1)`` normalised q-coordinate in ``[0, 1]``,
        broadcastable against ``coords``' leading dims. Required iff
        ``self.q_encoding is not None`` (i.e. ``n_qshells > 1``); ignored
        (indeed, the branch is skipped entirely) otherwise, which is exactly
        what makes the ``n_qshells=1`` path bit-identical to the pre-existing
        model.
        """
        feats = self.encoding(coords, level_weights=level_weights)
        if self.q_encoding is not None:
            if q_coord is None:
                raise ValueError(
                    "This field has n_qshells > 1 (q_encoding is present); "
                    "forward_coords requires q_coord."
                )
            q_feats = self.q_encoding(q_coord, level_weights=q_level_weights)
            feats = torch.cat([feats, q_feats], dim=-1)
        raw = self.head(self.trunk(feats))               # (..., C)
        scaled = raw * self.per_l_scale                  # commensurate channels

        # c00 ≥ 0 via softplus; higher orders linear.
        c00 = F.softplus(scaled[..., :1])
        coeffs = torch.cat([c00, scaled[..., 1:]], dim=-1)

        if ell_mask is not None:
            coeffs = coeffs * ell_mask
        return coeffs * self.output_scale

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

        Only valid when ``n_qshells == 1`` (no q_encoding) — with q
        conditioning there is no single "the" q value to sample the grid at;
        use :meth:`forward_at_q` instead.
        """
        if self.q_encoding is not None:
            raise RuntimeError(
                "This field has n_qshells > 1; use forward_at_q(q_norm, ...) "
                "instead of forward()."
            )
        coeffs = self.forward_coords(
            self.grid_coords, level_weights=level_weights, ell_mask=ell_mask
        )
        return coeffs.reshape(*self.volume_shape, self.num_coeffs)

    def forward_at_q(
        self,
        q_norm: torch.Tensor,
        level_weights: Optional[torch.Tensor] = None,
        ell_mask: Optional[torch.Tensor] = None,
        q_level_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample the full spatial grid at each of ``Q`` q-coordinates.

        Parameters
        ----------
        q_norm : ``(Q,)`` tensor, normalised q-coordinates in ``[0, 1]``
            (e.g. ``(log(q) - log(q_min)) / (log(q_max) - log(q_min))`` — q
            bins are log-spaced, see ``FrogboneDataContainer``).
        level_weights, ell_mask : as in :meth:`forward`.
        q_level_weights : optional ``(q_n_levels,)`` annealing weights for the
            q encoding (coarse-to-fine along q). ``None`` (default) uses the
            q encoding fully unlocked.

        Returns
        -------
        ``(Q, X, Y, Z, C)`` SH coefficients, one full spatial field per
        requested q-coordinate, evaluated in a single batched forward pass.
        """
        if self.q_encoding is None:
            raise RuntimeError("forward_at_q requires n_qshells > 1 (no q_encoding present).")
        Q = q_norm.shape[0]
        N = self.grid_coords.shape[0]
        coords_rep = self.grid_coords.unsqueeze(0).expand(Q, N, 3).reshape(Q * N, 3)
        q_rep = q_norm.to(self.grid_coords.device).view(Q, 1, 1).expand(Q, N, 1).reshape(Q * N, 1)
        coeffs = self.forward_coords(
            coords_rep, q_coord=q_rep,
            level_weights=level_weights, ell_mask=ell_mask, q_level_weights=q_level_weights,
        )
        return coeffs.reshape(Q, *self.volume_shape, self.num_coeffs)

    def super_resolution_coords(
        self, factor: float
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """``factor×`` denser grid coords over the SAME physical ``[0, 1]`` domain.

        The continuous domain is identical to the native grid built in
        :meth:`_build_grid_coords` (voxel-index range ``[0, n-1]`` per axis,
        centred and divided by the longest-axis extent).  We simply sample it on
        ``round(factor * n)`` points per axis instead of ``n``, so the corner
        samples coincide with the native grid corners.  Returns the flattened
        ``(M, 3)`` coords and the target ``(X', Y', Z')`` shape.
        """
        target = tuple(int(round(n * factor)) for n in self.volume_shape)
        centres = [(n - 1) / 2.0 for n in self.volume_shape]
        extent = float(max(self.volume_shape) - 1) if max(self.volume_shape) > 1 else 1.0
        axes = []
        for n, tn, c in zip(self.volume_shape, target, centres):
            pos = torch.linspace(0.0, float(n - 1), tn, dtype=torch.float32)
            axes.append((pos - c) / extent + 0.5)
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        return coords, target

    @torch.no_grad()
    def sample_super_resolution(
        self, factor: float = 2.0, chunk_size: int = 1 << 18
    ) -> torch.Tensor:
        """Query the trained field at ``factor×`` resolution → ``(X', Y', Z', C)``.

        Evaluates the full SH field (no annealing masks) on a denser grid.
        Coordinates are streamed in chunks of ``chunk_size`` points to bound peak
        memory (a 4× grid over a 66³ volume is ~18 M points).  Returned on CPU.
        """
        device = self.grid_coords.device
        coords, target = self.super_resolution_coords(factor)
        coords = coords.to(device)
        out = torch.empty(coords.shape[0], self.num_coeffs, dtype=torch.float32)
        for start in range(0, coords.shape[0], chunk_size):
            chunk = coords[start : start + chunk_size]
            out[start : start + chunk_size] = self.forward_coords(chunk).cpu()
        return out.reshape(*target, self.num_coeffs)

    def sh_regularization(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Mean ℓ(ℓ+1)-weighted per-voxel-per-channel energy of the field.

        Averaged (not summed) over voxels and channels so the raw magnitude is
        ~invariant to volume size and ``ell_max`` — a single ``reg_weight_sh``
        is then comparable across datasets instead of needing per-dataset
        recalibration (a plain ``.sum()`` scales linearly with voxel count,
        so the same weight meant wildly different things for e.g. a 60³ vs a
        141×111×141 volume).
        """
        return (coeffs**2 * self.laplace_beltrami).mean()

    def tv_regularization(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Mean squared total variation across all spatial axes and SH channels.

        Penalises finite differences along X, Y, Z, each averaged (not summed)
        over its own voxel/channel count — see ``sh_regularization`` for why
        averaging (rather than summing) is what makes ``reg_weight_tv``
        comparable across differently-sized volumes. Uses squared
        (Tikhonov-style) differences for everywhere-differentiability.
        Accepts either ``(X, Y, Z, C)`` (single q-shell) or ``(Q, X, Y, Z, C)``
        (multi-q, e.g. from :meth:`forward_at_q`) — the spatial axes are always
        the three right after any leading ``Q`` batch axis, never axis 0
        outright, so this dispatches on ``coeffs.ndim`` rather than assuming
        ``(X,Y,Z,C)`` unconditionally.
        """
        if coeffs.ndim == 4:
            spatial_axes = (0, 1, 2)
        elif coeffs.ndim == 5:
            spatial_axes = (1, 2, 3)
        else:
            raise ValueError(
                f"tv_regularization expects (X,Y,Z,C) or (Q,X,Y,Z,C), got shape {tuple(coeffs.shape)}"
            )
        tv = coeffs.new_zeros(())
        for d in spatial_axes:
            idx_a = [slice(None)] * coeffs.ndim
            idx_b = [slice(None)] * coeffs.ndim
            idx_a[d] = slice(1, None)
            idx_b[d] = slice(None, -1)
            tv = tv + (coeffs[tuple(idx_a)] - coeffs[tuple(idx_b)]).pow(2).mean()
        return tv

    def q_tv_regularization(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Squared total variation along the q axis (multi-q only).

        Penalises abrupt jumps between neighbouring q-shells' coefficient
        fields — a direct, explicit smoothness prior along q, complementing
        (not replacing) the implicit smoothness the q hash-grid encoding
        already provides via interpolation. Requires ``coeffs`` shaped
        ``(Q, X, Y, Z, C)`` with ``Q > 1`` (i.e. from :meth:`forward_at_q`
        called with more than one q-coordinate this step).
        """
        if coeffs.ndim != 5:
            raise ValueError(f"q_tv_regularization expects (Q,X,Y,Z,C), got shape {tuple(coeffs.shape)}")
        if coeffs.shape[0] < 2:
            return coeffs.new_zeros(())
        return (coeffs[1:] - coeffs[:-1]).pow(2).sum()
