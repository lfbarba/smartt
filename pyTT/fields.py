from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from e3nn import o3


@dataclass(frozen=True)
class GridSpec:
    """Specification of the structured collection of solid spheres."""

    dims: Tuple[int, int, int]
    radius: float = 1.0
    spacing: Optional[Union[float, Tuple[float, float, float]]] = None

    def __post_init__(self) -> None:
        for dim in self.dims:
            if dim <= 0:
                raise ValueError("Grid dimensions must be strictly positive.")

    @property
    def num_spheres(self) -> int:
        return math.prod(self.dims)

    @property
    def spacing_vector(self) -> torch.Tensor:
        if self.spacing is None:
            base = 2.0 * self.radius
            return torch.tensor((base, base, base), dtype=torch.get_default_dtype())
        if isinstance(self.spacing, (int, float)):
            return torch.tensor((self.spacing, self.spacing, self.spacing), dtype=torch.get_default_dtype())
        if len(self.spacing) != 3:
            raise ValueError("Spacing must be a float or a 3-tuple.")
        return torch.tensor(tuple(float(s) for s in self.spacing))


@dataclass(frozen=True)
class ShellHarmonicExpansion:
    """Spherical harmonic coefficients describing the field on shell radii."""

    data: torch.Tensor
    radii: torch.Tensor
    irreps: o3.Irreps


class SphericalBesselHarmonicField(nn.Module):
    """Representation of N×M×K solid spheres using a spherical Bessel–harmonic basis.

    Parameters
    ----------
    dims:
        Tuple defining the number of spheres along each axis (N, M, K).
    max_l:
        Maximum angular frequency ℓ considered in the spherical harmonics.
    num_radial:
        Number of radial basis functions per (ℓ, m) pair.
    radius:
        Radius of each sphere. Defaults to 1.0.
    learn_k:
        If ``True`` the radial wave numbers ``k_{ℓn}`` are learned.
    k_init:
        Optional initialisation for ``k_{ℓn}``. Can be a float, sequence of floats,
        or a sequence per ℓ. Defaults to equally spaced multiples of π.
    coeff_init_scale:
        Standard deviation for the initial coefficients ``c_{ℓmn}``.
    dtype/device:
        Optional dtype and device overrides for parameters.
    """

    def __init__(
        self,
        dims: Union[Tuple[int, int, int], Sequence[int]],
        max_l: int,
        num_radial: int,
        radius: float = 1.0,
        learn_k: bool = True,
        k_init: Optional[Union[float, Sequence[float], Sequence[Sequence[float]]]] = None,
        coeff_init_scale: float = 1e-2,
        spacing: Optional[Union[float, Tuple[float, float, float]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(dims) != 3:
            raise ValueError("dims must be a sequence of three integers (N, M, K).")
        if max_l < 0:
            raise ValueError("max_l must be non-negative.")
        if num_radial <= 0:
            raise ValueError("num_radial must be positive.")

        self._dtype = dtype if dtype is not None else torch.get_default_dtype()
        self._init_device = device if device is not None else torch.device("cpu")

        self.grid = GridSpec(tuple(int(d) for d in dims), radius=radius, spacing=spacing)
        self.max_l = int(max_l)
        self.num_radial = int(num_radial)
        self.learn_k = learn_k
        self.k_epsilon = 1e-3

        self._l_values = list(range(self.max_l + 1))
        self._total_m = sum(2 * l + 1 for l in self._l_values)
        self._m_slices = self._build_m_slices()

        spacing_vec = self.grid.spacing_vector.to(dtype=self._dtype, device=self._init_device)
        self.register_buffer(
            "centers",
            self._compute_centers(self.grid.dims, spacing_vec, self._init_device, self._dtype),
            persistent=False,
        )

        coeffs: list[nn.Parameter] = []
        self.log_k_params: Optional[nn.ParameterList] = None

        prepared_k = self._prepare_k_init(k_init)

        for l in self._l_values:
            coeff_tensor = torch.randn(
                self.grid.num_spheres,
                self.num_radial,
                2 * l + 1,
                device=self._init_device,
                dtype=self._dtype,
            ) * coeff_init_scale
            coeffs.append(nn.Parameter(coeff_tensor))

            k_tensor = prepared_k[l]
            if learn_k:
                if self.log_k_params is None:
                    self.log_k_params = nn.ParameterList()
                log_k = torch.log(torch.clamp(k_tensor, min=self.k_epsilon))
                self.log_k_params.append(nn.Parameter(log_k))
            else:
                self.register_buffer(f"k_values_{l}", k_tensor, persistent=False)

        self.coeffs = nn.ParameterList(coeffs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def dims(self) -> Tuple[int, int, int]:
        return self.grid.dims

    @property
    def radius(self) -> float:
        return float(self.grid.radius)

    @property
    def num_spheres(self) -> int:
        return self.grid.num_spheres

    @property
    def device(self) -> torch.device:
        return self.coeffs[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.coeffs[0].dtype

    def forward(
        self,
        relative_points: torch.Tensor,
        selection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the field at the given relative coordinates.

        Parameters
        ----------
        relative_points:
            Tensor of shape ``(S, ..., 3)`` holding coordinates relative to the
            centres of the selected spheres. ``S`` must equal the number of
            selected spheres or the total number of spheres if ``selection`` is ``None``.
        selection:
            Optional boolean mask or integer index specifying which spheres are
            represented by ``relative_points``.

        Returns
        -------
        torch.Tensor
            Field values with shape ``(S, ...)``.
        """

        rel_pts = relative_points.to(device=self.device, dtype=self.dtype)
        indices = self._selection_indices(selection)
        if indices is None:
            if rel_pts.shape[0] != self.num_spheres:
                raise ValueError(
                    "relative_points must provide coordinates for all spheres when selection is None.",
                )
            coeffs = self._select_coeffs(None)
        else:
            if rel_pts.shape[0] != indices.numel():
                raise ValueError(
                    "Number of coordinate sets must match the selection size.",
                )
            coeffs = [param[indices] for param in self.coeffs]

        r, directions = self._split_radius_and_direction(rel_pts)
        values = self._evaluate(r, directions, coeffs)
        return values

    @torch.no_grad()
    def slice(
        self,
        normal_vector: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        grid_resolution: int = 128,
        extent: Optional[float] = None,
        flatten: bool = False,
        return_frame: bool = False,
    ) -> torch.Tensor:
        """Return a plane slice orthogonal to ``normal_vector`` for all spheres.

        Parameters
        ----------
        normal_vector:
            3-vector defining the plane's normal. The vector does not need to be
            normalised.
        mask:
            Optional boolean mask of shape ``(N, M, K)`` indicating which spheres
            to evaluate. Others receive zeros.
        grid_resolution:
            Number of samples per axis in the returned square grid.
        extent:
            Half-width of the sampling plane. Defaults to the sphere radius.
        flatten:
            When ``True`` return only the active spheres stacked along the first
            dimension with shape ``(K, grid_resolution, grid_resolution)``, where
            ``K`` is the number of selected spheres. When ``False`` the output is
            reshaped back to ``(N, M, K, grid_resolution, grid_resolution)``.
        return_frame:
            When ``True``, also return the in-plane orthonormal basis vectors
            ``(u, v, n)`` used for slicing. The return type becomes a tuple
            ``(slice_tensor, (u, v, n))``. Defaults to ``False``.
        """

        if grid_resolution <= 1:
            raise ValueError("grid_resolution must be greater than one.")

        device = self.device
        dtype = self.dtype

        # Build a continuous in-plane frame by rotating canonical axes so that
        # the xy-plane aligns with the requested slicing plane (normal n_vec).
        # This avoids branch-induced flips from heuristic fallbacks.
        n_vec = self._safe_normalise(normal_vector.to(device=device, dtype=dtype))
        u_vec, v_vec = self._plane_basis_smooth(n_vec)

        if extent is None:
            extent = self.radius

        lin = torch.linspace(-extent, extent, grid_resolution, device=device, dtype=dtype)
        uu, vv = torch.meshgrid(lin, lin, indexing="ij")
        plane = uu[..., None] * u_vec + vv[..., None] * v_vec

        r, directions = self._split_radius_and_direction(plane.unsqueeze(0))

        selection_indices = self._selection_indices(mask)
        if selection_indices is None:
            coeffs = list(self.coeffs)
        else:
            coeffs = [param[selection_indices] for param in self.coeffs]

        active_count = coeffs[0].shape[0]
        if active_count == 0:
            if flatten:
                out = torch.zeros(
                    (0, grid_resolution, grid_resolution),
                    device=device,
                    dtype=dtype,
                )
                return (out, (u_vec, v_vec, n_vec)) if return_frame else out
            out = torch.zeros(
                (*self.dims, grid_resolution, grid_resolution),
                device=device,
                dtype=dtype,
            )
            return (out, (u_vec, v_vec, n_vec)) if return_frame else out

        values = self._evaluate(
            r.expand(active_count, -1, -1),
            directions.expand(active_count, -1, -1, -1),
            coeffs,
        )

        inside = r <= self.radius
        values = values.masked_fill(~inside.expand_as(values), 0.0)

        if flatten:
            return (values, (u_vec, v_vec, n_vec)) if return_frame else values

        full = torch.zeros(
            (self.num_spheres, grid_resolution, grid_resolution),
            device=device,
            dtype=dtype,
        )

        if selection_indices is None:
            full = values
        else:
            full[selection_indices] = values

        full = full.reshape(*self.dims, grid_resolution, grid_resolution)
        return (full, (u_vec, v_vec, n_vec)) if return_frame else full

    @torch.no_grad()
    def shell_harmonics(
        self,
        radii: Union[float, Sequence[float], torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        flatten: bool = False,
    ) -> ShellHarmonicExpansion:
        """Compute spherical harmonic coefficients on spherical shells.

        Parameters
        ----------
        radii:
            Radius or sequence of radii at which to evaluate the shell.
        mask:
            Optional boolean mask or index selection for spheres, analogous to
            :meth:`slice`.
        flatten:
            When ``True`` return only the active spheres stacked along the first
            dimension with shape ``(K, R, dim)``. Otherwise reshape back to the
            full grid ``(N, M, K, R, dim)``.

        Returns
        -------
        ShellHarmonicExpansion
            Dataclass containing the irreducible representation descriptor and
            harmonic coefficients of shape ``(..., R, dim)`` where ``dim`` is the
            total number of spherical harmonic components up to ``max_l``.
        """

        device = self.device
        dtype = self.dtype

        if isinstance(radii, torch.Tensor):
            shell_radii = radii.to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(radii, (list, tuple)):
            if len(radii) == 0:
                raise ValueError("radii sequence must not be empty.")
            shell_radii = torch.tensor(list(radii), device=device, dtype=dtype)
        else:
            shell_radii = torch.tensor([float(radii)], device=device, dtype=dtype)

        if shell_radii.numel() == 0:
            raise ValueError("At least one radius must be provided.")
        if torch.any(shell_radii < 0):
            raise ValueError("Radii must be non-negative.")

        selection_indices = self._selection_indices(mask)
        if selection_indices is None:
            coeffs = list(self.coeffs)
        else:
            coeffs = [param[selection_indices] for param in self.coeffs]

        active_count = coeffs[0].shape[0]
        irreps = o3.Irreps([(1, (l, (-1) ** l)) for l in self._l_values])
        components = irreps.dim
        num_radii = shell_radii.numel()

        if active_count == 0:
            empty_shape = (0, num_radii, components) if flatten else (*self.dims, num_radii, components)
            empty = torch.zeros(empty_shape, device=device, dtype=dtype)
            return ShellHarmonicExpansion(data=empty, radii=shell_radii, irreps=irreps)

        shell_values = torch.zeros((active_count, num_radii, components), device=device, dtype=dtype)

        offset = 0
        for idx, l in enumerate(self._l_values):
            dim = 2 * l + 1
            k_vals = self._k_values_for_l(idx)
            jl = self._spherical_jn(l, shell_radii[:, None] * k_vals)
            # Combine radial basis contributions for each sphere and radius.
            combined = torch.einsum("rn,snd->srd", jl, coeffs[idx])
            shell_values[:, :, offset:offset + dim] = combined
            offset += dim

        if flatten:
            data = shell_values
        else:
            full = torch.zeros((self.num_spheres, num_radii, components), device=device, dtype=dtype)
            if selection_indices is None:
                full = shell_values
            else:
                full[selection_indices] = shell_values
            data = full.reshape(*self.dims, num_radii, components)

        return ShellHarmonicExpansion(data=data, radii=shell_radii, irreps=irreps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_m_slices(self) -> list[Tuple[int, int]]:
        slices: list[Tuple[int, int]] = []
        start = 0
        for l in self._l_values:
            dim = 2 * l + 1
            slices.append((start, start + dim))
            start += dim
        return slices

    def _prepare_k_init(
        self,
        k_init: Optional[Union[float, Sequence[float], Sequence[Sequence[float]]]],
    ) -> list[torch.Tensor]:
        prepared: list[torch.Tensor] = []
        if k_init is None:
            base = torch.arange(1, self.num_radial + 1, dtype=self._dtype, device=self._init_device) * math.pi
            for _ in self._l_values:
                prepared.append(base.clone())
            return prepared

        if isinstance(k_init, (int, float)):
            base = torch.full((self.num_radial,), float(k_init), dtype=self._dtype, device=self._init_device)
            for _ in self._l_values:
                prepared.append(base.clone())
            return prepared

        if len(k_init) == self.num_radial and not isinstance(k_init[0], Sequence):
            base = torch.tensor(tuple(float(v) for v in k_init), dtype=self._dtype, device=self._init_device)
            for _ in self._l_values:
                prepared.append(base.clone())
            return prepared

        if len(k_init) != self.max_l + 1:
            raise ValueError("k_init must provide values for each ℓ or be broadcastable.")

        for l, values in enumerate(k_init):
            if len(values) != self.num_radial:
                raise ValueError(f"k_init for ℓ={l} must have length {self.num_radial}.")
            prepared.append(torch.tensor(tuple(float(v) for v in values), dtype=self._dtype, device=self._init_device))
        return prepared

    def _select_coeffs(
        self,
        selection: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        if selection is None:
            return [param for param in self.coeffs]

        sel_mask = self._selection_mask(selection)
        if sel_mask is None:
            return [param for param in self.coeffs]
        if sel_mask.ndim != 1:
            raise ValueError("Selection mask must be one-dimensional when provided.")
        return [param[sel_mask] for param in self.coeffs]

    def _selection_mask(self, selection: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if selection is None:
            return None
        if selection.dtype == torch.bool:
            flat = selection.reshape(-1)
            if flat.numel() != self.num_spheres:
                raise ValueError("Boolean selection mask has incorrect size.")
            return flat
        indices = selection.reshape(-1).to(torch.int64)
        mask = torch.zeros(self.num_spheres, dtype=torch.bool, device=self.device)
        mask[indices] = True
        return mask

    def _selection_indices(self, selection: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if selection is None:
            return None
        if selection.dtype == torch.bool:
            flat = selection.reshape(-1)
            if flat.numel() != self.num_spheres:
                raise ValueError("Boolean selection mask has incorrect size.")
            return flat.nonzero(as_tuple=False).squeeze(-1)
        return selection.reshape(-1).to(torch.int64)

    def _compute_centers(
        self,
        dims: Tuple[int, int, int],
        spacing: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        grid_ranges = [torch.arange(d, device=device, dtype=dtype) for d in dims]
        zz, yy, xx = torch.meshgrid(*grid_ranges, indexing="ij")
        offsets = torch.stack((xx, yy, zz), dim=-1)
        centred = offsets * spacing.to(dtype)
        return centred.reshape(-1, 3)

    def _safe_normalise(self, vector: torch.Tensor) -> torch.Tensor:
        norm = vector.norm(dim=-1, keepdim=True)
        safe_value = torch.tensor(1.0, device=vector.device, dtype=vector.dtype)
        safe = torch.where(norm <= 0, safe_value, norm)
        return vector / safe

    def _split_radius_and_direction(
        self,
        vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r = vectors.norm(dim=-1)
        zero = (r == 0)
        directions = torch.zeros_like(vectors)
        directions[~zero] = vectors[~zero] / r[~zero].unsqueeze(-1)
        if torch.any(zero):
            directions[zero] = torch.tensor([0.0, 0.0, 1.0], device=vectors.device, dtype=vectors.dtype)
        return r, directions

    def _plane_basis(self, normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        abs_n = normal.abs()
        smallest = torch.argmin(abs_n)
        basis = torch.zeros_like(normal)
        basis[smallest] = 1.0
        u = F.normalize(torch.cross(normal, basis, dim=0), dim=0)
        v = F.normalize(torch.cross(normal, u, dim=0), dim=0)
        return u, v

    def _plane_basis_smooth(self, normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct a smooth in-plane orthonormal basis (u, v) for a given unit normal.

        We compute a rotation R that maps e_z to ``normal`` using a Rodrigues
        formulation. Then set u = R e_x, v = R e_y. This yields a continuous
        frame for all normals except exactly at normal ≈ -e_z, where any 180°
        rotation about an in-plane axis is valid; we choose a fixed rotation
        about the x-axis to keep behaviour deterministic.
        """
        device = normal.device
        dtype = normal.dtype
        ez = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        ey = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

        # Ensure unit normal (caller guarantees, but be safe)
        n = self._safe_normalise(normal)
        c = torch.dot(ez, n)  # cos(theta)

        # Thresholds for degeneracy handling
        eps = torch.tensor(1e-7, device=device, dtype=dtype)

        if c >= 1.0 - eps:  # n ~ +ez -> identity rotation
            u = ex
            v = ey
            return u, v
        if c <= -1.0 + eps:  # n ~ -ez -> rotate 180 deg about x-axis
            Rx_pi = torch.tensor(
                [[1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0, 0.0, -1.0]],
                device=device,
                dtype=dtype,
            )
            u = Rx_pi @ ex
            v = Rx_pi @ ey
            return u, v

        k = torch.cross(ez, n)  # rotation axis (unnormalized)
        s = torch.norm(k)       # sin(theta)
        k = k / s               # normalize axis

        # Skew-symmetric matrix of k
        kx, ky, kz = k[0], k[1], k[2]
        K = torch.stack(
            (
                torch.stack((torch.tensor(0.0, device=device, dtype=dtype), -kz, ky)),
                torch.stack((kz, torch.tensor(0.0, device=device, dtype=dtype), -kx)),
                torch.stack((-ky, kx, torch.tensor(0.0, device=device, dtype=dtype))),
            )
        )
        I = torch.eye(3, device=device, dtype=dtype)
        # Use the identity: R = I + sin(theta) K + (1 - cos(theta)) K^2
        R = I + s * K + (1.0 - c) * (K @ K)

        u = R @ ex
        v = R @ ey
        # Final orthonormalization (guard tiny numerical drift)
        u = F.normalize(u, dim=0)
        v = F.normalize(torch.cross(n, u, dim=0), dim=0)
        return u, v

    def _evaluate(
        self,
        radii: torch.Tensor,
        directions: torch.Tensor,
        coeffs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        S = coeffs[0].shape[0]
        spatial_shape = radii.shape[1:]
        radii_flat = radii.reshape(S, -1)
        directions_flat = directions.reshape(S, -1, 3)

        harmonics = o3.spherical_harmonics(
            self._l_values,
            directions_flat,
            normalize=True,
            normalization="component",
        )

        result = torch.zeros(S, radii_flat.shape[1], device=radii.device, dtype=radii.dtype)
        offset = 0
        for idx, l in enumerate(self._l_values):
            dim = 2 * l + 1
            slice_start, slice_end = offset, offset + dim
            offset += dim

            k_values = self._k_values_for_l(idx)
            jl = self._spherical_jn(l, radii_flat[..., None] * k_values)
            coeff = coeffs[idx]
            radial = torch.matmul(jl, coeff)
            Y_l = harmonics[..., slice_start:slice_end]
            result += (radial * Y_l).sum(dim=-1)

        return result.reshape((S, *spatial_shape))

    def _k_values_for_l(self, idx: int) -> torch.Tensor:
        if self.learn_k:
            assert self.log_k_params is not None
            return F.softplus(self.log_k_params[idx]) + self.k_epsilon
        buffer = getattr(self, f"k_values_{self._l_values[idx]}")
        return buffer

    def _spherical_jn(self, order: int, x: torch.Tensor) -> torch.Tensor:
        safe_x = torch.where(x == 0, torch.ones_like(x), x)
        sin_term = torch.sin(safe_x)
        cos_term = torch.cos(safe_x)

        j0 = torch.where(x == 0, torch.ones_like(x), sin_term / safe_x)
        if order == 0:
            return j0

        j1 = torch.where(
            x == 0,
            torch.zeros_like(x),
            sin_term / (safe_x * safe_x) - cos_term / safe_x,
        )
        if order == 1:
            return j1

        j_prev, j_curr = j0, j1
        for n in range(1, order):
            coef = (2 * n + 1.0) * torch.reciprocal(safe_x)
            j_next = coef * j_curr - j_prev
            j_next = torch.where(x == 0, torch.zeros_like(j_next), j_next)
            j_prev, j_curr = j_curr, j_next
        return j_curr

