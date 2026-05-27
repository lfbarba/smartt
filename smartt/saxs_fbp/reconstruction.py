"""SAXS tensor tomography FBP reconstruction.

For each target q-direction y on the upper hemisphere, builds an independent scalar
CT sub-problem and solves it with FBP. The result F[i, x, y, z] gives the value of
the reciprocal-space sphere function at direction y_directions[i] for each voxel.

The decomposition into sub-problems is done by NearestNeighbor from mumott:
  projection_matrix[n, m, k] = fraction of segment m in projection n whose arc
  lies within the Voronoi cell of direction k on the q-sphere.

saxs_fbp_reconstruction builds all K sinograms in one GPU einsum and pre-computes
the ASTRA geometry once, so the GPU stays loaded across the K backprojections.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from mumott import DataContainer

from smartt.projectors.astra_projector import (
    backproject,
    _create_astra_geometries_from_mumott,
    _backproject_single_gpu,
    _backproject_single_cpu,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sphere grid
# ---------------------------------------------------------------------------

def fibonacci_hemisphere(k: int, pole_gap_deg: float = 0.0,
                         half_space: str = 'z') -> np.ndarray:
    """k evenly distributed unit vectors on a hemisphere.

    Uses the Fibonacci / golden-ratio spiral mapping to distribute points
    with roughly uniform solid-angle coverage.

    Parameters
    ----------
    k :
        Number of directions.
    pole_gap_deg :
        Angular gap to keep around the pole axis (the beam axis), in degrees.
        Directions within ``pole_gap_deg`` of the pole are excluded.
        Default 0 = full hemisphere.

        Pass ``alpha_max_deg`` (the goniometer outer-tilt limit) to exclude
        directions that have no accessible projections: for y_k within
        alpha_max of the beam axis there is no beam both ⊥ y_k and within
        the accessible polar cap, so the sub-CT has zero data.
    half_space : str
        ``'z'`` (default): sample z > 0 hemisphere (pole at [0, 0, 1]).
        ``'y'``: rotate the grid so directions have y > 0 (pole at [0, 1, 0]),
        matching the SICI/goniometer visualisation convention where the xz-plane
        is the equatorial reference plane.

    Returns
    -------
    directions : (k, 3) float64 array
        Unit vectors on the requested hemisphere.
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    z_max = math.cos(math.radians(pole_gap_deg))
    i = np.arange(k, dtype=np.float64)
    phi = 2.0 * math.pi * i / golden
    z = (i + 0.5) / k * z_max
    r = np.sqrt(np.maximum(0.0, 1.0 - z ** 2))
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)
    if half_space == 'y':
        # Rx(-90°): (x, y, z) → (x, z, -y)  — maps z-pole [0,0,1] → y-pole [0,1,0]
        dirs = np.column_stack([dirs[:, 0], dirs[:, 2], -dirs[:, 1]])
    return dirs


# ---------------------------------------------------------------------------
# Ramp filter
# ---------------------------------------------------------------------------

def _ramp_filter(
    sino_kij: torch.Tensor,
    filter_type: str = 'hann',
    det_spacing: float = 1.0,
) -> torch.Tensor:
    """Ramp filter applied along the last (detector-column) axis.

    Parameters
    ----------
    sino_kij : (K, I, J) float32 tensor
        Sinogram in ASTRA layout (det_rows, n_views, det_cols). Filtering is
        along det_cols (axis -1), which is the transaxial direction.
    filter_type :
        One of ``'ram-lak'``, ``'shepp-logan'``, ``'cosine'``,
        ``'hamming'``, ``'hann'``.
    det_spacing :
        Detector pixel pitch (same units as voxel size).

    Returns
    -------
    filtered : (K, I, J) float32 tensor
    """
    _J = sino_kij.shape[-1]
    device = sino_kij.device

    # Zero-pad to next power of two (avoids circular wrap-around artefacts).
    padded_J = int(2 ** math.ceil(math.log2(max(64, 2 * _J))))
    pad = padded_J - _J

    sino_padded = torch.nn.functional.pad(sino_kij.float(), (0, pad), mode='replicate')
    sino_fft = torch.fft.rfft(sino_padded, dim=-1)

    freq = torch.fft.rfftfreq(padded_J, d=det_spacing, device=device)
    freq_abs = freq.abs()
    freq_abs[0] = 0.0

    freq_max = float(freq_abs.max()) if freq_abs.numel() > 1 else 1.0
    if freq_max == 0.0:
        freq_max = 1.0
    freq_norm = freq_abs / freq_max

    ftype = filter_type.lower()
    if ftype == 'ram-lak':
        window = torch.ones_like(freq_abs)
    elif ftype == 'shepp-logan':
        window = torch.ones_like(freq_abs)
        window[1:] = torch.sinc(freq_norm[1:])
    elif ftype == 'cosine':
        window = torch.cos((math.pi / 2.0) * freq_norm)
    elif ftype == 'hamming':
        window = 0.54 + 0.46 * torch.cos(math.pi * freq_norm)
    elif ftype == 'hann':
        window = 0.5 * (1.0 + torch.cos(math.pi * freq_norm))
    else:
        raise ValueError(
            f"Unknown filter_type {filter_type!r}. "
            "Choose from: 'ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann'."
        )

    ramp = (2.0 * freq_abs * window).view(1, 1, -1)
    filtered = torch.fft.irfft(sino_fft * ramp, n=padded_J, dim=-1)[..., :_J]
    return (filtered * det_spacing).to(sino_kij.dtype)


# ---------------------------------------------------------------------------
# Single-direction FBP
# ---------------------------------------------------------------------------

def fbp_with_mumott_geometry(
    sino: torch.Tensor,
    sub_geometry,
    filter_type: str = 'hann',
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """FBP reconstruction of a scalar volume from a mumott sub-geometry.

    Parameters
    ----------
    sino : (N_sub, J, K_det) float32 tensor
        Scalar sinogram. Invalid pixels must already be zeroed.
    sub_geometry :
        mumott ``Geometry`` with N_sub projections — the output of
        ``NearestNeighbor.get_sub_geometry``.
    filter_type :
        Ramp-filter window. Passed to :func:`_ramp_filter`.
    device :
        Target device. Defaults to ``sino.device``.

    Returns
    -------
    vol : (X, Y, Z) float32 tensor
        Reconstructed scalar field.
    """
    if device is None:
        device = sino.device if isinstance(sino, torch.Tensor) else torch.device('cpu')

    sino = sino.to(device).float()
    N_sub = sino.shape[0]

    # sino layout is (N_sub, J, K_det) = (I, J, K) in astra_projector convention.
    # For filtering we permute to ASTRA (K_det, N_sub, J), filter along J, then
    # permute back so backproject() receives the expected (I, J, K) layout.
    sino_kij = sino.permute(2, 0, 1).contiguous()           # (K_det, N_sub, J)
    filtered_kij = _ramp_filter(sino_kij, filter_type)
    filtered_ijk = filtered_kij.permute(1, 2, 0).contiguous()  # (N_sub, J, K_det)

    vol = backproject(filtered_ijk, sub_geometry, device=device)  # (X, Y, Z)

    if N_sub > 0:
        vol = vol * (math.pi / (2.0 * N_sub))

    return vol


# ---------------------------------------------------------------------------
# GPU-side helpers
# ---------------------------------------------------------------------------

def _fbp_with_precomputed_geom(
    sino: torch.Tensor,
    vol_geom: dict,
    proj_geom: dict,
    vol_shape: tuple,
    filter_type: str,
    device: torch.device,
) -> torch.Tensor:
    """Ramp-filter + backproject using already-built ASTRA geometry objects.

    Parameters
    ----------
    sino : (N, J, K_det) float32 tensor
        Full sinogram (non-contributing rows must already be zeroed).
    vol_geom, proj_geom :
        Pre-computed ASTRA geometry dicts (built once from the full geometry).
    vol_shape : (X, Y, Z) tuple
    filter_type : str
    device : torch.device

    Returns
    -------
    vol : (X, Y, Z) float32 tensor
    """
    sino = sino.to(device).float()
    # Permute to ASTRA (K_det, N, J), filter along J (det_cols), permute back.
    sino_kij = sino.permute(2, 0, 1).contiguous()
    filtered_kij = _ramp_filter(sino_kij, filter_type)
    filtered_ijk = filtered_kij.permute(1, 2, 0).contiguous()  # (N, J, K_det)

    if device.type == 'cuda':
        return _backproject_single_gpu(filtered_ijk, vol_geom, proj_geom, vol_shape, device)
    return _backproject_single_cpu(filtered_ijk, vol_geom, proj_geom, vol_shape, device)


# ---------------------------------------------------------------------------
# GPU projection matrix (replaces mumott NearestNeighbor CPU integration) 
# ---------------------------------------------------------------------------

def _build_projection_matrix_gpu(
    probed_coordinates,
    y_directions: np.ndarray,
    enforce_friedel_symmetry: bool = True,
    n_samples: int = 64,
    device: torch.device = None,
    method: str = 'voronoi',
    threshold: float = 0.3,
) -> torch.Tensor:
    """Build the projection matrix on the GPU via SLERP arc sampling.

    Replaces mumott's CPU adaptive-Simpson arc integration with fixed SLERP
    arc sampling and a vectorized dot-product lookup.

    For n_samples >= 32 the ``'voronoi'`` result is numerically equivalent to
    the CPU NearestNeighbor with integration_mode='simpson',
    integration_tolerance=1e-3.

    Parameters
    ----------
    method : str
        ``'voronoi'`` (default): each arc sample votes for its single nearest
        direction (Voronoi partition); rows of pm sum to 1.
        ``'ball'``: each arc sample votes for every direction within
        ``threshold`` radians; rows of pm can sum to > 1, which widens
        contributions and alleviates the missing-wedge problem.
    threshold : float
        Angular radius in radians for the ``'ball'`` method. Ignored for
        ``'voronoi'``. With Friedel symmetry a sample contributes to
        direction k when ``|dot(sample, d_k)| >= cos(threshold)``.

    Returns
    -------
    pm : (N, M, K) float32 tensor on *device*
        projection_matrix[n, m, k] = fraction of arc samples of segment m in
        projection n that fall in the Voronoi cell (``'voronoi'``) or ball
        (``'ball'``) of y_directions[k].
    """
    if method not in ('voronoi', 'ball'):
        raise ValueError(f"Unknown method {method!r}. Choose 'voronoi' or 'ball'.")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vec = probed_coordinates.vector                        # (N, M, I, 3)
    off_arr = probed_coordinates.great_circle_offset       # broadcastable to (N, M, I, 3)
    N, M = vec.shape[0], vec.shape[1]
    K = y_directions.shape[0]

    # Arc endpoints (normalised); off corrects for small-circle arcs
    start = vec[:, :, 0, :].copy()    # (N, M, 3)
    end   = vec[:, :, -1, :].copy()   # (N, M, 3)
    start /= np.linalg.norm(start, axis=-1, keepdims=True).clip(1e-10)
    end   /= np.linalg.norm(end,   axis=-1, keepdims=True).clip(1e-10)
    off_np = np.broadcast_to(off_arr[..., 0, :], (N, M, 3)).copy()

    cs = torch.as_tensor(start - off_np, dtype=torch.float32).to(device)   # (N, M, 3)
    ce = torch.as_tensor(end   - off_np, dtype=torch.float32).to(device)   # (N, M, 3)
    of = torch.as_tensor(off_np,         dtype=torch.float32).to(device)   # (N, M, 3)

    dirs = torch.as_tensor(y_directions, dtype=torch.float32, device=device)  # (K, 3)
    dirs_full = torch.cat([dirs, -dirs], dim=0) if enforce_friedel_symmetry else dirs

    # Subtended arc angle and SLERP weights
    omega     = (cs * ce).sum(-1).clamp(-1.0, 1.0).acos()   # (N, M)
    sin_omega = omega.sin()                                  # (N, M)
    degen     = sin_omega.abs() < 1e-10                      # (N, M)

    t   = torch.linspace(0.0, 1.0, n_samples, device=device).view(1, 1, -1)  # (1, 1, S)
    om  = omega.unsqueeze(-1)      # (N, M, 1)
    so  = sin_omega.unsqueeze(-1)  # (N, M, 1)
    deg = degen.unsqueeze(-1)      # (N, M, 1)

    a = torch.where(deg, 1.0 - t, (1.0 - t).mul(om).sin() / so.clamp(1e-10))  # (N, M, S)
    b = torch.where(deg, t,        t.mul(om).sin()         / so.clamp(1e-10))  # (N, M, S)

    # Arc sample points on unit sphere: (N, M, S, 3)
    arc = (a.unsqueeze(-1) * cs.unsqueeze(2)
         + b.unsqueeze(-1) * ce.unsqueeze(2)
         + of.unsqueeze(2))
    arc = arc / arc.norm(dim=-1, keepdim=True).clamp(1e-10)

    # Chunk over projections so the intermediate (C*M*S, n_dirs) tensor stays
    # under ~512 MB regardless of K.
    n_dirs = dirs_full.shape[0]
    chunk = max(1, min(N, int(512 * 1024 ** 2 // (M * n_samples * n_dirs * 4))))
    pm = torch.zeros(N, M, K, dtype=torch.float32, device=device)

    if method == 'voronoi':
        nn_idx = torch.empty(N, M, n_samples, dtype=torch.long, device=device)
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            flat = arc[i:j].reshape((j - i) * M * n_samples, 3)  # (C*M*S, 3)
            nn_idx[i:j] = (flat @ dirs_full.T).argmax(-1).view(j - i, M, n_samples)

        if enforce_friedel_symmetry:
            nn_idx = nn_idx % K

        pm.scatter_add_(2, nn_idx, torch.ones(N, M, n_samples, dtype=torch.float32, device=device))
        pm /= n_samples

    else:  # method == 'ball'
        cos_thresh = math.cos(threshold)
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            flat = arc[i:j].reshape((j - i) * M * n_samples, 3)  # (C*M*S, 3)
            dots = flat @ dirs_full.T                             # (C*M*S, n_dirs_full)
            if enforce_friedel_symmetry:
                # Sample is within threshold of direction k iff |dot(sample, d_k)| >= cos(threshold)
                in_ball = dots[:, :K].abs() >= cos_thresh         # (C*M*S, K)
            else:
                in_ball = dots >= cos_thresh                      # (C*M*S, K)
            pm[i:j] += in_ball.float().view(j - i, M, n_samples, K).sum(dim=2) / n_samples

    return pm


# ---------------------------------------------------------------------------
# Drop-in replacement for mumott NearestNeighbor (visualization helper)
# ---------------------------------------------------------------------------

class FBPProjectionMatrix:
    """Wraps the GPU-computed projection matrix for post-reconstruction use.

    Provides the same interface as ``mumott.NearestNeighbor`` that the
    visualization notebooks rely on, built from the matrix already computed
    during :func:`saxs_fbp_reconstruction` — no re-integration needed.
    """

    def __init__(self, pm: np.ndarray, y_directions: np.ndarray):
        self._pm = pm                      # (N, M, K) float32 numpy
        self._y_directions = y_directions  # (K, 3) float64 numpy

    @property
    def projection_matrix(self) -> np.ndarray:
        """(N, M, K) arc-fraction weights, same as NearestNeighbor.projection_matrix."""
        return self._pm

    def get_sub_geometry(self, direction_index: int, geometry, data_container=None):
        """Build sub-geometry and scalar sinogram for one q-direction.

        Drop-in replacement for ``NearestNeighbor.get_sub_geometry``.  Uses the
        pre-computed projection matrix instead of re-running arc integration.

        Parameters
        ----------
        direction_index :
            Index into ``y_directions`` / last axis of ``projection_matrix``.
        geometry :
            mumott ``Geometry`` of the full problem.
        data_container :
            Optional mumott ``DataContainer``.  When provided, returns the
            weighted scalar sinogram and weights as numpy arrays.

        Returns
        -------
        sub_geometry :
            mumott ``Geometry`` containing only the contributing projections.
        data_tuple :
            ``(data_array, weight_array)`` each shaped ``(N_sub, J, K_det)``,
            or ``None`` when ``data_container`` is not supplied.
        """
        from copy import deepcopy

        pm_k = self._pm[:, :, direction_index]   # (N, M)

        sub_geometry = deepcopy(geometry)
        sub_geometry.delete_projections()
        sub_geometry.detector_angles = np.array([0])
        sub_geometry.detector_direction_origin = np.array([0, 0, 0])
        sub_geometry.detector_direction_positive_90 = np.array([0, 0, 0])

        data_list: list = []
        weight_list: list = []

        for n in range(len(geometry)):
            if not np.any(pm_k[n] > 0.0):
                continue

            sub_geometry.append(deepcopy(geometry[n]))

            if data_container is not None:
                w = pm_k[n]                                            # (M,)
                proj_w = data_container.projections[n].weights         # (J, K_det, M)
                proj_d = data_container.projections[n].data            # (J, K_det, M)

                ww = proj_w * w[np.newaxis, np.newaxis, :]             # (J, K_det, M)
                sumw = ww.sum(axis=-1)                                  # (J, K_det)
                sumd = (proj_d * ww).sum(axis=-1)                      # (J, K_det)

                weight_list.append(sumw)
                data_list.append(
                    np.divide(sumd, sumw, out=np.zeros_like(sumd), where=sumw != 0)
                )

        if data_container is None:
            return sub_geometry, None
        if len(data_list) == 0:
            logger.warning('No projections found for direction index %d.', direction_index)
            return sub_geometry, None
        return sub_geometry, (np.stack(data_list, axis=0), np.stack(weight_list, axis=0))


# ---------------------------------------------------------------------------
# Full reconstruction
# ---------------------------------------------------------------------------

def saxs_fbp_reconstruction(
    dc: DataContainer,
    k_fibonacci: int = 50,
    filter_type: str = 'hann',
    n_projection_samples: int = 64,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    projection_method: str = 'voronoi',
    ball_threshold: float = 0.3,
    return_matrix: bool = False,
    half_space: str = 'z',
) -> tuple[torch.Tensor, np.ndarray]:
    """Reconstruct the q-sphere function of each voxel with independent FBPs.

    For each of ``k_fibonacci`` target q-directions y on the upper hemisphere,
    the scalar sinogram is built by contracting the full data tensor with the
    NearestNeighbor projection matrix on the GPU — one einsum per direction,
    rather than a CPU NumPy loop.  The ASTRA geometry is constructed once from
    the full projection set and reused for every backprojection: non-contributing
    projections are zeroed in the sinogram and thus contribute nothing to BP3D.

    Parameters
    ----------
    dc :
        mumott ``DataContainer`` (geometry + projection data loaded from HDF5).
    k_fibonacci :
        Number of target q-directions (Fibonacci hemisphere grid).
    filter_type :
        Ramp-filter window: ``'ram-lak'``, ``'hann'`` (default), ``'hamming'``,
        ``'cosine'``, ``'shepp-logan'``.
    n_projection_samples :
        Number of SLERP sample points per arc used when building the projection
        matrix on the GPU.  64 (default) matches the precision of mumott's
        adaptive-Simpson CPU integration at tolerance 1e-3.
    device :
        Compute device. Defaults to CUDA if available.
    verbose :
        Show a per-direction progress bar.
    projection_method : str
        ``'voronoi'`` (default): each arc sample is assigned to its nearest
        direction; pm rows sum to 1.  ``'ball'``: each arc sample contributes
        to every direction within ``ball_threshold`` radians; pm rows can sum
        to > 1, widening contributions to alleviate the missing-wedge problem.
    ball_threshold : float
        Angular radius in radians used when ``projection_method='ball'``.
        Ignored for ``'voronoi'``.
    return_matrix : bool
        Returns the geometry as well in a tuple
    half_space : str
        Which half-space to sample y-directions from: ``'z'`` (default,
        z > 0) or ``'y'`` (y > 0, aligns with goniometer frame).

    Returns
    -------
    reconstruction : (K, X, Y, Z) float32 tensor
        ``reconstruction[i]`` is the scalar volume for q-direction
        ``y_directions[i]``.
    y_directions : (K, 3) float64 ndarray
        Unit vectors of the K target q-directions on the upper hemisphere.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    geometry = dc.geometry
    vol_shape = tuple(geometry.volume_shape)

    # Exclude directions within alpha_max of the beam axis: those have no
    # accessible projections (no beam can be ⊥ y_k within the tilt cone).
    outer_angles = np.asarray(list(geometry.outer_angles))
    alpha_max_deg = math.degrees(float(np.max(np.abs(outer_angles))))
    y_directions = fibonacci_hemisphere(k_fibonacci, pole_gap_deg=alpha_max_deg, half_space=half_space)

    # ------------------------------------------------------------------
    # 1. Build projection matrix on GPU (replaces CPU NearestNeighbor)
    # ------------------------------------------------------------------
    logger.info(
        "Building projection matrix on GPU: K=%d, n_samples=%d, method=%s",
        k_fibonacci, n_projection_samples, projection_method,
    )
    pm_gpu = _build_projection_matrix_gpu(
        geometry.probed_coordinates,
        y_directions,
        enforce_friedel_symmetry=True,
        n_samples=n_projection_samples,
        device=device,
        method=projection_method,
        threshold=ball_threshold,
    )  # (N, M, K) already on device

    # ------------------------------------------------------------------
    # 2. Pre-compute full ASTRA geometry once (no SAXSProjector per iter)
    # ------------------------------------------------------------------
    vol_geom, proj_geom = _create_astra_geometries_from_mumott(geometry)

    # ------------------------------------------------------------------
    # 3. Move data to GPU
    #    dc.data shape  : (N, J, K_det, M)
    #    dc.weights     : (N, J, K_det, M)  — >0 means valid pixel
    # ------------------------------------------------------------------
    logger.info("Moving data to %s...", device)
    data_gpu = torch.tensor(
        dc.data.astype(np.float32), device=device,
    )  # (N, J, K_det, M)
    valid_gpu = torch.tensor(
        (dc.weights > 0).astype(np.float32), device=device,
    )  # (N, J, K_det, M)
    dw_gpu = data_gpu * valid_gpu   # validity-masked data

    zero_vol = torch.zeros(vol_shape, dtype=torch.float32, device=device)
    results: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # 4. One GPU einsum per direction → ramp filter → BP3D
    # ------------------------------------------------------------------
    iterator = range(k_fibonacci)
    if verbose:
        iterator = tqdm(iterator, desc="SAXS FBP", unit="dir")

    for k in iterator:
        pm_k = pm_gpu[:, :, k]   # (N, M) arc-fraction weights for direction k

        # Weighted sum over segments → scalar sinogram on GPU
        # sino_num[n, j, d] = Σ_m  data[n,j,d,m] · valid[n,j,d,m] · pm_k[n,m]
        # sino_den[n, j, d] = Σ_m  valid[n,j,d,m] · pm_k[n,m]
        sino_num = torch.einsum('njdm, nm -> njd', dw_gpu, pm_k)    # (N, J, K_det)
        sino_den = torch.einsum('njdm, nm -> njd', valid_gpu, pm_k) # (N, J, K_det)

        sino_k = sino_num / sino_den.clamp(min=1e-10)
        sino_k[sino_den < 1e-10] = 0.0   # zero out pixels with no valid data

        # Number of projections that actually contribute (for FBP normalisation)
        n_sub = int((sino_den.sum(dim=(-1, -2)) > 0).sum())
        if n_sub == 0:
            logger.debug("Direction %d: no contributing projections.", k)
            results.append(zero_vol)
            if verbose:
                iterator.set_postfix(n_sub=0)
            continue

        vol = _fbp_with_precomputed_geom(
            sino_k, vol_geom, proj_geom, vol_shape, filter_type, device,
        )
        vol = vol * (math.pi / (2.0 * n_sub))
        results.append(vol)

        if verbose:
            iterator.set_postfix(n_sub=n_sub)

    reconstruction = torch.stack(results, dim=0)  # (K, X, Y, Z)
    if return_matrix:
        pm_helper = FBPProjectionMatrix(pm_gpu.cpu().numpy(), y_directions)
        return reconstruction, y_directions, pm_helper
    return reconstruction, y_directions


# ---------------------------------------------------------------------------
# Missing-wedge masks
# ---------------------------------------------------------------------------

def missing_wedge_masks(
    y_directions: np.ndarray,
    vol_shape: tuple,
    geometry,
) -> np.ndarray:
    """Boolean missing-wedge masks in Fourier space for each FBP sub-volume.

    For sub-CT k (rotation axis y_k), a Fourier frequency q_f is inaccessible
    when the unique beam direction orthogonal to both y_k and q_f,

        b_int = (y_k × q_f) / |y_k × q_f|,

    falls outside the polar cap of accessible beam directions.  The cap is
    characterised by the global tilt_max inferred from the geometry: beams
    deviate from the beam axis (p_direction_0) by at most tilt_max.

    Parameters
    ----------
    y_directions : (K, 3) ndarray
        Unit q-direction vectors (output of :func:`fibonacci_hemisphere`).
    vol_shape : (X, Y, Z) tuple
        Shape of each reconstructed sub-volume.
    geometry :
        mumott ``Geometry`` of the full dataset.  Used to extract
        ``p_direction_0`` (beam axis) and the rotation matrices (to compute
        tilt_max empirically).

    Returns
    -------
    masks : (K, X, Y, Z) bool ndarray
        ``True`` where the Fourier bin is in the missing wedge.
    """
    # --- tilt_max from outer_angles (the physical constraint) --------------------
    # The outer_angle is the goniometer tilt.  For the standard SAXS-TT geometry
    # (beam along p_direction_0, outer tilt up to alpha_max) the accessible beam
    # directions form a polar cap around p_direction_0 with half-angle alpha_max,
    # so the z-component of any accessible beam (projected onto p_direction_0) is
    # >= cos(alpha_max).
    #
    # We deliberately use the outer_angle from the geometry rather than the actual
    # beam-direction dot products: complex scan trajectories (helical, conical, …)
    # can produce beam directions that span the full sphere while the physical tilt
    # constraint is still alpha_max, so reading beam_z empirically would give a
    # falsely wide accessible range and an all-zero missing-wedge mask.
    p0 = np.asarray(geometry.p_direction_0, dtype=float)
    p0 /= np.linalg.norm(p0)

    outer_angles = np.asarray(list(geometry.outer_angles))   # (N,) radians
    alpha_max = float(np.max(np.abs(outer_angles)))
    cos_alpha = math.cos(alpha_max)

    logger.debug("missing_wedge_masks: p0=%s  alpha_max=%.2f°  cos_alpha=%.4f",
                 p0, math.degrees(alpha_max), cos_alpha)

    # --- 3-D Fourier frequency grid ------------------------------------------
    X, Y, Z = vol_shape
    gx, gy, gz = np.meshgrid(
        np.fft.fftfreq(X), np.fft.fftfreq(Y), np.fft.fftfreq(Z), indexing='ij'
    )
    qf = np.stack([gx, gy, gz], axis=-1)   # (X, Y, Z, 3)

    # --- per-direction mask --------------------------------------------------
    masks = np.empty((len(y_directions), X, Y, Z), dtype=bool)

    for k, y_k in enumerate(y_directions):
        cross = np.cross(y_k, qf)                        # (X, Y, Z, 3)
        cross_norm = np.linalg.norm(cross, axis=-1)      # (X, Y, Z)
        parallel = cross_norm < 1e-10                     # q_f ∥ y_k

        # |b_int · p0| = |(y_k × q_f) · p0| / |y_k × q_f|
        # b_int is accessible when this projection onto the beam axis >= cos(alpha_max).
        # Friedel symmetry means ±b_int are equivalent, so we use the absolute value.
        b_int_dot_p0 = np.abs((cross * p0).sum(axis=-1)) / np.where(parallel, 1.0, cross_norm)

        missing = b_int_dot_p0 < cos_alpha
        missing[parallel] = False   # q_f ∥ y_k: all contributing beams see it

        masks[k] = missing

    return masks
