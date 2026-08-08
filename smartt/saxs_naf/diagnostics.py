"""Effective-rank diagnostics for SAXS NAF: is the anisotropic shape
voxel-independent, or is a single template being read out everywhere at
different amplitudes?

A hash-grid encoder gives every voxel a genuinely distinct input feature
vector, so a repeating-shape artifact is not a structural inability to vary
spatially. It can still arise from two different, independently-fixable
causes, each targeted by a function here:

* **Decoder collapse** (:func:`head_weight_rank`) — the linear readout
  ``coeffs(x) = W @ trunk_out(x)`` has a (near-)rank-1 ``W`` restricted to the
  ℓ>0 rows, so every voxel's rich, distinct ``trunk_out(x)`` gets projected
  onto the same output direction ``v``: ``coeffs(x) ≈ a(x)·v``. Amplitude
  ``a(x)`` can vary richly (matching complex intensity data) while the shape
  ``v`` stays fixed.
* **Encoder collapse** (:func:`trunk_output_rank`) — ``trunk_out(x)`` itself
  has collapsed onto a low-dimensional manifold (e.g. because only coarse,
  near-global hash levels are active), so no downstream readout could give
  voxels independent shapes even if it wanted to.

:func:`shape_effective_rank` is model-agnostic (works on any ``(X,Y,Z,C)`` SH
array — NAF, mumott_sh/gk, or ground truth) and measures the net effect
directly on the reconstructed field, independent of which cause produced it.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _effective_rank(values: np.ndarray) -> Dict[str, float]:
    """Participation ratio + #components for 90% energy of non-negative values.

    Participation ratio ``(Σs²)² / Σs⁴`` (computed here from singular values,
    so on their squares = energy) is a continuous "effective number of
    directions": 1.0 for a perfectly rank-1 spectrum, up to ``len(values)``
    for a flat one. Robust to the exact rank cutoff choice that a hard
    ``matrix_rank`` threshold would need.
    """
    s = np.asarray(values, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return {"participation_ratio": 0.0, "n_components_90": 0}
    energy = s ** 2
    total = energy.sum()
    pr = float(total ** 2 / (energy ** 2).sum())
    cum = np.cumsum(energy) / total
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    return {"participation_ratio": pr, "n_components_90": n90}


def shape_effective_rank(
    coeffs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    min_c00_percentile: float = 50.0,
    max_voxels: int = 50_000,
    seed: int = 0,
) -> Dict:
    """Effective rank of the per-voxel *normalised* anisotropic shape vector.

    Model-agnostic — takes any ``(X, Y, Z, C)`` SH coefficient array. Each
    voxel's ``ell>0`` coefficients are divided by its ``c00`` (factoring out
    amplitude), then SVD'd across voxels. A collapsed (near-1) effective rank
    means most voxels share approximately the same normalised shape, just
    rescaled — the "same RSM everywhere" failure mode. Higher is better.

    Parameters
    ----------
    coeffs : ``(X, Y, Z, C)`` float array.
    mask : optional ``(X, Y, Z)`` boolean array selecting voxels to include.
        Defaults to ``c00`` above its ``min_c00_percentile`` (background /
        near-zero-intensity voxels have ill-defined shape and would otherwise
        dominate via division by a tiny ``c00``).
    max_voxels : subsample this many voxels (for SVD cost) if more are masked.

    Returns
    -------
    dict with ``participation_ratio``, ``n_components_90``, the leading
    singular values, ``n_voxels`` used, and ``n_channels`` (``C - 1``).
    """
    c00 = coeffs[..., 0]
    aniso = coeffs[..., 1:]
    if mask is None:
        thresh = np.percentile(c00, min_c00_percentile)
        mask = c00 > thresh
    vecs = aniso[mask] / np.clip(c00[mask][:, None], 1e-8, None)
    n = vecs.shape[0]
    if n > max_voxels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_voxels, replace=False)
        vecs = vecs[idx]
        n = max_voxels
    if n < 2:
        return {"error": "not enough voxels in mask", "n_voxels": int(n)}
    s = np.linalg.svd(vecs.astype(np.float64), compute_uv=False)
    out = _effective_rank(s)
    out.update(
        singular_values=s[:20].tolist(),
        n_voxels=int(n),
        n_channels=int(vecs.shape[1]),
    )
    return out


def head_weight_rank(model) -> Dict:
    """Effective rank of the linear head's ℓ>0 output rows (decoder collapse).

    Low rank means the map from trunk features to SH-coefficient space
    projects (almost) every voxel's features onto the same output direction,
    regardless of how rich the per-voxel input actually is. Requires a
    :class:`~smartt.saxs_naf.model.SaxsNafField` (or anything exposing
    ``.head.weight`` shaped ``(C, hidden)``).
    """
    W = model.head.weight.detach().cpu().numpy().astype(np.float64)  # (C, hidden)
    W_aniso = W[1:]  # exclude the c00 row
    s = np.linalg.svd(W_aniso, compute_uv=False)
    out = _effective_rank(s)
    out.update(singular_values=s.tolist(), shape=list(W_aniso.shape))
    return out


def trunk_output_rank(model, max_voxels: int = 20_000, seed: int = 0) -> Dict:
    """Effective rank of the shared trunk's per-voxel output features (encoder diversity).

    Evaluated on the native grid with no annealing masks (full field). Low
    rank means many voxels share (nearly) the same feature vector — e.g.
    because they fall in the same coarse hash cell — so no linear readout
    downstream could give them independent shapes even if it wanted to.
    Requires a :class:`~smartt.saxs_naf.model.SaxsNafField`.
    """
    import torch

    device = model.grid_coords.device
    coords = model.grid_coords
    n = coords.shape[0]
    if n > max_voxels:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        idx = torch.randperm(n, device=device, generator=g)[:max_voxels]
        coords = coords[idx]
    with torch.no_grad():
        feats = model.encoding(coords)
        trunk_out = model.trunk(feats)
    arr = trunk_out.detach().cpu().numpy().astype(np.float64)
    s = np.linalg.svd(arr, compute_uv=False)
    out = _effective_rank(s)
    out.update(
        singular_values=s[:20].tolist(),
        n_voxels=int(arr.shape[0]),
        n_features=int(arr.shape[1]),
    )
    return out


def full_rank_report(model, coeffs: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
    """Bundle all three diagnostics for a trained :class:`SaxsNafField`."""
    return {
        "shape_effective_rank": shape_effective_rank(coeffs, mask=mask),
        "head_weight_rank": head_weight_rank(model),
        "trunk_output_rank": trunk_output_rank(model),
    }
