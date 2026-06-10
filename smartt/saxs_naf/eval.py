"""Evaluation for SAXS NAF: per-RSM-direction reconstruction comparison.

Mirrors ``notebooks/saxs_fbp_method_comparison.ipynb``: discretise the sphere
into ``K`` Fibonacci directions, evaluate each model's SH-coefficient field at
those directions to obtain ``(K, X, Y, Z)`` RSM volumes, crop to the inscribed
sphere, and compare per direction.  Works with or without a remounted/full
reference DataContainer.

The SH point-evaluation here uses the **same convention as
``forward_quadrature``** (4π-normalised real SH, Condon-Shortley cancelled), so
it is consistent with the basis the NAF coefficients are optimised in — do NOT
substitute mumott's ``_get_projection_matrix`` for NAF coefficients.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from smartt.shutils.evaulate_sh import associated_legendre, _generate_lm_list


def evaluate_real_sh(directions: torch.Tensor, ell_max: int) -> torch.Tensor:
    """Real-SH basis matrix ``B[k, c] = Y_c(direction_k)``.

    Parameters
    ----------
    directions : ``(K, 3)`` unit vectors (any float dtype/device).
    ell_max : Maximum even degree.

    Returns
    -------
    ``(K, C)`` tensor in the ``forward_quadrature`` convention.
    """
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    cos_theta = z.clamp(-1.0, 1.0)
    phi = torch.atan2(y, x)

    lm_list = _generate_lm_list(ell_max)
    cols: List[torch.Tensor] = []
    for l, m in lm_list:
        ma = abs(m)
        P = associated_legendre(l, ma, cos_theta)
        two_minus_delta = 1 if ma == 0 else 2
        factor = math.sqrt(
            two_minus_delta * (2 * l + 1)
            * (math.factorial(l - ma) / float(math.factorial(l + ma)))
        )
        cs = (-1.0) ** ma          # cancel Condon-Shortley built into associated_legendre
        if m > 0:
            col = cs * factor * P * torch.cos(ma * phi)
        elif m < 0:
            col = cs * factor * P * torch.sin(ma * phi)
        else:
            col = factor * P
        cols.append(col)
    return torch.stack(cols, dim=-1)   # (K, C)


def coeffs_to_rsm_volumes(
    coeffs: torch.Tensor, directions: np.ndarray, ell_max: int
) -> torch.Tensor:
    """``(X, Y, Z, C)`` coefficients → ``(K, X, Y, Z)`` directional volumes."""
    dirs = torch.as_tensor(directions, dtype=coeffs.dtype, device=coeffs.device)
    B = evaluate_real_sh(dirs, ell_max)                    # (K, C)
    return torch.einsum("xyzc,kc->kxyz", coeffs, B)


def relative_anisotropy(rsm_volumes: torch.Tensor, sphere_mask=None) -> torch.Tensor:
    """Per-voxel RA = std / mean of the RSM samples over the K directions.

    A robust scalar anisotropy proxy (robust to the missing wedge), computed
    directly from the directional samples rather than the SH tensor.
    """
    mean = rsm_volumes.mean(dim=0)
    std = rsm_volumes.std(dim=0)
    ra = std / mean.clamp_min(1e-8)
    return ra


def evaluate_models(
    models: Dict[str, torch.Tensor],
    *,
    ell_max: int = 8,
    K: int = 30,
    half_space: str = "y",
    alpha_deg: Optional[float] = None,
    goniometer_axis: Optional[np.ndarray] = None,
    cube_size: Optional[int] = None,
) -> Dict:
    """Evaluate several coefficient fields into cropped RSM volumes for comparison.

    Parameters
    ----------
    models : mapping ``name -> (X, Y, Z, C)`` SH-coefficient tensor.  Different
        methods (NAF, direct-GD, mumott-LBFGS, …) keyed by label.
    ell_max, K, half_space : RSM sampling controls.
    alpha_deg, goniometer_axis : if given, annotate each direction with its
        missing-arc (degrees) for wedge-severity bucketing.
    cube_size : inscribed-cube side (multiple of 8); default from volume shape.

    Returns
    -------
    dict with ``y_directions`` ``(K,3)``, ``volumes`` (``name -> (K,d,d,d)``),
    ``sphere_mask`` ``(d,d,d)`` bool, ``cube_size``, and optional
    ``missing_arcs_deg`` ``(K,)``.
    """
    from smartt.saxs_fbp import fibonacci_hemisphere
    from smartt.saxs_isonet.preprocess import spherical_crop

    y_directions = fibonacci_hemisphere(K, half_space=half_space)

    volumes: Dict[str, torch.Tensor] = {}
    sphere_mask = None
    d = None
    for name, coeffs in models.items():
        vols = coeffs_to_rsm_volumes(coeffs.float(), y_directions, ell_max)  # (K,X,Y,Z)
        cropped, sphere_mask, d = spherical_crop(vols.cpu(), cube_size=cube_size)
        volumes[name] = cropped

    out = {
        "y_directions": y_directions,
        "volumes": volumes,
        "sphere_mask": sphere_mask,
        "cube_size": d,
    }
    if alpha_deg is not None:
        from smartt.saxs_isonet.wedge import all_missing_arcs

        out["missing_arcs_deg"] = np.degrees(
            all_missing_arcs(y_directions, alpha_deg, goniometer_axis)
        )
    return out


def plot_rsm_direction(result: Dict, k: int, axis: str = "z", slice_index=None):
    """Quick orthogonal-slice comparison of all models for direction ``k``.

    Mirrors the fbp-comparison notebook's per-direction view: one column per
    model, shared colour scale from the reference (first) model's sphere interior.
    """
    import matplotlib.pyplot as plt

    volumes = result["volumes"]
    sphere_mask = result["sphere_mask"]
    names = list(volumes.keys())
    d = result["cube_size"]
    if slice_index is None:
        slice_index = d // 2

    ref = volumes[names[0]][k].numpy()
    lo, hi = np.percentile(ref[sphere_mask], [2, 98])

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), squeeze=False)
    for col, name in enumerate(names):
        vol = volumes[name][k].numpy()
        sl = {"x": vol[slice_index], "y": vol[:, slice_index], "z": vol[:, :, slice_index]}[axis]
        im = axes[0, col].imshow(sl.T, cmap="inferno", vmin=lo, vmax=hi, origin="lower")
        axes[0, col].set_title(name)
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)

    ydir = np.round(result["y_directions"][k], 3)
    title = f"RSM direction k={k}  y={ydir}"
    if "missing_arcs_deg" in result:
        title += f"  missing_arc={result['missing_arcs_deg'][k]:.1f}°"
    fig.suptitle(title)
    fig.tight_layout()
    return fig
