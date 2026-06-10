"""Missing-wedge geometry for the SAXS-TT isonet pipeline.

Terminology
-----------
RSM direction (``rsm_dir`` / ``rsm_dirs[k]``)
    A reciprocal-space-map location on the q-sphere (formerly ``y_dirs[k]``).
    Each RSM direction has its own scalar sub-CT volume with its own missing
    wedge, whose orientation and size depend on ``rsm_dir`` and the goniometer.
projection / beam directions
    The measurable beam set — the full sphere minus two polar caps of
    half-angle ``alpha_deg`` about ``goniometer_axis``.  Not enumerated here.

Geometric model
---------------
A beam direction b is measurable iff ``|b · goniometer_axis| ≤ cos(alpha_deg)``.
By the Fourier slice theorem, a frequency f of the sub-CT for ``rsm_dir`` is
MEASURED iff some measurable b satisfies b ⊥ f and b ⊥ rsm_dir.  The unique
candidate is b ∝ rsm_dir × f, giving the closed form

    f MEASURED  ⟺  |(rsm_dir × f) · goniometer_axis|  ≤  cos(alpha) · |f_perp|

with f_perp = f − (f · rsm_dir) rsm_dir.

Canonical frame
---------------
For the model to fill a wedge of unknown orientation, training and inference
both work in a *canonical frame*: ``canonical_rotation(rsm_dir, g)`` sends
g → z and rsm_dir into the xz-plane, so the wedge for a given polar angle θ is
fully determined (orientation + size) by the single scalar θ — the azimuthal
degree of freedom that otherwise made equal-arc wedges ambiguous is removed.

Public API
----------
goniometer_axis_for_half_space(half_space)
missing_arc_length(rsm_dir, alpha_deg, goniometer_axis=None) -> float
all_missing_arcs(rsm_dirs, alpha_deg, goniometer_axis=None) -> np.ndarray
missing_wedge_mask_3d(rsm_dir, alpha_deg, volume_shape, goniometer_axis=None) -> np.ndarray
canonical_rotation(rsm_dir, goniometer_axis=None) -> np.ndarray
canonical_rsm_dir(rsm_dir, goniometer_axis=None) -> np.ndarray
sinusoidal_wedge_embedding(missing_arc, dim=128, device=None) -> torch.Tensor
plot_wedge_hemisphere(...) / plot_wedge_mask_slices(...)
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def goniometer_axis_for_half_space(half_space: str) -> np.ndarray:
    """Return the tilt-axis unit vector matching fibonacci_hemisphere's half_space."""
    if half_space == 'z':
        return np.array([0., 0., 1.])
    if half_space == 'y':
        return np.array([0., 1., 0.])
    raise ValueError(f"Unknown half_space {half_space!r}. Choose 'z' or 'y'.")


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.],
    ])


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit vector ``a`` onto unit vector ``b`` (Rodrigues)."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:                  # already aligned
        return np.eye(3)
    if c < -1.0 + 1e-8:                 # antiparallel — 180° about any ⊥ axis
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        axis = _unit(np.cross(a, perp))
        vx = _skew(axis)
        return np.eye(3) + 2.0 * vx @ vx   # 180° rotation
    vx = _skew(v)
    s2 = float(np.dot(v, v))
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s2)


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0.],
        [s,  c, 0.],
        [0., 0., 1.],
    ])


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c, 0., s],
        [0., 1., 0.],
        [-s, 0., c],
    ])


# ---------------------------------------------------------------------------
# Conditioning scalar
# ---------------------------------------------------------------------------

def missing_arc_length(
    rsm_dir: np.ndarray,
    alpha_deg: float,
    goniometer_axis: Optional[np.ndarray] = None,
) -> float:
    """Arc length (radians) of the UNMEASURED portion of the great circle ⊥ rsm_dir.

    Returns a value in [0, 2π]:
      0   — no missing wedge (rsm_dir at the goniometer pole).
      2π  — entire great circle missing (not reached in practice).

    This monotone scalar is the conditioning value: in the canonical frame it
    determines the wedge orientation and size uniquely.
    """
    if goniometer_axis is None:
        goniometer_axis = np.array([0., 0., 1.])
    g = _unit(goniometer_axis)
    y = _unit(rsm_dir)

    cos_alpha = np.cos(np.radians(alpha_deg))
    sin_theta = float(np.sqrt(max(0., 1. - np.dot(y, g) ** 2)))

    if sin_theta <= cos_alpha:          # rsm_dir near pole — full circle measurable
        return 0.0
    return float(2 * np.pi - 4 * np.arcsin(min(1., cos_alpha / sin_theta)))


def all_missing_arcs(
    rsm_dirs: np.ndarray,
    alpha_deg: float,
    goniometer_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorised :func:`missing_arc_length` over all K RSM directions (radians)."""
    return np.array([
        missing_arc_length(r, alpha_deg, goniometer_axis) for r in rsm_dirs
    ])


# ---------------------------------------------------------------------------
# 3-D Fourier mask
# ---------------------------------------------------------------------------

def missing_wedge_mask_3d(
    rsm_dir: np.ndarray,
    alpha_deg: float,
    volume_shape: tuple,
    goniometer_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean 3-D Fourier-space mask in **unshifted** (fftfreq) layout.

    ``True = measured``, ``False = missing``.  Computed analytically from the
    closed-form Fourier-slice condition — no geometry object needed.

    Parameters
    ----------
    rsm_dir : (3,) unit RSM direction for this sub-CT volume.
    alpha_deg : half-angle (°) of the unmeasurable polar caps.
    volume_shape : (Nx, Ny, Nz) real-space dimensions.
    goniometer_axis : tilt-axis unit vector. Defaults to [0,0,1].
    """
    if goniometer_axis is None:
        goniometer_axis = np.array([0., 0., 1.])
    g = _unit(goniometer_axis)
    y = _unit(rsm_dir)
    cos_alpha = np.cos(np.radians(alpha_deg))

    FX, FY, FZ = np.meshgrid(
        np.fft.fftfreq(volume_shape[0]),
        np.fft.fftfreq(volume_shape[1]),
        np.fft.fftfreq(volume_shape[2]),
        indexing='ij',
    )

    # Scalar triple product (rsm_dir × f) · g
    cross_dot_g = (
        g[0] * (y[1] * FZ - y[2] * FY) +
        g[1] * (y[2] * FX - y[0] * FZ) +
        g[2] * (y[0] * FY - y[1] * FX)
    )

    f_dot_y = y[0] * FX + y[1] * FY + y[2] * FZ
    f_perp = np.sqrt(np.maximum(FX**2 + FY**2 + FZ**2 - f_dot_y**2, 0.))

    mask = np.abs(cross_dot_g) <= cos_alpha * f_perp

    # Enforce Friedel symmetry: keep(f) == keep(-f).  The analytic condition is
    # even in f, but the discrete fftfreq grid breaks this on the DC/Nyquist
    # planes (a frequency maps to itself under negation there).  Without exact
    # symmetry, carving a wedge from a real volume and taking the real part
    # reintroduces energy into the wedge.  AND with the mirrored mask fixes it.
    negs = tuple((-np.arange(s)) % s for s in volume_shape)
    mask = mask & mask[np.ix_(*negs)]

    mask[0, 0, 0] = True                # DC always measured
    return mask


# ---------------------------------------------------------------------------
# Canonical frame
# ---------------------------------------------------------------------------

def canonical_rotation(
    rsm_dir: np.ndarray,
    goniometer_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rotation R (3×3) mapping the wedge of ``rsm_dir`` into the canonical frame.

    The canonical frame is defined so that the missing-wedge cone axis lies along
    the y-axis (ky direction) and the goniometer axis is in the yz-plane:

      R @ rsm_dir  = y_hat = (0, 1, 0)
      R @ g        = (0, cosθ, sinθ)   where θ = angle(rsm_dir, g)

    In this frame the missing-wedge condition simplifies to:

      |fx · sinθ| > cos(α) · √(fx² + fz²)

    which is a band symmetric about the kz-axis, **identical in every ky slice**.
    This is the clean orientation expected in the notebook: as you move off-centre
    in ky, the wedge boundary stays aligned (no tilt).

    ``R`` is used at inference to rotate the whole volume into canonical
    orientation before patching, and ``Rᵀ`` rotates the filled result back.
    """
    if goniometer_axis is None:
        goniometer_axis = np.array([0., 0., 1.])
    g = _unit(goniometer_axis)
    r = _unit(rsm_dir)
    y_hat = np.array([0., 1., 0.])

    # Step 1: rotate rsm_dir → y_hat.
    R1 = _rotation_between(r, y_hat)

    # Step 2: rotate about y_hat so the goniometer axis lands in the yz-plane
    # (x-component = 0, z-component = sinθ > 0).
    g1 = R1 @ g               # g in the R1 frame; g1_y = cosθ (preserved by R1)
    # Azimuthal angle of g1's xz-projection from the z-axis.
    phi = np.arctan2(g1[0], g1[2])   # atan2(0,0)=0 → identity when g ∥ y
    R2 = _rot_y(-phi)

    return R2 @ R1


def canonical_rsm_dir(
    rsm_dir: np.ndarray,
    goniometer_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """RSM direction in the canonical frame — always y_hat = (0, 1, 0).

    The canonical rotation is defined to map rsm_dir → y_hat, so this is a
    constant.  The function is kept for API symmetry and as a reminder that the
    canonical mask must use y_hat as the RSM direction.
    """
    return np.array([0., 1., 0.])


def canonical_goniometer_axis(
    rsm_dir: np.ndarray,
    goniometer_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Goniometer axis in the canonical frame: ``(0, cosθ, sinθ)``.

    θ = angle(rsm_dir, goniometer_axis) is preserved by the rotation, and the
    canonical rotation maps g into the yz-plane.  The canonical missing-wedge
    mask must be computed with this axis (not z_hat) to produce the correct
    kz-aligned band.
    """
    if goniometer_axis is None:
        goniometer_axis = np.array([0., 0., 1.])
    g = _unit(goniometer_axis)
    r = _unit(rsm_dir)
    cos_theta = float(np.clip(np.dot(r, g), -1., 1.))
    sin_theta = float(np.sqrt(max(0., 1. - cos_theta ** 2)))
    return np.array([0., cos_theta, sin_theta])


# ---------------------------------------------------------------------------
# Model conditioning embedding
# ---------------------------------------------------------------------------

def sinusoidal_wedge_embedding(
    missing_arc: float,
    dim: int = 128,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sinusoidal embedding of the missing-arc scalar → shape (1, 1, dim).

    Ready to use as ``encoder_hidden_states`` in UNet3DConditionModel (caller
    expands the batch dimension).  ``missing_arc`` is normalised to [0, 1] by
    dividing by 2π, matching diffusion-model timestep-embedding convention.
    """
    if device is None:
        device = torch.device('cpu')
    t = torch.tensor([missing_arc / (2 * np.pi)], dtype=torch.float32, device=device)
    half = dim // 2
    freqs = torch.exp(
        -np.log(10_000) * torch.arange(half, dtype=torch.float32, device=device) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]                  # (1, half)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)   # (1, dim) for even dim
    if emb.shape[-1] < dim:                             # odd dim: pad one column
        emb = torch.cat([emb, torch.zeros(1, dim - emb.shape[-1], device=device)], dim=-1)
    return emb.unsqueeze(0)                             # (1, 1, dim)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_wedge_hemisphere(
    rsm_dir: np.ndarray,
    alpha_deg: float,
    goniometer_axis: Optional[np.ndarray] = None,
    n_points: int = 600,
    swap_yz: bool = True,
    title_extra: str = '',
):
    """Plotly figure showing measured (green) vs missing (red) great-circle arcs.

    Overlays:
    - Great circle ⊥ rsm_dir, coloured green (measured) / red (missing).
    - Two dashed orange circles marking the belt boundaries (|b·g| = cos α).
    - A blue marker for rsm_dir itself.

    Parameters
    ----------
    swap_yz : match the (x, z, y) axis convention used by hemisphere_plotly
        in the saxs_fbp_test notebook.  True by default.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for plot_wedge_hemisphere")

    if goniometer_axis is None:
        goniometer_axis = np.array([0., 0., 1.])
    g = _unit(goniometer_axis)
    y = _unit(rsm_dir)
    cos_alpha = np.cos(np.radians(alpha_deg))

    def _plot_coords(pts: np.ndarray):
        """pts : (N, 3) — return (px, py, pz) for plotly."""
        if swap_yz:
            return pts[:, 0], pts[:, 2], pts[:, 1]
        return pts[:, 0], pts[:, 1], pts[:, 2]

    # Great circle orthogonal to rsm_dir
    perp = np.array([1., 0., 0.]) if abs(y[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(y, perp); u /= np.linalg.norm(u)
    v = np.cross(y, u)
    t_gc = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    gc = np.cos(t_gc)[:, None] * u + np.sin(t_gc)[:, None] * v  # (N, 3)

    measurable = np.abs(gc @ g) <= cos_alpha  # (N,) bool

    # Split into contiguous segments of same measurability
    changes = np.where(np.diff(measurable.astype(int)) != 0)[0] + 1
    bounds = [0] + list(changes) + [n_points]
    segments = [(bounds[i], bounds[i + 1], bool(measurable[bounds[i]]))
                for i in range(len(bounds) - 1)]

    traces = []
    _seen_meas = _seen_miss = False
    for start, end, is_meas in segments:
        idx = list(range(start, end)) + [start]   # wrap to close the gap
        seg = gc[idx]
        px, py, pz = _plot_coords(seg)
        color = 'limegreen' if is_meas else 'crimson'
        label = 'measured arc' if is_meas else 'missing arc'
        show = (is_meas and not _seen_meas) or (not is_meas and not _seen_miss)
        if is_meas:
            _seen_meas = True
        else:
            _seen_miss = True
        traces.append(go.Scatter3d(
            x=px, y=py, z=pz, mode='lines',
            line=dict(color=color, width=5), name=label, showlegend=show,
        ))

    # Belt boundary circles (|b · g| = cos α)
    e1 = np.cross(g, [1., 0., 0.] if abs(g[0]) < 0.9 else [0., 1., 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g, e1)
    sin_alpha = np.sin(np.radians(alpha_deg))
    t_bc = np.linspace(0, 2 * np.pi, 300)
    for i, sign in enumerate([1., -1.]):
        bc = (sin_alpha * (np.cos(t_bc)[:, None] * e1 + np.sin(t_bc)[:, None] * e2)
              + sign * cos_alpha * g)
        px, py, pz = _plot_coords(bc)
        traces.append(go.Scatter3d(
            x=px, y=py, z=pz, mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
            name=f'belt boundary (α={alpha_deg:.0f}°)', showlegend=(i == 0),
        ))

    # rsm_dir marker
    px, py, pz = _plot_coords(y[np.newaxis])
    traces.append(go.Scatter3d(
        x=px, y=py, z=pz, mode='markers',
        marker=dict(size=9, color='steelblue'),
        name=f'rsm_dir ({y[0]:.2f}, {y[1]:.2f}, {y[2]:.2f})',
    ))

    arc = missing_arc_length(rsm_dir, alpha_deg, goniometer_axis)
    pct = arc / (2 * np.pi) * 100
    scene = (dict(xaxis_title='x', yaxis_title='z', zaxis_title='y')
             if swap_yz else dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
    fig = go.Figure(traces)
    fig.update_layout(
        scene=scene,
        title=(f'Missing wedge  α={alpha_deg:.0f}°  |  '
               f'missing arc = {np.degrees(arc):.1f}°  ({pct:.0f}% of circle)'
               + (f'  |  {title_extra}' if title_extra else '')),
        width=700, height=560,
        legend=dict(font=dict(size=10)),
    )
    return fig


def plot_wedge_mask_slices(
    rsm_dir: np.ndarray,
    alpha_deg: float,
    volume_shape: tuple = (64, 64, 64),
    goniometer_axis: Optional[np.ndarray] = None,
):
    """Matplotlib figure with three central slices of the 3-D Fourier mask.

    Green = measured, purple = missing.  Panels show the XY, XZ, YZ planes
    through the DC component (centre of the fftshifted Fourier volume).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mask = missing_wedge_mask_3d(rsm_dir, alpha_deg, volume_shape, goniometer_axis)
    mask_s = np.fft.fftshift(mask)      # DC to centre

    cx, cy, cz = [s // 2 for s in volume_shape]
    slices = {
        'XY  (z=0)': mask_s[:, :, cz],
        'XZ  (y=0)': mask_s[:, cy, :],
        'YZ  (x=0)': mask_s[cx, :, :],
    }

    cmap = mcolors.ListedColormap(['#7b2d8b', '#4caf50'])  # purple=missing, green=measured
    arc = missing_arc_length(rsm_dir, alpha_deg, goniometer_axis)
    pct = arc / (2 * np.pi) * 100

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, sl) in zip(axes, slices.items()):
        im = ax.imshow(sl.T, origin='lower', cmap=cmap, vmin=0, vmax=1,
                       interpolation='nearest', aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('axis 0 freq'); ax.set_ylabel('axis 1 freq')
        frac = sl.mean() * 100
        ax.text(0.02, 0.97, f'{frac:.0f}% measured', transform=ax.transAxes,
                va='top', fontsize=8, color='white')

    y_str = f'({rsm_dir[0]:.2f},{rsm_dir[1]:.2f},{rsm_dir[2]:.2f})'
    g_label = '' if goniometer_axis is None else f'  g={np.round(goniometer_axis,2)}'
    fig.suptitle(
        f'Fourier mask slices  |  rsm_dir={y_str}  α={alpha_deg:.0f}°{g_label}\n'
        f'missing arc={np.degrees(arc):.1f}°  ({pct:.0f}% of great circle)',
        fontsize=11,
    )
    plt.colorbar(im, ax=axes[-1], fraction=0.046,
                 ticks=[0.25, 0.75]).set_ticklabels(['missing', 'measured'])
    plt.tight_layout()
    return fig
