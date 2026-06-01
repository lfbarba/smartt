"""Missing-wedge geometry for the SAXS-TT isonet pipeline.

Geometric model
---------------
The goniometer defines a *measurable belt* on the unit sphere of beam directions:
all directions b satisfying

    |b · goniometer_axis|  ≤  cos(alpha_deg)

i.e. the full sphere minus two polar caps of half-angle alpha_deg.
Directions near the pole are NOT measurable (the goniometer cannot reach them).

For volume k (q-direction y_dir), the Fourier slice theorem says projection n
samples the plane perpendicular to its beam direction b_n.  A frequency f is
MEASURED iff some measurable b_n satisfies b_n ⊥ f and b_n ⊥ y_dir.
The unique such candidate is b ∝ y_dir × f, giving the closed-form condition:

    f MEASURED  ⟺  |(y_dir × f) · goniometer_axis|  ≤  cos(alpha) · |f_perp|

where f_perp = f − (f · y_dir) y_dir.

Verification:
  y_dir ‖ goniometer_axis (pole):  (pole × f) · axis ≡ 0  → all f measured ✓
  y_dir ⊥ goniometer_axis (equator): largest possible missing wedge ✓

Public API
----------
goniometer_axis_for_half_space(half_space)
missing_arc_length(y_dir, alpha_deg, goniometer_axis=None) -> float
missing_wedge_mask_3d(y_dir, alpha_deg, volume_shape, goniometer_axis=None) -> np.ndarray
sinusoidal_wedge_embedding(missing_arc, dim=128, device=None) -> torch.Tensor
plot_wedge_hemisphere(y_dir, alpha_deg, goniometer_axis=None, ...) -> plotly Figure
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


# ---------------------------------------------------------------------------
# Conditioning scalar
# ---------------------------------------------------------------------------

def missing_arc_length(
    y_dir: np.ndarray,
    alpha_deg: float,
    goniometer_axis: Optional[np.ndarray] = None,
) -> float:
    """Arc length (radians) of the UNMEASURED portion of the great circle ⊥ y_dir.

    Returns a value in [0, 2π]:
      0      — no missing wedge (y_dir near the goniometer pole).
      2π     — entire great circle missing (impossible in practice).

    Parameters
    ----------
    y_dir : (3,) array — unit q-direction for this sub-CT.
    alpha_deg : half-angle (degrees) of the unmeasurable polar caps.
    goniometer_axis : tilt axis unit vector. Defaults to [0,0,1].
        Use goniometer_axis_for_half_space(half_space) to match the
        convention used by fibonacci_hemisphere.
    """
    if goniometer_axis is None:
        goniometer_axis = np.array([0., 0., 1.])
    g = _unit(goniometer_axis)
    y = _unit(y_dir)

    cos_alpha = np.cos(np.radians(alpha_deg))
    # sin of polar angle of y_dir from goniometer_axis
    sin_theta = float(np.sqrt(max(0., 1. - np.dot(y, g) ** 2)))

    if sin_theta <= cos_alpha:  # y_dir near pole — full great circle is measurable
        return 0.0
    return float(2 * np.pi - 4 * np.arcsin(min(1., cos_alpha / sin_theta)))


def all_missing_arcs(
    y_dirs: np.ndarray,
    alpha_deg: float,
    goniometer_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorised missing_arc_length over all K directions.

    Returns (K,) float array of missing arc lengths in radians.
    """
    return np.array([
        missing_arc_length(y, alpha_deg, goniometer_axis) for y in y_dirs
    ])


# ---------------------------------------------------------------------------
# 3-D Fourier mask
# ---------------------------------------------------------------------------

def missing_wedge_mask_3d(
    y_dir: np.ndarray,
    alpha_deg: float,
    volume_shape: tuple,
    goniometer_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean 3-D Fourier-space mask.  True = measured, False = missing.

    Computed analytically from the closed-form condition derived from the
    Fourier slice theorem.  No geometry object or projection matrix needed.

    Parameters
    ----------
    y_dir : (3,) array — unit q-direction for this sub-CT volume.
    alpha_deg : half-angle (degrees) of the unmeasurable polar caps.
    volume_shape : (Nx, Ny, Nz) — real-space volume dimensions.
    goniometer_axis : tilt-axis unit vector. Defaults to [0,0,1].
        Pass goniometer_axis_for_half_space('y') when using half_space='y'.

    Returns
    -------
    mask : (Nx, Ny, Nz) bool array.  True = that Fourier voxel was measured.
    """
    if goniometer_axis is None:
        goniometer_axis = np.array([0., 0., 1.])
    g = _unit(goniometer_axis)
    y = _unit(y_dir)
    cos_alpha = np.cos(np.radians(alpha_deg))

    # 3-D FFT frequency grid (each axis in [-0.5, 0.5))
    FX, FY, FZ = np.meshgrid(
        np.fft.fftfreq(volume_shape[0]),
        np.fft.fftfreq(volume_shape[1]),
        np.fft.fftfreq(volume_shape[2]),
        indexing='ij',
    )

    # Scalar triple product  (y × f) · g
    cross_dot_g = (
        g[0] * (y[1] * FZ - y[2] * FY) +
        g[1] * (y[2] * FX - y[0] * FZ) +
        g[2] * (y[0] * FY - y[1] * FX)
    )

    # |f_perp| = |f - (f·y)y|
    f_dot_y = y[0] * FX + y[1] * FY + y[2] * FZ
    f_perp = np.sqrt(np.maximum(FX**2 + FY**2 + FZ**2 - f_dot_y**2, 0.))

    mask = np.abs(cross_dot_g) <= cos_alpha * f_perp
    mask[0, 0, 0] = True  # DC always measured
    return mask


# ---------------------------------------------------------------------------
# Model conditioning embedding
# ---------------------------------------------------------------------------

def sinusoidal_wedge_embedding(
    missing_arc: float,
    dim: int = 128,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sinusoidal embedding of the missing-arc scalar.

    Returns shape (1, 1, dim) — ready to use as encoder_hidden_states in
    UNet3DConditionModel (batch dimension is 1; caller should expand to batch).

    missing_arc is normalised to [0, 1] (divided by 2π) before embedding,
    matching the convention of diffusion-model timestep embeddings.
    """
    if device is None:
        device = torch.device('cpu')
    t = torch.tensor([missing_arc / (2 * np.pi)], dtype=torch.float32, device=device)
    half = dim // 2
    freqs = torch.exp(
        -np.log(10_000) * torch.arange(half, dtype=torch.float32, device=device) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]             # (1, half)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (1, dim)
    return emb.unsqueeze(0)                         # (1, 1, dim)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_wedge_hemisphere(
    y_dir: np.ndarray,
    alpha_deg: float,
    goniometer_axis: Optional[np.ndarray] = None,
    n_points: int = 600,
    swap_yz: bool = True,
    title_extra: str = '',
):
    """Plotly figure showing measured (green) vs missing (red) great-circle arcs.

    Overlays:
    - Great circle ⊥ y_dir, coloured green (measured) / red (missing).
    - Two dashed orange circles marking the belt boundaries (|b·g| = cos α).
    - A blue marker for y_dir itself.

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
    y = _unit(y_dir)
    cos_alpha = np.cos(np.radians(alpha_deg))

    def _plot_coords(pts: np.ndarray):
        """pts : (N, 3) — return (px, py, pz) for plotly."""
        if swap_yz:
            return pts[:, 0], pts[:, 2], pts[:, 1]
        return pts[:, 0], pts[:, 1], pts[:, 2]

    # ------------------------------------------------------------------
    # Great circle orthogonal to y_dir
    # ------------------------------------------------------------------
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
        # wrap around: append first point to close visual gap
        idx = list(range(start, end)) + [start]
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
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color=color, width=5),
            name=label,
            showlegend=show,
        ))

    # ------------------------------------------------------------------
    # Belt boundary circles  (|b · g| = cos α)
    # ------------------------------------------------------------------
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
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
            name=f'belt boundary (α={alpha_deg:.0f}°)',
            showlegend=(i == 0),
        ))

    # ------------------------------------------------------------------
    # y_dir marker
    # ------------------------------------------------------------------
    yd = y[np.newaxis]  # (1, 3)
    px, py, pz = _plot_coords(yd)
    traces.append(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='markers',
        marker=dict(size=9, color='steelblue'),
        name=f'y_dir ({y[0]:.2f}, {y[1]:.2f}, {y[2]:.2f})',
    ))

    arc = missing_arc_length(y_dir, alpha_deg, goniometer_axis)
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
    y_dir: np.ndarray,
    alpha_deg: float,
    volume_shape: tuple = (64, 64, 64),
    goniometer_axis: Optional[np.ndarray] = None,
):
    """Matplotlib figure with three central slices of the 3-D Fourier mask.

    Green = measured, purple = missing.  The three panels show the XY, XZ, YZ
    planes through the DC component (index 0 in FFT convention, i.e. centre
    of the shifted Fourier volume).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    mask = missing_wedge_mask_3d(y_dir, alpha_deg, volume_shape, goniometer_axis)
    # fftshift so DC is at the centre of each slice
    mask_s = np.fft.fftshift(mask)

    cx, cy, cz = [s // 2 for s in volume_shape]
    slices = {
        'XY  (z=0)': mask_s[:, :, cz],
        'XZ  (y=0)': mask_s[:, cy, :],
        'YZ  (x=0)': mask_s[cx, :, :],
    }

    cmap = mcolors.ListedColormap(['#7b2d8b', '#4caf50'])  # purple=missing, green=measured
    arc = missing_arc_length(y_dir, alpha_deg, goniometer_axis)
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

    y_str = f'({y_dir[0]:.2f},{y_dir[1]:.2f},{y_dir[2]:.2f})'
    g_label = '' if goniometer_axis is None else f'  g={np.round(goniometer_axis,2)}'
    fig.suptitle(
        f'Fourier mask slices  |  y_dir={y_str}  α={alpha_deg:.0f}°{g_label}\n'
        f'missing arc={np.degrees(arc):.1f}°  ({pct:.0f}% of great circle)',
        fontsize=11,
    )
    plt.colorbar(im, ax=axes[-1], fraction=0.046,
                 ticks=[0.25, 0.75]).set_ticklabels(['missing', 'measured'])
    plt.tight_layout()
    return fig
