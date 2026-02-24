"""
Visualization functions for tensor tomography reconstructions.

This module provides plotting utilities to evaluate and compare reconstructions:
- Projection residual plots (MSE and MAE vs measured projections in a DataContainer)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple

from mumott import DataContainer
from mumott.methods.projectors import SAXSProjector
from mumott.methods.basis_sets import SphericalHarmonics


def plot_projection_residuals(
    reconstruction: np.ndarray,
    data_container: DataContainer,
    ell_max: int = 8,
    label: str = "Reconstruction",
    figsize: Optional[Tuple[float, float]] = None,
    use_cuda: bool = False,
    ax: Optional[np.ndarray] = None,
    log_scale_mse: bool = False,
) -> Tuple[plt.Figure, np.ndarray, dict]:
    """
    Forward-project a reconstruction and plot MSE and MAE vs measured projections.

    The reconstruction is forward-projected using the geometry defined in
    ``data_container``.  The resulting predictions are then compared against the
    measured projections stored in ``data_container.data``, giving per-projection
    and per-channel MSE and MAE values.

    Parameters
    ----------
    reconstruction : np.ndarray
        Reconstructed spherical-harmonic coefficient field with shape
        ``(nx, ny, nz, n_coefficients)``.
    data_container : DataContainer
        A mumott DataContainer that provides both the geometry (used for
        forward projection) and the measured projections (used as reference).
    ell_max : int
        Maximum spherical harmonic order used during reconstruction (default: 8).
    label : str
        Label used in plot titles to identify this reconstruction.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.  If *None* a sensible default
        is chosen automatically.
    use_cuda : bool
        If ``True``, use the CUDA-accelerated projector.  Requires a compatible
        GPU and the ``SAXSProjectorCUDA`` class to be importable.
    ax : np.ndarray of Axes, optional
        Pre-existing array of 4 Axes to draw into (shape ``(2, 2)`` or flat with
        4 elements).  When supplied, ``figsize`` is ignored.  If *None* a new
        figure is created.
    log_scale_mse : bool
        If ``True``, display MSE axes (per-projection and per-channel) on a
        logarithmic y-scale (default: False).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    axes : np.ndarray
        Flat array of the four Axes objects.
    stats : dict
        Dictionary with summary statistics:

        - ``mse_per_projection`` – ndarray, shape ``(n_projections,)``
        - ``mae_per_projection`` – ndarray, shape ``(n_projections,)``
        - ``mse_per_channel``    – ndarray, shape ``(n_channels,)``
        - ``mae_per_channel``    – ndarray, shape ``(n_channels,)``
        - ``global_mse``         – float
        - ``global_mae``         – float
    """
    geometry = data_container.geometry
    measured = data_container.data  # shape: (n_proj, J, K, n_channels)

    # ------------------------------------------------------------------ #
    # Forward-project the reconstruction                                  #
    # ------------------------------------------------------------------ #
    if use_cuda:
        try:
            from mumott.methods.projectors import SAXSProjectorCUDA
            projector = SAXSProjectorCUDA(geometry)
        except Exception as exc:
            raise ImportError(
                "Could not import SAXSProjectorCUDA. "
                "Set use_cuda=False to fall back to the CPU projector."
            ) from exc
    else:
        projector = SAXSProjector(geometry)

    basis_set = SphericalHarmonics(
        ell_max=ell_max,
        probed_coordinates=geometry.probed_coordinates,
    )

    predicted = basis_set.forward(
        projector.forward(reconstruction.astype(np.float64))
    )  # shape: (n_proj, J, K, n_channels)

    # Align shapes: measured may have fewer channels (e.g. only one channel)
    n_proj = predicted.shape[0]
    n_channels = predicted.shape[-1]

    # Ensure measured has the correct number of channels
    if measured.shape[-1] != n_channels:
        raise ValueError(
            f"Measured data has {measured.shape[-1]} channel(s) but the "
            f"forward projection produced {n_channels} channel(s).  "
            "Make sure ell_max matches the reconstruction."
        )

    residuals = predicted - measured  # (n_proj, J, K, n_channels)

    # ------------------------------------------------------------------ #
    # Per-projection statistics (mean over spatial dims and channels)     #
    # ------------------------------------------------------------------ #
    mse_per_projection = np.mean(residuals ** 2, axis=(1, 2, 3))   # (n_proj,)
    mae_per_projection = np.mean(np.abs(residuals), axis=(1, 2, 3))  # (n_proj,)

    # ------------------------------------------------------------------ #
    # Per-channel statistics (mean over projections and spatial dims)     #
    # ------------------------------------------------------------------ #
    mse_per_channel = np.mean(residuals ** 2, axis=(0, 1, 2))      # (n_channels,)
    mae_per_channel = np.mean(np.abs(residuals), axis=(0, 1, 2))   # (n_channels,)

    global_mse = float(np.mean(residuals ** 2))
    global_mae = float(np.mean(np.abs(residuals)))

    stats = dict(
        mse_per_projection=mse_per_projection,
        mae_per_projection=mae_per_projection,
        mse_per_channel=mse_per_channel,
        mae_per_channel=mae_per_channel,
        global_mse=global_mse,
        global_mae=global_mae,
    )

    # ------------------------------------------------------------------ #
    # Plotting                                                            #
    # ------------------------------------------------------------------ #
    if ax is None:
        if figsize is None:
            figsize = (14, 10)
        fig, axes_2d = plt.subplots(2, 2, figsize=figsize)
        axes = axes_2d.flatten()
    else:
        axes = np.asarray(ax).flatten()
        if len(axes) < 4:
            raise ValueError("ax must contain at least 4 Axes objects.")
        fig = axes[0].get_figure()

    proj_indices = np.arange(n_proj)
    channel_indices = np.arange(n_channels)

    mse_ylabel = "MSE (log)" if log_scale_mse else "MSE"

    # --- Top-left: MSE per projection ---
    axes[0].bar(proj_indices, mse_per_projection, color="steelblue", alpha=0.8)
    axes[0].axhline(
        global_mse, color="red", linestyle="--",
        label=f"Global MSE = {global_mse:.3e}",
    )
    if log_scale_mse:
        axes[0].set_yscale("log")
    axes[0].set_xlabel("Projection index")
    axes[0].set_ylabel(mse_ylabel)
    axes[0].set_title(f"MSE per projection\n[{label}]")
    axes[0].legend(fontsize=8)

    # --- Top-right: MAE per projection ---
    axes[1].bar(proj_indices, mae_per_projection, color="darkorange", alpha=0.8)
    axes[1].axhline(
        global_mae, color="red", linestyle="--",
        label=f"Global MAE = {global_mae:.3e}",
    )
    axes[1].set_xlabel("Projection index")
    axes[1].set_ylabel("MAE")
    axes[1].set_title(f"MAE per projection\n[{label}]")
    axes[1].legend(fontsize=8)

    # --- Bottom-left: MSE per channel ---
    axes[2].bar(channel_indices, mse_per_channel, color="steelblue", alpha=0.8)
    axes[2].axhline(
        global_mse, color="red", linestyle="--",
        label=f"Global MSE = {global_mse:.3e}",
    )
    if log_scale_mse:
        axes[2].set_yscale("log")
    axes[2].set_xlabel("Detector channel")
    axes[2].set_ylabel(mse_ylabel)
    axes[2].set_title(f"MSE per channel\n[{label}]")
    axes[2].legend(fontsize=8)

    # --- Bottom-right: MAE per channel ---
    axes[3].bar(channel_indices, mae_per_channel, color="darkorange", alpha=0.8)
    axes[3].axhline(
        global_mae, color="red", linestyle="--",
        label=f"Global MAE = {global_mae:.3e}",
    )
    axes[3].set_xlabel("Detector channel")
    axes[3].set_ylabel("MAE")
    axes[3].set_title(f"MAE per channel\n[{label}]")
    axes[3].legend(fontsize=8)

    fig.suptitle(
        f"Projection residuals vs measured data — {label}\n"
        f"Global MSE = {global_mse:.3e}  |  Global MAE = {global_mae:.3e}",
        fontsize=11,
    )
    fig.tight_layout()

    return fig, axes, stats


def plot_projection_residuals_comparison(
    reconstructions: dict,
    data_container: DataContainer,
    ell_max: int = 8,
    figsize: Optional[Tuple[float, float]] = None,
    use_cuda: bool = False,
    log_scale_mse: bool = False,
) -> Tuple[plt.Figure, np.ndarray, dict]:
    """
    Overlay MSE and MAE curves for multiple reconstructions on shared axes.

    This is a convenience wrapper around :func:`plot_projection_residuals` that
    produces a single 2-row × 2-column figure comparing several reconstructions
    side-by-side using line plots instead of bars.

    Parameters
    ----------
    reconstructions : dict
        Mapping of ``label -> reconstruction_array``.  Each reconstruction must
        have shape ``(nx, ny, nz, n_coefficients)``.
    data_container : DataContainer
        DataContainer providing both geometry and measured projections.
    ell_max : int
        Maximum spherical harmonic order (default: 8).
    figsize : tuple of float, optional
        Figure size.  Defaults to ``(14, 10)``.
    use_cuda : bool
        Use CUDA-accelerated projector (default: False).
    log_scale_mse : bool
        If ``True``, display MSE axes on a logarithmic y-scale (default: False).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray  (4,)
    all_stats : dict
        Mapping of ``label -> stats dict`` (same format as returned by
        :func:`plot_projection_residuals`).
    """
    if figsize is None:
        figsize = (14, 10)

    fig, axes_2d = plt.subplots(2, 2, figsize=figsize)
    axes = axes_2d.flatten()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    all_stats = {}
    geometry = data_container.geometry
    measured = data_container.data

    for idx, (label, reconstruction) in enumerate(reconstructions.items()):
        color = colors[idx % len(colors)]

        if use_cuda:
            try:
                from mumott.methods.projectors import SAXSProjectorCUDA
                projector = SAXSProjectorCUDA(geometry)
            except Exception as exc:
                raise ImportError(
                    "Could not import SAXSProjectorCUDA. "
                    "Set use_cuda=False to fall back to the CPU projector."
                ) from exc
        else:
            projector = SAXSProjector(geometry)

        basis_set = SphericalHarmonics(
            ell_max=ell_max,
            probed_coordinates=geometry.probed_coordinates,
        )
        predicted = basis_set.forward(
            projector.forward(reconstruction.astype(np.float64))
        )

        residuals = predicted - measured
        n_proj = predicted.shape[0]
        n_channels = predicted.shape[-1]

        mse_per_projection = np.mean(residuals ** 2, axis=(1, 2, 3))
        mae_per_projection = np.mean(np.abs(residuals), axis=(1, 2, 3))
        mse_per_channel = np.mean(residuals ** 2, axis=(0, 1, 2))
        mae_per_channel = np.mean(np.abs(residuals), axis=(0, 1, 2))
        global_mse = float(np.mean(residuals ** 2))
        global_mae = float(np.mean(np.abs(residuals)))

        all_stats[label] = dict(
            mse_per_projection=mse_per_projection,
            mae_per_projection=mae_per_projection,
            mse_per_channel=mse_per_channel,
            mae_per_channel=mae_per_channel,
            global_mse=global_mse,
            global_mae=global_mae,
        )

        proj_indices = np.arange(n_proj)
        channel_indices = np.arange(n_channels)
        lw = 1.5

        axes[0].plot(
            proj_indices, mse_per_projection, color=color, lw=lw,
            marker="o", ms=4, label=f"{label} (global={global_mse:.2e})",
        )
        axes[1].plot(
            proj_indices, mae_per_projection, color=color, lw=lw,
            marker="o", ms=4, label=f"{label} (global={global_mae:.2e})",
        )
        axes[2].plot(
            channel_indices, mse_per_channel, color=color, lw=lw,
            marker="s", ms=5, label=f"{label} (global={global_mse:.2e})",
        )
        axes[3].plot(
            channel_indices, mae_per_channel, color=color, lw=lw,
            marker="s", ms=5, label=f"{label} (global={global_mae:.2e})",
        )

    mse_ylabel = "MSE (log)" if log_scale_mse else "MSE"
    if log_scale_mse:
        axes[0].set_yscale("log")
        axes[2].set_yscale("log")

    axes[0].set_xlabel("Projection index")
    axes[0].set_ylabel(mse_ylabel)
    axes[0].set_title("MSE per projection")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Projection index")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE per projection")
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Detector channel")
    axes[2].set_ylabel(mse_ylabel)
    axes[2].set_title("MSE per channel")
    axes[2].legend(fontsize=8)

    axes[3].set_xlabel("Detector channel")
    axes[3].set_ylabel("MAE")
    axes[3].set_title("MAE per channel")
    axes[3].legend(fontsize=8)

    fig.suptitle(
        "Projection residuals vs measured data — comparison",
        fontsize=12,
    )
    fig.tight_layout()

    return fig, axes, all_stats


# ─────────────────────────────────────────────────────────────────────────── #
# 3-D projection-direction plots                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def beam_directions_from_geometry(geometry) -> np.ndarray:
    """Return the beam direction for each projection in the lab frame.

    Each direction is computed as ``R_i @ p_direction_0``, where ``R_i`` is
    the rotation matrix of the *i*-th projection.

    Parameters
    ----------
    geometry : mumott.Geometry
        A mumott Geometry object (e.g. ``dc.geometry``).

    Returns
    -------
    np.ndarray, shape (n_projections, 3)
        Unit vectors pointing along the beam for each projection.
    """
    p0 = np.array(geometry.p_direction_0)
    rots = np.array(geometry.rotations)  # (n, 3, 3)
    return rots @ p0                      # (n, 3)


def _draw_lab_axes(ax, geometry, scale: float = 1.3, labels: bool = True) -> None:
    """Draw the three lab-frame basis vectors as coloured quiver arrows on *ax*."""
    origin = np.zeros(3)
    for vec, color, lbl in [
        (np.array(geometry.p_direction_0), "navy",      "beam (p₀)"),
        (np.array(geometry.j_direction_0), "darkred",   "j₀"),
        (np.array(geometry.k_direction_0), "darkgreen", "k₀"),
    ]:
        v = vec * scale
        ax.quiver(*origin, *v, color=color, linewidth=2, arrow_length_ratio=0.15)
        if labels:
            ax.text(*(v * 1.08), lbl, color=color, fontsize=8, ha="center")


def _geometry_to_xyz(geometry):
    """Convert a geometry's inner/outer angles to Cartesian (x, y, z) unit vectors."""
    inner = np.array(geometry.inner_angles)
    tilt  = np.pi / 2 - np.array(geometry.outer_angles)
    x = np.sin(tilt) * np.cos(inner)
    y = np.sin(tilt) * np.sin(inner)
    z = np.cos(tilt)
    return x, y, z


def _unit_sphere(n: int = 100):
    """Return (X, Y, Z) mesh arrays for a unit sphere."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(n), np.cos(v))
    return X, Y, Z


def plot_projection_directions(
    geometry,
    figsize: Tuple[float, float] = (7, 6),
    elev: float = 10,
    azim: float = 70,
    title: Optional[str] = None,
    axis_scale: float = 1.3,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the projection directions of a geometry on a unit sphere.

    Directions are derived from the geometry's inner/outer angles (the same
    parameterisation the instrument uses) and shown as red dots.  The three
    lab-frame axes (beam p₀, detector j₀, detector k₀) are overlaid as
    coloured arrows so the orientation of the instrument frame is visible.

    Parameters
    ----------
    geometry : mumott.Geometry
        The geometry whose projection directions should be visualised.
    figsize : tuple of float
        Figure size ``(width, height)`` in inches (default: ``(7, 6)``).
    elev : float
        Elevation angle of the 3-D view in degrees (default: 10).
    azim : float
        Azimuth angle of the 3-D view in degrees (default: 70).
    title : str, optional
        Figure title.  Auto-generated from projection count when *None*.
    axis_scale : float
        Length of the lab-frame axis arrows relative to the unit sphere
        (default: 1.3).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : mpl_toolkits.mplot3d.Axes3D
    """
    x, y, z = _geometry_to_xyz(geometry)
    n_proj = len(x)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    X, Y, Z = _unit_sphere()
    ax.plot_surface(X, Y, Z, color="b", alpha=0.07)
    ax.scatter(x, y, z, color="r", s=10, label=f"{n_proj} projections")
    _draw_lab_axes(ax, geometry, scale=axis_scale)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    if title is None:
        title = (
            f"Projection directions ({n_proj} proj)\n"
            "navy=beam p₀  |  darkred=j₀  |  darkgreen=k₀"
        )
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    return fig, ax


def plot_projection_directions_comparison(
    reference_geometry,
    new_directions,
    figsize: Tuple[float, float] = (14, 6),
    elev: float = 20,
    azim: float = 45,
    axis_scale: float = 1.3,
    new_label: Optional[str] = None,
) -> Tuple[plt.Figure, tuple]:
    """Two-panel 3-D comparison of new directions against an existing geometry.

    * **Left panel** — *new_directions* alone, with lab-frame axes.
    * **Right panel** — *new_directions* (green) overlaid on the original
      directions (red) derived from *reference_geometry*, with lab-frame axes.

    Parameters
    ----------
    reference_geometry : mumott.Geometry
        Geometry of the existing experiment (provides the original projection
        directions and the lab-frame axis orientations).
    new_directions : np.ndarray of shape (n, 3) or mumott.Geometry
        Either Cartesian (x, y, z) unit vectors for the new projection
        directions, or a mumott Geometry object whose inner/outer angles are
        used to derive those directions.
    figsize : tuple of float
        Figure size (default: ``(14, 6)``).
    elev : float
        Elevation angle for both 3-D views in degrees (default: 20).
    azim : float
        Azimuth angle for both 3-D views in degrees (default: 45).
    axis_scale : float
        Arrow scale relative to unit sphere (default: 1.3).
    new_label : str, optional
        Legend label for the new directions.  Defaults to ``"New (N pts)"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of two mpl_toolkits.mplot3d.Axes3D
        ``(ax_new, ax_comparison)``
    """
    # Accept either a bare xyz array or a Geometry object
    if hasattr(new_directions, "inner_angles"):
        xn, yn, zn = _geometry_to_xyz(new_directions)
        new_dirs_xyz = np.column_stack([xn, yn, zn])
    else:
        new_dirs_xyz = np.asarray(new_directions)

    n_new = len(new_dirs_xyz)
    if new_label is None:
        new_label = f"New ({n_new} pts)"

    x_orig, y_orig, z_orig = _geometry_to_xyz(reference_geometry)
    n_orig = len(x_orig)

    X, Y, Z = _unit_sphere()
    fig = plt.figure(figsize=figsize)

    # ── Left: new directions only ──────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, Z, color="b", alpha=0.07)
    ax1.scatter(
        new_dirs_xyz[:, 0], new_dirs_xyz[:, 1], new_dirs_xyz[:, 2],
        color="g", s=50, label=new_label,
    )
    _draw_lab_axes(ax1, reference_geometry, scale=axis_scale)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_title(
        "New Projection Directions\n"
        "navy=beam p₀  |  darkred=j₀  |  darkgreen=k₀"
    )
    ax1.legend(fontsize=8)
    ax1.view_init(elev=elev, azim=azim)

    # ── Right: original vs new ─────────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(X, Y, Z, color="b", alpha=0.07)
    ax2.scatter(
        x_orig, y_orig, z_orig,
        color="r", s=10, alpha=0.5, label=f"Original ({n_orig} pts)",
    )
    ax2.scatter(
        new_dirs_xyz[:, 0], new_dirs_xyz[:, 1], new_dirs_xyz[:, 2],
        color="g", s=50, label=new_label,
    )
    _draw_lab_axes(ax2, reference_geometry, scale=axis_scale)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.set_title(
        "Comparison: Original vs New\n"
        "The missing cone sits around the beam axis (navy arrow)"
    )
    ax2.legend(fontsize=8)
    ax2.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    return fig, (ax1, ax2)
