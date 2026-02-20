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
