"""Forward-project a ground-truth SH volume into a saved ``DataContainer``."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from mumott.core.geometry import Geometry
from mumott.data_handling import DataContainer

from mumott_al import (
    fibonacci_hemisphere,
    create_geometry_from_directions,
    generate_geometry_and_projections,
    create_synthetic_data_container,
)
from smartt.saxs_naf.cache import save_recon


def build_datacontainer_from_gt(
    ground_truth: np.ndarray,
    reference_geometry: Geometry,
    output_dir: str,
    *,
    dataset_name: str = "dataset_from_npy",
    gt_name: str = "ground_truth",
    n_fibonacci: int = 500,
    ell_max: int = 8,
    missing_wedge_angle: float = 45.0,
    upper: bool = True,
    half_space: str = "z",
    gt_params: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Forward-project ``ground_truth`` and save a ``DataContainer`` + GT cache.

    Pipeline (identical to ``SyntheticDataContainers.ipynb`` from step 5 on):

    1. Generate ``n_fibonacci`` Fibonacci-hemisphere directions honouring the
       missing wedge.
    2. Build a :class:`~mumott.core.geometry.Geometry` from those directions,
       copying detector metadata (detector angles, two-theta, system vectors)
       from ``reference_geometry`` but using the ground truth's own shape.
    3. Forward-project ``ground_truth`` into a ``ProjectionStack``.
    4. Wrap it in a ``DataContainer`` and write it to
       ``{output_dir}/{dataset_name}.h5``.
    5. Persist ``ground_truth`` via :func:`smartt.saxs_naf.cache.save_recon`
       (``.npy`` + JSON sidecar) under ``cache_dir`` so benchmark notebooks can
       reload it with ``load_recon``.

    Parameters
    ----------
    ground_truth : ndarray, shape ``(X, Y, Z, n_coeffs)``
        Spherical-harmonics ground-truth volume, ready to forward-project.
    reference_geometry : Geometry
        Geometry whose detector metadata is copied into the new geometry.
    output_dir : str
        Directory for the ``.h5`` DataContainer (created if absent).
    dataset_name : str
        Basename (no extension) for the saved DataContainer.
    gt_name : str
        Logical name used for the ground-truth cache entry.
    n_fibonacci : int
        Number of Fibonacci-hemisphere projection directions.
    ell_max : int
        Maximum SH order; must be consistent with ``ground_truth.shape[-1]``.
    missing_wedge_angle : float
        Half-opening angle (degrees) of the missing wedge.
    upper : bool
        Sample the positive (``True``) or negative half-sphere.
    half_space : str
        Axis defining the hemisphere, ``'z'`` (default) or ``'y'``.
    gt_params : dict, optional
        Provenance parameters stored in the ground-truth sidecar.  A default
        dict describing the geometry is used/extended if not supplied.
    cache_dir : str, optional
        Directory for the ground-truth cache.  Defaults to
        ``{output_dir}/recon_cache``.
    verbose : bool
        Print progress if ``True``.

    Returns
    -------
    dict with keys ``data_container``, ``geometry``, ``directions``,
    ``save_path``, ``gt_path``, ``projection_stack``.
    """
    if ground_truth.ndim != 4:
        raise ValueError(
            f"ground_truth must be 4-D (X, Y, Z, n_coeffs); got shape {ground_truth.shape}"
        )

    n_coeffs = ground_truth.shape[-1]
    expected_c = sum(2 * ell + 1 for ell in range(0, int(ell_max) + 1, 2))
    if n_coeffs != expected_c:
        raise ValueError(
            f"ground_truth has {n_coeffs} coefficients but ell_max={ell_max} "
            f"expects {expected_c}. Pass a matching ell_max."
        )

    os.makedirs(output_dir, exist_ok=True)
    cache_dir = cache_dir or os.path.join(output_dir, "recon_cache")
    save_path = os.path.join(output_dir, f"{dataset_name}.h5")

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Fibonacci-hemisphere directions ------------------------------------
    directions = fibonacci_hemisphere(
        n_fibonacci,
        upper=upper,
        missing_wedge_angle=missing_wedge_angle,
        half_space=half_space,
    )
    _log(f"[1] Generated {len(directions)} Fibonacci directions")

    # 2. Geometry from directions (shapes from the ground truth) ------------
    new_geometry = create_geometry_from_directions(
        directions=directions,
        reference_geometry=reference_geometry,
        volume_shape=ground_truth.shape[:3],
        projection_shape=ground_truth.shape[1:3],
    )
    _log(
        f"[2] Geometry: {len(new_geometry.inner_angles)} projections, "
        f"volume_shape={tuple(new_geometry.volume_shape)}, "
        f"projection_shape={tuple(new_geometry.projection_shape)}"
    )

    # 3-4. Forward projection -> ProjectionStack ----------------------------
    _, projection_stack = generate_geometry_and_projections(
        reconstruction=ground_truth.astype(np.float64),
        directions=directions,
        reference_geometry=new_geometry,
        ell_max=ell_max,
        return_data_container=True,
    )
    _log(f"[3] Projection stack: {projection_stack.data.shape}")

    # 5. Wrap in a DataContainer and save -----------------------------------
    data_container = create_synthetic_data_container(
        geometry=new_geometry,
        projection_stack=projection_stack,
        save_path=save_path,
    )
    _log(f"[4] DataContainer saved to: {save_path}")

    # 6. Persist ground truth in the save_recon cache format ----------------
    params = dict(
        method="npy_ground_truth",
        dataset_name=dataset_name,
        ell_max=ell_max,
        n_fibonacci=n_fibonacci,
        missing_wedge_angle=missing_wedge_angle,
        upper=upper,
        half_space=half_space,
    )
    if gt_params:
        params.update(gt_params)
    gt_path = save_recon(cache_dir, gt_name, ground_truth.astype(np.float32), params)
    _log(f"[5] Ground truth saved to: {gt_path}")

    return dict(
        data_container=data_container,
        geometry=new_geometry,
        directions=directions,
        projection_stack=projection_stack,
        save_path=save_path,
        gt_path=gt_path,
        gt_params=params,
    )
