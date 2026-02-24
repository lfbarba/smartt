"""
Geometry utilities for creating new projection directions and synthetic data.

Functions for:
- Generating evenly distributed points on a hemisphere
- Creating Geometry objects from arbitrary directions
- Forward propagating reconstructions to create synthetic projections
- Building DataContainers with synthetic data
"""

import numpy as np
from typing import Optional, Tuple, Union
import copy

from mumott import Geometry
from mumott.core.geometry import GeometryTuple
from mumott.core.projection_stack import ProjectionStack, Projection
from mumott.core.probed_coordinates import ProbedCoordinates
from mumott.methods.projectors import SAXSProjector
from mumott.methods.basis_sets import SphericalHarmonics
from mumott.data_handling import DataContainer
from scipy.spatial.transform import Rotation as R


def fibonacci_hemisphere(
    n_points: int,
    upper: bool = True,
    missing_wedge_angle: float = 45.0,
) -> np.ndarray:
    """
    Generate approximately evenly distributed points on a hemisphere using the Fibonacci spiral method,
    respecting a missing wedge constraint.

    Points are sampled over the accessible angular range, from the equator (theta = 90°) up to
    ``missing_wedge_angle`` degrees from the beam axis (z-axis). Directions within the missing
    wedge (theta < ``missing_wedge_angle``) are excluded.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    upper : bool
        If True, generate points on the upper hemisphere (z >= 0), else lower hemisphere (z <= 0).
    missing_wedge_angle : float
        Half-opening angle of the missing wedge in degrees, measured from the z-axis (beam axis).
        Directions with polar angle smaller than this value are excluded.
        Default is 45.0°, corresponding to a ±45° tilt range.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) with (x, y, z) coordinates on the unit hemisphere.
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    points = np.zeros((n_points, 3))

    # boundary_z = cos(missing_wedge_angle) is the maximum |z| value accessible
    boundary_z = np.cos(np.radians(missing_wedge_angle))

    for i in range(n_points):
        # z increases linearly from 0 (equator) to boundary_z (missing-wedge boundary)
        z = i / (n_points - 1) * boundary_z if n_points > 1 else 0.0
        theta = np.arccos(z)
        phi = i * golden_angle
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        points[i] = [x, y, z if upper else -z]

    return points


def cartesian_to_spherical(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical coordinates (inner_angle, outer_angle).
    
    In mumott convention:
    - inner_angle (phi) is the rotation about the z-axis (azimuthal angle)
    - outer_angle (theta) is the polar angle from the z-axis (tilt angle)
    
    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (n, 3) with (x, y, z) coordinates.
        
    Returns
    -------
    inner_angles : np.ndarray
        Azimuthal angles in radians.
    outer_angles : np.ndarray
        Polar angles in radians (measured from z-axis).
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    # inner_angle (azimuthal angle, phi) - rotation in xy plane
    inner_angles = np.arctan2(y, x)
    
    # outer_angle (polar angle from z-axis, theta)
    r = np.sqrt(x**2 + y**2 + z**2)
    outer_angles = np.arccos(np.clip(z / r, -1, 1))
    
    return inner_angles, outer_angles


def create_rotation_matrix(
    inner_angle: float, 
    outer_angle: float,
    inner_axis: np.ndarray = np.array([0., 0., 1.]),
    outer_axis: np.ndarray = np.array([1., 0., 0.])
) -> np.ndarray:
    """
    Create a rotation matrix from inner and outer angles.
    
    Parameters
    ----------
    inner_angle : float
        Rotation angle about the inner axis (radians).
    outer_angle : float
        Rotation angle about the outer axis (radians).
    inner_axis : np.ndarray
        Inner rotation axis (default: z-axis).
    outer_axis : np.ndarray
        Outer rotation axis (default: x-axis).
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    R_inner = R.from_rotvec(inner_angle * inner_axis).as_matrix()
    R_outer = R.from_rotvec(outer_angle * outer_axis).as_matrix()
    return R_outer @ R_inner


def create_geometry_from_directions(
    directions: np.ndarray,
    reference_geometry: Optional[Geometry] = None,
    projection_shape: Tuple[int, int] = None,
    volume_shape: Tuple[int, int, int] = None,
    detector_angles: np.ndarray = None,
    inner_axis: np.ndarray = np.array([0., 0., 1.]),
    outer_axis: np.ndarray = np.array([1., 0., 0.]),
    j_offsets: Optional[np.ndarray] = None,
    k_offsets: Optional[np.ndarray] = None,
    copy_from_reference: bool = False,
) -> Geometry:
    """
    Create a new Geometry object from a set of projection directions.
    
    Parameters
    ----------
    directions : np.ndarray
        Array of shape (n, 3) with unit vectors representing projection directions (x, y, z).
        Alternatively, can be shape (n, 2) with (inner_angle, outer_angle) in radians.
    reference_geometry : Geometry, optional
        A reference geometry to copy settings from (projection_shape, volume_shape, detector_angles, etc.).
        If not provided, must specify projection_shape and volume_shape.
    projection_shape : tuple of int, optional
        Shape of each projection (j, k). Required if reference_geometry is None.
    volume_shape : tuple of int, optional
        Shape of the reconstruction volume (x, y, z). Required if reference_geometry is None.
    detector_angles : np.ndarray, optional
        Azimuthal angles of detector segments. If None, uses reference_geometry or default.
    inner_axis : np.ndarray
        Inner rotation axis (default: z-axis [0, 0, 1]). Only used when copy_from_reference=False and
        reference_geometry does not provide per-projection axes.
    outer_axis : np.ndarray
        Outer rotation axis (default: x-axis [1, 0, 0]). Only used when copy_from_reference=False and
        reference_geometry does not provide per-projection axes.
    j_offsets : np.ndarray, optional
        Per-projection detector j-offsets of shape (n,). Defaults to zeros.
    k_offsets : np.ndarray, optional
        Per-projection detector k-offsets of shape (n,). Defaults to zeros.
    copy_from_reference : bool
        If True and reference_geometry is provided with the same number of projections as directions,
        copy the rotation matrices, j_offsets, k_offsets, inner_axes, and outer_axes for each
        projection directly from the reference. Use this when reproducing the exact same geometry
        (e.g., for full-data synthetic projections). Defaults to False.
        
    Returns
    -------
    Geometry
        A new Geometry object with the specified projection directions.
    """
    # Convert directions to inner/outer angles if given as Cartesian coordinates
    if directions.shape[1] == 3:
        inner_angles, outer_angles = cartesian_to_spherical(directions)
    else:
        inner_angles = np.asarray(directions[:, 0])
        outer_angles = np.asarray(directions[:, 1])
    
    n_projections = len(inner_angles)
    
    # Validate copy_from_reference
    if copy_from_reference:
        if reference_geometry is None:
            raise ValueError("copy_from_reference=True requires reference_geometry to be provided.")
        if len(reference_geometry.inner_angles) != n_projections:
            raise ValueError(
                f"copy_from_reference=True requires reference_geometry to have the same number of "
                f"projections as directions ({n_projections}), but got {len(reference_geometry.inner_angles)}."
            )

    # Create new geometry
    geometry = Geometry()
    
    # Copy settings from reference geometry if provided
    if reference_geometry is not None:
        geometry._projection_shape = reference_geometry.projection_shape.copy()
        geometry._volume_shape = reference_geometry.volume_shape.copy()
        geometry._detector_angles = reference_geometry.detector_angles.copy()
        geometry._two_theta = reference_geometry.two_theta.copy()
        geometry._full_circle_covered = reference_geometry.full_circle_covered
        # Copy system vectors
        geometry._p_direction_0 = reference_geometry.p_direction_0.copy()
        geometry._j_direction_0 = reference_geometry.j_direction_0.copy()
        geometry._k_direction_0 = reference_geometry.k_direction_0.copy()
        geometry._detector_direction_origin = reference_geometry.detector_direction_origin.copy()
        geometry._detector_direction_positive_90 = reference_geometry.detector_direction_positive_90.copy()
    else:
        if projection_shape is None or volume_shape is None:
            raise ValueError("Must provide either reference_geometry or both projection_shape and volume_shape")
        geometry._projection_shape = np.array(projection_shape)
        geometry._volume_shape = np.array(volume_shape)
        if detector_angles is not None:
            geometry._detector_angles = detector_angles
    
    # Resolve per-projection offset arrays
    _j_offsets = np.zeros(n_projections) if j_offsets is None else np.asarray(j_offsets, dtype=float)
    _k_offsets = np.zeros(n_projections) if k_offsets is None else np.asarray(k_offsets, dtype=float)

    # Add projections with the new geometry
    for i in range(n_projections):
        if copy_from_reference:
            # Copy rotation and axes directly from the reference — preserves the exact physical
            # setup including stage-axis conventions and any per-projection offset calibration.
            rotation    = np.array(reference_geometry.rotations[i])
            j_off       = reference_geometry.j_offsets[i]
            k_off       = reference_geometry.k_offsets[i]
            proj_inner_axis = np.array(reference_geometry.inner_axes[i])
            proj_outer_axis = np.array(reference_geometry.outer_axes[i])
        else:
            # Use provided or default axes and compute rotation from angles
            proj_inner_axis = np.array(reference_geometry.inner_axes[i]) \
                if (reference_geometry is not None and hasattr(reference_geometry, 'inner_axes')
                    and i < len(reference_geometry.inner_axes)) \
                else inner_axis
            proj_outer_axis = np.array(reference_geometry.outer_axes[i]) \
                if (reference_geometry is not None and hasattr(reference_geometry, 'outer_axes')
                    and i < len(reference_geometry.outer_axes)) \
                else outer_axis
            rotation = create_rotation_matrix(
                inner_angles[i], outer_angles[i],
                proj_inner_axis, proj_outer_axis
            )
            j_off = _j_offsets[i]
            k_off = _k_offsets[i]
        
        geom_tuple = GeometryTuple(
            rotation=rotation,
            j_offset=j_off,
            k_offset=k_off,
            inner_angle=inner_angles[i],
            outer_angle=outer_angles[i],
            inner_axis=proj_inner_axis,
            outer_axis=proj_outer_axis
        )
        geometry.append(geom_tuple)
    
    return geometry


def create_synthetic_projections(
    reconstruction: np.ndarray,
    new_geometry: Geometry,
    ell_max: int = 8,
    reference_data_shape: Tuple[int, ...] = None,
    create_data_container: bool = False,
    reference_dc: 'DataContainer' = None,
) -> Union[np.ndarray, ProjectionStack]:
    """
    Create synthetic projections from a reconstruction using forward propagation.
    
    Parameters
    ----------
    reconstruction : np.ndarray
        The reconstruction volume with spherical harmonic coefficients.
        Shape: (x, y, z, n_coefficients).
    new_geometry : Geometry
        The new geometry with projection directions.
    ell_max : int
        Maximum order of spherical harmonics (default: 8).
    reference_data_shape : tuple, optional
        Shape of reference data (n_projections, j, k, n_detector_segments) for determining
        the output shape. If None, inferred from geometry.
    create_data_container : bool
        If True, creates and returns a ProjectionStack instead of just the projections array.
    reference_dc : DataContainer, optional
        Reference DataContainer to copy metadata from.
        
    Returns
    -------
    np.ndarray or ProjectionStack
        Synthetic projections array of shape (n_projections, j, k, n_detector_segments),
        or a ProjectionStack if create_data_container=True.
    """
    # Create projector with new geometry
    projector = SAXSProjector(new_geometry)
    
    # Create basis set with new geometry's probed coordinates
    basis_set = SphericalHarmonics(ell_max=ell_max, probed_coordinates=new_geometry.probed_coordinates)
    
    # Perform forward propagation
    # First: project the volume (integrates along rays)
    projected_coefficients = projector.forward(reconstruction.astype(np.float64))
    
    # Second: apply basis set forward to get detector values
    synthetic_projections = basis_set.forward(projected_coefficients)
    
    if not create_data_container:
        return synthetic_projections
    
    # Create ProjectionStack
    n_projections = len(new_geometry.inner_angles)
    
    # Create projection stack
    projection_stack = ProjectionStack()
    
    for i in range(n_projections):
        # Get geometry info for this projection
        geom_tuple = GeometryTuple(
            rotation=new_geometry.rotations[i],
            j_offset=new_geometry.j_offsets[i],
            k_offset=new_geometry.k_offsets[i],
            inner_angle=new_geometry.inner_angles[i],
            outer_angle=new_geometry.outer_angles[i],
            inner_axis=new_geometry.inner_axes[i],
            outer_axis=new_geometry.outer_axes[i]
        )
        
        # Create projection with synthetic data
        projection = Projection(
            data=synthetic_projections[i],
            weights=np.ones_like(synthetic_projections[i]),
            **geom_tuple._asdict()
        )
        projection_stack.append(projection)
    
    return projection_stack


def generate_geometry_and_projections(
    reconstruction: np.ndarray,
    directions: np.ndarray,
    reference_geometry: Geometry,
    ell_max: int = 8,
    return_data_container: bool = False,
    copy_from_reference: bool = False,
    j_offsets: Optional[np.ndarray] = None,
    k_offsets: Optional[np.ndarray] = None,
) -> Tuple[Geometry, Union[np.ndarray, ProjectionStack]]:
    """
    Convenience function to create new geometry and synthetic projections in one call.
    
    Parameters
    ----------
    reconstruction : np.ndarray
        The reconstruction volume with spherical harmonic coefficients.
        Shape: (x, y, z, n_coefficients).
    directions : np.ndarray
        Array of shape (n, 3) with unit vectors representing projection directions (x, y, z).
        Alternatively, can be shape (n, 2) with (inner_angle, outer_angle) in radians.
    reference_geometry : Geometry
        A reference geometry to copy settings from.
    ell_max : int
        Maximum order of spherical harmonics (default: 8).
    return_data_container : bool
        If True, returns a ProjectionStack instead of just the projections array.
    copy_from_reference : bool
        If True, rotation matrices, j/k-offsets, and per-projection axes are copied directly
        from reference_geometry (requires same number of projections as directions).
        Use this when reproducing the exact same geometry for forward-projection comparisons.
    j_offsets : np.ndarray, optional
        Per-projection detector j-offsets of shape (n,). Ignored when copy_from_reference=True.
    k_offsets : np.ndarray, optional
        Per-projection detector k-offsets of shape (n,). Ignored when copy_from_reference=True.
        
    Returns
    -------
    new_geometry : Geometry
        The newly created geometry.
    synthetic_projections : np.ndarray or ProjectionStack
        The synthetic projections.
    """
    # Create new geometry
    new_geometry = create_geometry_from_directions(
        directions=directions,
        reference_geometry=reference_geometry,
        copy_from_reference=copy_from_reference,
        j_offsets=j_offsets,
        k_offsets=k_offsets,
    )
    
    # Create synthetic projections
    synthetic_projections = create_synthetic_projections(
        reconstruction=reconstruction,
        new_geometry=new_geometry,
        ell_max=ell_max,
        create_data_container=return_data_container,
    )
    
    return new_geometry, synthetic_projections


def create_synthetic_data_container(
    geometry: Geometry,
    projection_stack: ProjectionStack,
    reference_dc: DataContainer = None,
    save_path: Optional[str] = None,
) -> DataContainer:
    """
    Create a DataContainer from a geometry and projection stack, optionally saving it to disk.
    
    Parameters
    ----------
    geometry : Geometry
        The geometry object with projection directions.
    projection_stack : ProjectionStack
        The projection stack containing the synthetic projection data.
    reference_dc : DataContainer, optional
        A reference DataContainer to copy additional metadata from (e.g., detector_angles, two_theta).
    save_path : str, optional
        Path to save the DataContainer as an HDF5 file. Must end with '.h5'.
        If None, the DataContainer is not saved.
        
    Returns
    -------
    DataContainer
        A new DataContainer with the synthetic data.
    """
    # Create an empty DataContainer
    synthetic_dc = DataContainer(data_path=None, data_type=None)
    
    # First, append all projections (this will set up the geometry with correct projection_shape)
    for i in range(len(projection_stack)):
        old_proj = projection_stack[i]
        
        # Create a new Projection object with the same data and geometry
        new_proj = Projection(
            data=old_proj.data.copy(),
            weights=old_proj.weights.copy(),
            rotation=old_proj.rotation,
            j_offset=old_proj.j_offset,
            k_offset=old_proj.k_offset,
            inner_angle=old_proj.inner_angle,
            outer_angle=old_proj.outer_angle,
            inner_axis=old_proj.inner_axis,
            outer_axis=old_proj.outer_axis
        )
        synthetic_dc.append(new_proj)
    
    # Now copy additional geometry settings from reference or from the provided geometry
    if reference_dc is not None:
        # Copy geometry configuration from reference
        synthetic_dc.geometry._projection_shape = reference_dc.geometry.projection_shape.copy()
        synthetic_dc.geometry._volume_shape = reference_dc.geometry.volume_shape.copy()
        synthetic_dc.geometry._detector_angles = reference_dc.geometry.detector_angles.copy()
        synthetic_dc.geometry._two_theta = reference_dc.geometry.two_theta.copy()
        synthetic_dc.geometry._full_circle_covered = reference_dc.geometry.full_circle_covered
        synthetic_dc.geometry._p_direction_0 = reference_dc.geometry.p_direction_0.copy()
        synthetic_dc.geometry._j_direction_0 = reference_dc.geometry.j_direction_0.copy()
        synthetic_dc.geometry._k_direction_0 = reference_dc.geometry.k_direction_0.copy()
        synthetic_dc.geometry._detector_direction_origin = reference_dc.geometry.detector_direction_origin.copy()
        synthetic_dc.geometry._detector_direction_positive_90 = reference_dc.geometry.detector_direction_positive_90.copy()
    else:
        # Copy from provided geometry
        synthetic_dc.geometry._projection_shape = geometry.projection_shape.copy()
        synthetic_dc.geometry._volume_shape = geometry.volume_shape.copy()
        synthetic_dc.geometry._detector_angles = geometry.detector_angles.copy()
        synthetic_dc.geometry._two_theta = geometry.two_theta.copy()
        synthetic_dc.geometry._full_circle_covered = geometry.full_circle_covered
        synthetic_dc.geometry._p_direction_0 = geometry.p_direction_0.copy()
        synthetic_dc.geometry._j_direction_0 = geometry.j_direction_0.copy()
        synthetic_dc.geometry._k_direction_0 = geometry.k_direction_0.copy()
        synthetic_dc.geometry._detector_direction_origin = geometry.detector_direction_origin.copy()
        synthetic_dc.geometry._detector_direction_positive_90 = geometry.detector_direction_positive_90.copy()
    
    # Save to disk if path is provided
    if save_path is not None:
        if not save_path.endswith('.h5'):
            raise ValueError("save_path must end with '.h5'")
        synthetic_dc.write(save_path)
        print(f"DataContainer saved to: {save_path}")
    
    return synthetic_dc


def generate_and_save_synthetic_data(
    reconstruction: np.ndarray,
    directions: np.ndarray,
    reference_dc: DataContainer,
    ell_max: int = 8,
    save_path: Optional[str] = None,
) -> DataContainer:
    """
    Complete pipeline: generate geometry, create synthetic projections, build DataContainer, and optionally save.
    
    This is a convenience function that combines all steps into one call.
    
    Parameters
    ----------
    reconstruction : np.ndarray
        The reconstruction volume with spherical harmonic coefficients.
        Shape: (x, y, z, n_coefficients).
    directions : np.ndarray
        Array of shape (n, 3) with unit vectors representing projection directions (x, y, z).
        Alternatively, can be shape (n, 2) with (inner_angle, outer_angle) in radians.
    reference_dc : DataContainer
        A reference DataContainer to copy settings from.
    ell_max : int
        Maximum order of spherical harmonics (default: 8).
    save_path : str, optional
        Path to save the DataContainer as an HDF5 file. Must end with '.h5'.
        If None, the DataContainer is not saved.
        
    Returns
    -------
    DataContainer
        A new DataContainer with the synthetic data and geometry.
        
    Example
    -------
    >>> # Generate 20 evenly sampled projections on the top hemisphere
    >>> directions = fibonacci_hemisphere(20, upper=True)
    >>> synthetic_dc = generate_and_save_synthetic_data(
    ...     reconstruction=results['x'],
    ...     directions=directions,
    ...     reference_dc=dc,
    ...     ell_max=8,
    ...     save_path='synthetic_projections.h5'
    ... )
    """
    # Step 1: Create geometry and synthetic projections
    new_geometry, projection_stack = generate_geometry_and_projections(
        reconstruction=reconstruction,
        directions=directions,
        reference_geometry=reference_dc.geometry,
        ell_max=ell_max,
        return_data_container=True,  # Get ProjectionStack
    )
    
    # Step 2: Create DataContainer and optionally save
    synthetic_dc = create_synthetic_data_container(
        geometry=new_geometry,
        projection_stack=projection_stack,
        reference_dc=reference_dc,
        save_path=save_path,
    )
    
    return synthetic_dc
