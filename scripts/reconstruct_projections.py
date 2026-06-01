"""
Reconstruct 2D SAXS detector images from mumott qbin HDF5 files.

Each output image assembles the 79 q-rings (one per qbin file) and 8 azimuthal
arc segments (Friedel-expanded to 16) into a 2D (qx, qy) Cartesian image via
linear interpolation.  Pixels with no data coverage are NaN.

Output: single HDF5 file with shape (N_proj, J, K, res, res).
"""
import argparse
import glob
import os

import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", help="Directory where reconstructed_projections.h5 is written")
    p.add_argument("--data-dir", default="/myhome/data/smartt/shared/frogbone/",
                   help="Directory containing dataset_qbin_NNNN.h5 files")
    p.add_argument("--resolution", type=int, default=256,
                   help="Output image size (resolution x resolution pixels)")
    p.add_argument("--projection-indices", type=int, nargs="+", default=None,
                   help="Subset of projection indices to process (default: all)")
    return p.parse_args()


def load_q_values(data_files):
    qs = []
    for fpath in data_files:
        with h5py.File(fpath, "r") as h:
            qs.append(float(h["q"][...]))
    return np.array(qs)


def build_polar_coordinates(q_values, detector_angles):
    """Return (points, n_angles_full) for Friedel-expanded polar grid."""
    all_angles = np.concatenate([detector_angles, detector_angles + np.pi])
    QQ, AA = np.meshgrid(q_values, all_angles, indexing="ij")
    qx = (QQ * np.sin(AA)).ravel()
    qy = (QQ * np.cos(AA)).ravel()
    return np.column_stack([qx, qy]), len(all_angles)


def build_cartesian_grid(q_max, resolution):
    axis = np.linspace(-q_max, q_max, resolution)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    return axis, np.column_stack([gx.ravel(), gy.ravel()])


def load_projection_data(data_files, proj_idx, spatial_shape, n_arcs):
    """Load raw data for one projection from all qbin files. Returns (n_qbins, J, K, n_arcs)."""
    n_qbins = len(data_files)
    out = np.empty((n_qbins, spatial_shape[0], spatial_shape[1], n_arcs), dtype=np.float32)
    for q_idx, fpath in enumerate(data_files):
        with h5py.File(fpath, "r") as h:
            out[q_idx] = h[f"projections/{proj_idx}/data"][:]
    return out


def reconstruct_projection(all_data, tri, grid_points, q_values, n_arcs_full, spatial_shape, resolution):
    """
    Interpolate one projection's qbin data onto a Cartesian grid.

    all_data: (n_qbins, J, K, n_arcs_orig)
    Returns: (J, K, resolution, resolution) float32
    """
    n_qbins, J, K, n_arcs_orig = all_data.shape

    # Friedel expansion: mirror arcs at phi+pi give same intensity
    values_full = np.concatenate([all_data, all_data], axis=3)  # (n_qbins, J, K, n_arcs_full)

    # Reshape to (n_points, J*K) for vectorised interpolation over all scan points
    values_flat = values_full.reshape(n_qbins * n_arcs_full, J * K)  # (n_points, J*K)

    interp = LinearNDInterpolator(tri, values_flat, fill_value=np.nan)
    result = interp(grid_points)  # (resolution*resolution, J*K)

    result = result.reshape(resolution, resolution, J, K)
    result = result.transpose(2, 3, 0, 1)  # (J, K, resolution, resolution)
    return result.astype(np.float32)


def main():
    args = parse_args()

    data_files = sorted(glob.glob(os.path.join(args.data_dir, "dataset_qbin_*.h5")))
    if not data_files:
        raise FileNotFoundError(f"No dataset_qbin_*.h5 files found in {args.data_dir}")

    print(f"Found {len(data_files)} qbin files")

    q_values = load_q_values(data_files)
    q_max = q_values.max()

    with h5py.File(data_files[0], "r") as h:
        detector_angles = h["detector_angles"][:]
        n_projections = len(h["projections"])
        sample = h["projections/0/data"][:]
        spatial_shape = sample.shape[:2]   # (J, K)
        n_arcs_orig = sample.shape[2]      # 8

    projection_indices = np.array(args.projection_indices
                                  if args.projection_indices is not None
                                  else list(range(n_projections)))

    print(f"Spatial shape: {spatial_shape}, arcs: {n_arcs_orig}, "
          f"projections to process: {len(projection_indices)}/{n_projections}")

    points, n_arcs_full = build_polar_coordinates(q_values, detector_angles)
    print(f"Building Delaunay triangulation on {len(points)} scattered points …")
    tri = Delaunay(points)

    res = args.resolution
    axis, grid_points = build_cartesian_grid(q_max, res)
    print(f"Output grid: {res}x{res}, q range [{-q_max:.4f}, {q_max:.4f}]")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "reconstructed_projections.h5")

    n_saved = len(projection_indices)
    J, K = spatial_shape

    with h5py.File(out_path, "w") as out:
        out.attrs["source_data_dir"] = args.data_dir
        out.create_dataset("qx_axis", data=axis)
        out.create_dataset("qy_axis", data=axis)
        out.create_dataset("projection_indices", data=projection_indices)

        ds = out.create_dataset(
            "images",
            shape=(n_saved, J, K, res, res),
            dtype=np.float32,
            chunks=(1, J, K, res, res),
            compression="gzip",
            compression_opts=4,
        )

        for save_idx, proj_idx in enumerate(projection_indices):
            print(f"  Projection {proj_idx:3d}  ({save_idx + 1}/{n_saved})", flush=True)
            all_data = load_projection_data(data_files, proj_idx, spatial_shape, n_arcs_orig)
            ds[save_idx] = reconstruct_projection(
                all_data, tri, grid_points, q_values, n_arcs_full, spatial_shape, res
            )

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
