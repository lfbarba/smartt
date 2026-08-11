"""Migrate a mumott 0.2-era raw HDF5 dataset to the mumott >=2.0 DataContainer format.

Background
----------
mumott 0.2's ``DataContainer`` loaded "raw" per-projection HDF5 files
(``projections/<i>/{data,diode,weights,rotations,tilts,offset_j,offset_k}``
plus a root-level ``detector_angles`` dataset) and required a follow-up
``DataContainer.transform(TransformParameters(...))`` call to align the
beamline's raw axis/angle conventions with the reconstruction convention.
That transform only ever did three things to the data:

1. permute/flip the ``(j, k, detector-segment)`` axes of ``data``/``diode``/
   ``weights`` according to ``data_sorting`` and ``data_index_origin``;
2. flip the sign of ``principal_rotation``/``secondary_rotation`` according
   to the ``*_right_handed`` flags;
3. remap ``detector_angles`` via
   ``phi -> angle(exp(-1**a * i*phi) * exp(i*atan2(detector_angle_0)))``
   and flip the sign of the alignment offsets.

Current mumott (>=2.0) no longer has ``TransformParameters``/``transform()``.
Its ``DataContainer`` instead reads a per-projection 3x3 ``rotation_matrix``
directly (or, if absent, builds one from ``inner_angle``/``outer_angle`` about
default axes ``inner_axis=(0,0,-1)``, ``outer_axis=(1,0,0)``). Comparing
mumott 0.2's ``ProjectionParameters._calculate_basis_vectors`` with current
mumott's default ``p_direction_0=(0,1,0)``, ``j_direction_0=(1,0,0)``,
``k_direction_0=(0,0,1)`` shows the *zero-rotation* basis is identical between
versions, and that the old ``(vector_j, vector_p, vector_k)`` triplet is
exactly the image, under the sought rotation matrix ``R``, of
``(j_direction_0, p_direction_0, k_direction_0)``. So::

    R = column_stack([vector_j, vector_p, vector_k])

reproduces the exact old geometry in the new schema, without needing to
reverse-engineer inner/outer angle conventions.

This script performs step 1-3 (the "transform") using the *actual* mumott
0.2 library (must be run once, from a throwaway venv with ``mumott==0.2``
installed, to dump an intermediate .npz — see ``dump_legacy_stack.py`` /
the ``--dump`` mode below run under that venv), then (under the current
mumott environment) builds ``R`` per projection and writes a mumott
>=2.0-compatible .h5 file.

Usage
-----
Step 1 (inside a mumott==0.2 venv)::

    python migrate_legacy_mumott_h5.py dump SRC_H5 OUT_NPZ \\
        --data-sorting 0 1 2 --data-index-origin 0 0 \\
        --principal-right-handed --secondary-right-handed \\
        --detector-angle-0 0 1 --no-detector-angle-right-handed \\
        --offset-positive-j --offset-positive-k

Step 2 (inside the current mumott environment)::

    python migrate_legacy_mumott_h5.py build OUT_NPZ DEST_H5
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def cmd_dump(args: argparse.Namespace) -> None:
    from mumott.data_handling import DataContainer  # mumott==0.2
    from mumott.data_handling.transform_parameters import TransformParameters
    from os.path import dirname, basename

    dc = DataContainer(data_path=dirname(args.src) or ".",
                        data_filename=basename(args.src),
                        data_type="h5")
    print(dc)
    if not dc.angles_in_radians:
        raise RuntimeError(
            "Angles were not detected as radians. Re-run with an explicit "
            "degrees_to_radians() call added to this script before transform()."
        )

    tp = TransformParameters(
        data_sorting=tuple(args.data_sorting),
        data_index_origin=tuple(args.data_index_origin),
        principal_rotation_right_handed=args.principal_right_handed,
        secondary_rotation_right_handed=args.secondary_right_handed,
        detector_angle_0=tuple(args.detector_angle_0),
        detector_angle_right_handed=args.detector_angle_right_handed,
        offset_positive=(args.offset_positive_j, args.offset_positive_k),
    )
    dc.transform(tp)
    stack = dc.stack

    data_list = [np.asarray(f.data) for f in stack]
    diode_list = [np.asarray(f.diode) for f in stack]
    weights_list = [np.asarray(f.weights) for f in stack]

    np.savez(
        args.out,
        principal_rotation=np.asarray(stack.principal_rotation, dtype=np.float64),
        secondary_rotation=np.asarray(stack.secondary_rotation, dtype=np.float64),
        detector_angles=np.asarray(stack.detector_angles, dtype=np.float64),
        j_offset=np.asarray(stack.j_offset, dtype=np.float64),
        k_offset=np.asarray(stack.k_offset, dtype=np.float64),
        volume_shape=np.asarray(stack.volume_shape),
        data=np.array(data_list, dtype=object),
        diode=np.array(diode_list, dtype=object),
        weights=np.array(weights_list, dtype=object),
    )
    print(f"Saved intermediate arrays to {args.out}")


def _rotation_matrices(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Reproduce mumott 0.2 ``ProjectionParameters._calculate_basis_vectors``
    and re-express the result as a per-projection rotation matrix in the
    convention expected by current mumott (see module docstring)."""
    vector_p = np.stack([np.sin(-alpha) * np.cos(beta),
                         np.cos(alpha) * np.cos(beta),
                         np.sin(-beta)], axis=-1)
    vector_j = np.stack([np.cos(alpha),
                         np.sin(alpha),
                         np.zeros_like(alpha)], axis=-1)
    vector_k = np.stack([np.sin(beta) * np.sin(-alpha),
                         np.sin(beta) * np.cos(alpha),
                         np.cos(beta)], axis=-1)
    # columns = images of (j_direction_0, p_direction_0, k_direction_0) = (ex, ey, ez)
    return np.stack([vector_j, vector_p, vector_k], axis=-1)


# Some beamline azimuthal-integration pipelines fill invalid/masked q-bins with
# a float32-max sentinel instead of NaN. This is a *finite* value, so it slips
# right past `nonfinite_replacement_value` and, squared in the loss, silently
# blows up the reconstruction. The 2D per-pixel `weights` mask on its own can't
# exclude it either, since the corruption is specific to individual detector
# channels/projections, not whole pixels shared across the dataset.
SENTINEL_THRESHOLD = 1e30


def cmd_build(args: argparse.Namespace) -> None:
    import h5py

    npz = np.load(args.npz, allow_pickle=True)
    alpha = npz["principal_rotation"]
    beta = npz["secondary_rotation"]
    detector_angles = npz["detector_angles"]
    j_offset = npz["j_offset"]
    k_offset = npz["k_offset"]
    data = npz["data"]
    diode = npz["diode"]
    weights = npz["weights"]

    n = len(alpha)
    rotations = _rotation_matrices(alpha, beta)

    if args.volume_shape is not None:
        volume_shape = np.array(args.volume_shape)
    else:
        volume_shape = npz["volume_shape"]
        # mumott 0.2 itself only ever fell back to this (j, j, k) shape (see
        # DataContainer._h5_to_stack) when no volume_shape was stored in the
        # raw file. Some legacy files stored something else instead (observed:
        # a value whose last entry was actually the projection count, not a
        # z-extent) — that's not a real reconstruction volume, so recompute
        # the same fallback mumott 0.2 would have used, unless overridden
        # explicitly via --volume-shape.
        proj_shape = np.asarray(data[0]).shape[:2]
        expected = np.array([proj_shape[0], proj_shape[0], proj_shape[1]])
        if not np.array_equal(volume_shape, expected):
            print(f"Warning: stored volume_shape {volume_shape.tolist()} does not match "
                  f"the (j, j, k) fallback {expected.tolist()} mumott 0.2 itself would have "
                  "used for a file with no explicit volume_shape. Using the fallback instead "
                  "— pass --volume-shape to override.")
            volume_shape = expected

    total_sentinel = 0
    with h5py.File(args.dest, "w") as f:
        f.create_dataset("detector_angles", data=detector_angles)
        f.create_dataset("volume_shape", data=volume_shape)
        grp = f.create_group("projections")
        for i in range(n):
            p = grp.create_group(str(i))
            data_i = np.asarray(data[i], dtype=np.float64)
            sentinel = data_i >= SENTINEL_THRESHOLD
            n_sentinel = int(sentinel.sum())
            if n_sentinel:
                total_sentinel += n_sentinel
                data_i = np.where(sentinel, 0.0, data_i)

            weights_i = np.broadcast_to(
                np.asarray(weights[i], dtype=bool)[..., np.newaxis], data_i.shape
            ).copy()
            weights_i &= ~sentinel

            p.create_dataset("data", data=data_i.astype(np.float32))
            p.create_dataset("diode", data=np.asarray(diode[i], dtype=np.float32))
            p.create_dataset("weights", data=weights_i)
            p.create_dataset("j_offset", data=np.float64(j_offset[i]))
            p.create_dataset("k_offset", data=np.float64(k_offset[i]))
            p.create_dataset("rotation_matrix", data=rotations[i])
    print(f"Wrote {n} projections to {args.dest}")
    if total_sentinel:
        print(f"Masked out {total_sentinel} float32-max sentinel entries "
              f"(threshold={SENTINEL_THRESHOLD:g}) as invalid/zero-weighted.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dump = sub.add_parser("dump", help="Run under mumott==0.2 to apply the legacy transform.")
    p_dump.add_argument("src")
    p_dump.add_argument("out")
    p_dump.add_argument("--data-sorting", type=int, nargs=3, default=(0, 1, 2))
    p_dump.add_argument("--data-index-origin", type=int, nargs=2, default=(0, 0))
    p_dump.add_argument("--principal-right-handed", action="store_true", default=True)
    p_dump.add_argument("--no-principal-right-handed", dest="principal_right_handed",
                        action="store_false")
    p_dump.add_argument("--secondary-right-handed", action="store_true", default=True)
    p_dump.add_argument("--no-secondary-right-handed", dest="secondary_right_handed",
                        action="store_false")
    p_dump.add_argument("--detector-angle-0", type=int, nargs=2, default=(1, 0))
    p_dump.add_argument("--detector-angle-right-handed", action="store_true", default=True)
    p_dump.add_argument("--no-detector-angle-right-handed", dest="detector_angle_right_handed",
                        action="store_false")
    p_dump.add_argument("--offset-positive-j", action="store_true", default=True)
    p_dump.add_argument("--no-offset-positive-j", dest="offset_positive_j", action="store_false")
    p_dump.add_argument("--offset-positive-k", action="store_true", default=True)
    p_dump.add_argument("--no-offset-positive-k", dest="offset_positive_k", action="store_false")
    p_dump.set_defaults(func=cmd_dump)

    p_build = sub.add_parser("build", help="Run under current mumott to write the new .h5 file.")
    p_build.add_argument("npz")
    p_build.add_argument("dest")
    p_build.add_argument("--volume-shape", type=int, nargs=3, default=None,
                         help="Override the reconstruction volume shape (nx, ny, nz). "
                              "Defaults to the mumott 0.2 (j, j, k) fallback if the "
                              "stored value looks bogus, else the stored value.")
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
