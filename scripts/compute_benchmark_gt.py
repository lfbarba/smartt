"""Compute and cache the benchmark ground-truth reconstruction.

Run this script from a terminal (not Jupyter) to avoid kernel timeouts.
The combined_dc reconstruction on a 141^3 volume with 567 projections takes
~80 minutes with 20 LBFGS iterations. This script saves the result to the
same cache path that saxs_naf_benchmark.ipynb expects, so the notebook
loads it on next run without recomputing.

Usage
-----
    cd /myhome/smartt/notebooks   # so relative CACHE_DIR resolves correctly
    python ../scripts/compute_benchmark_gt.py

Optional flags
--------------
    --gt-method sh|gk        default: sh
    --ell-max N              default: 8
    --n-iterations N         default: 20
    --laplacian-weight W     default: 0.1
    --cache-dir PATH         default: cache/saxs_naf_benchmark
    --force                  recompute even if cached
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mumott.data_handling import DataContainer

from smartt.saxs_naf.cache import load_recon, save_recon
from smartt.saxs_naf.metrics import compute_ground_truth

DATA_DIR = Path("/myhome/data/smartt/shared/b411")


def build_combined_dc():
    combined_dc = DataContainer(str(DATA_DIR / "dataset_b411R_inf_1_0.220_1.900.h5"))
    combined_dc.geometry.full_circle_covered = False
    for proj in combined_dc.projections:
        proj.diode   = np.pad(proj.diode,   ((0, 0), (27, 28)),        mode="constant", constant_values=1)
        proj.data    = np.pad(proj.data,    ((0, 0), (27, 28), (0, 0)), mode="constant", constant_values=0)
        proj.weights = np.pad(proj.weights, ((0, 0), (27, 28), (0, 0)), mode="constant", constant_values=0)

    dc_rem = DataContainer(str(DATA_DIR / "dataset_b411R_inf1_remount_0.220_1.900.h5"))
    dc_rem.geometry.full_circle_covered = False
    n_rem = len(dc_rem.projections)
    for _ in range(n_rem):
        dc_rem.projections[0].data *= 0.75
        frame = dc_rem.projections[0]
        del dc_rem.projections[0]
        combined_dc.projections.append(frame)

    combined_dc.geometry.read(str(DATA_DIR / "combined_geometry.h5"))
    return combined_dc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-method",        default="sh",  choices=["sh", "gk"])
    p.add_argument("--ell-max",          type=int,   default=8)
    p.add_argument("--n-iterations",     type=int,   default=20)
    p.add_argument("--laplacian-weight", type=float, default=1e-1)
    p.add_argument("--maxcor",           type=int,   default=5,
                   help="L-BFGS-B correction vectors. Must be ≤8 for the 141³ combined "
                        "volume (maxcor≥9 triggers a 32-bit int overflow → segfault).")
    p.add_argument("--cache-dir",        default="cache/saxs_naf_benchmark")
    p.add_argument("--force",            action="store_true")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    gt_params = dict(
        gt_method=args.gt_method,
        ell_max=args.ell_max,
        n_iterations=args.n_iterations,
        laplacian_weight=args.laplacian_weight,
        maxcor=args.maxcor,
    )

    if not args.force:
        cached = load_recon(cache_dir, "ground_truth", gt_params)
        if cached is not None:
            print(f"Already cached at {cache_dir}.  shape={cached.shape}  (use --force to recompute)")
            return

    print("Building combined DataContainer…")
    combined_dc = build_combined_dc()
    vol = tuple(combined_dc.geometry.volume_shape)
    n   = len(combined_dc.projections)
    print(f"  volume={vol}  projections={n}")

    print(f"\nRunning compute_ground_truth (gt_method={args.gt_method}, "
          f"ell_max={args.ell_max}, n_iterations={args.n_iterations}, "
          f"laplacian_weight={args.laplacian_weight})…")
    print("  Expected wall-time: ~80 min  (4 min/LBFGS iter × 20 iters)")
    t0 = time.time()

    coeffs_gt = compute_ground_truth(combined_dc, **gt_params)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.  shape={coeffs_gt.shape}")

    path = save_recon(cache_dir, "ground_truth", coeffs_gt, gt_params)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
