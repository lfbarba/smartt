#!/usr/bin/env python3
"""Recompute held-out reprojection NRMSE for every cached naf_stochangular result.

Source data for NAF_TWO_PHASE_RESULTS.md §1. Reads ell_max/holdout_frac/
holdout_seed straight from each cached result's JSON sidecar (never guesses),
so it stays correct even as datasets/params evolve. Read-only against the
cache — safe to rerun any time to refresh the numbers in that doc.

Usage
-----
python /myhome/smartt/scripts/naf_holdout_report.py
"""
import sys
sys.path.insert(0, "/myhome/smartt")

import glob
import json
import os

import numpy as np
import torch

from smartt.data_containers import get_dataset
from smartt.saxs_naf.metrics import split_holdout, _holdout_nrmse


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p2_jsons = sorted(glob.glob(
        "/myhome/data/smartt/shared/**/naf_stochangular_*_main_*.json", recursive=True
    ))
    p2_jsons = [p for p in p2_jsons if "_p1_" not in p]
    # cf-carolina's qbin_010 default was later found to be 100% NaN
    # (weights.sum()==0 across the whole dataset, a pre-existing data
    # problem) and superseded by qbin_071 -- see project memory
    # project_cf_carolina_dataset. Exclude the stale qbin_010 cache entry.
    p2_jsons = [p for p in p2_jsons if "/qbin_010/" not in p]

    rows = []
    for p2_json in p2_jsons:
        params = json.loads(open(p2_json).read())
        dataset = params["dataset"]
        ell_max = params["ell_max"]
        holdout_frac = params["holdout_frac"]
        holdout_seed = params["holdout_seed"]
        dc_type = params["dc_type"]

        # phase1's hash suffix differs from phase2's (different params dict),
        # so glob for it by prefix within the same directory rather than
        # trying to derive it from the phase2 filename.
        p1_matches = glob.glob(os.path.join(
            os.path.dirname(p2_json), f"naf_stochangular_p1_{dataset}_main_*.npy"
        ))
        p1_npy = p1_matches[0] if p1_matches else None
        p2_npy = p2_json.replace(".json", ".npy")

        try:
            ds = get_dataset(dataset)
        except Exception as e:
            print(f"[skip] {dataset}: {e}")
            continue
        dc = ds.get_dc(dc_type) if dc_type != "main" else ds.get_main_dc()
        _, held_dc = split_holdout(dc, fraction=holdout_frac, seed=holdout_seed)

        row = {"dataset": dataset, "ell_max": ell_max, "holdout_frac": holdout_frac}
        row["phase1"] = _holdout_nrmse(np.load(p1_npy), held_dc, ell_max, device) if p1_npy else None
        row["phase2"] = _holdout_nrmse(np.load(p2_npy), held_dc, ell_max, device)
        rows.append(row)
        if row["phase1"] is not None:
            print(f"{dataset:22s} ell_max={ell_max:2d} phase1={row['phase1']:.4f}  "
                  f"phase2={row['phase2']:.4f}  delta={row['phase2']-row['phase1']:+.4f}")
        else:
            print(f"{dataset:22s} ell_max={ell_max:2d} phase1=N/A    phase2={row['phase2']:.4f}")

    return rows


if __name__ == "__main__":
    main()
