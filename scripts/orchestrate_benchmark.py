#!/usr/bin/env python3
"""Check cache status and print RunAI commands for missing reconstructions.

Run this script on the server (read-only, no heavy computation).
Copy-paste the printed commands into a local terminal to launch jobs.

Examples
--------
# Check all datasets, all methods:
python /myhome/smartt/scripts/orchestrate_benchmark.py

# Specific datasets only:
python /myhome/smartt/scripts/orchestrate_benchmark.py --datasets b411 frogbone

# Skip NAF sweep, only baseline methods:
python /myhome/smartt/scripts/orchestrate_benchmark.py --methods mumott_sh mumott_gk
"""
import sys
sys.path.insert(0, "/myhome/smartt")

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

from smartt.data_containers import REGISTRY, get_dataset
from smartt.saxs_naf.cache import load_recon, _param_hash


# ---------------------------------------------------------------------------
# Default benchmark configuration
# Keeps these in sync with reconstruct_job.py
# ---------------------------------------------------------------------------

_HOLDOUT_FRAC = 0.0
_HOLDOUT_SEED = 42
_ELL_MAX      = 8

# Fixed NAF base params (not swept)
_BASE_NAF = dict(
    n_iterations=2000,
    batch_size=100,
)

# HP sweep grid for NAF
# _SWEEP_REG_SH = [5e-5, 1e-5, 5e-6, 1e-6]
# _SWEEP_REG_TV = [1e-4, 5e-5, 1e-5, 5e-6]
# _SWEEP_LR     = [1e-2, 5e-3, 1e-3]

_SWEEP_REG_SH = [0]
_SWEEP_REG_TV = [0]
_SWEEP_LR     = [0.0005]

# Fixed mumott params
_MUMOTT = dict(
    mumott_iters=20,
    laplacian_weight=0.1,
    maxcor=5,
)

# RunAI submission settings
_RUNAI_IMAGE   = "lfbarba/sdsc_image:1.0.1"
_RUNAI_PROJECT = "sdate-luisb"
_LAUNCHER      = "bash /myhome/smartt/scripts/smartt_launcher.sh"
_WORKER        = "python /myhome/smartt/scripts/reconstruct_job.py"

# Resource flags per method type (modern RunAI workspace CLI)
_RESOURCES_NAF = (
    "--gpu-request-type portion --gpu-portion-request 0.2 "
    "--node-type A100 --large-shm "
    "--cpu-core-request 4 --cpu-core-limit 10 --cpu-memory-limit 64G"
)
_RESOURCES_MUMOTT = (
    "--cpu-core-request 4 --cpu-core-limit 10 --cpu-memory-limit 128G"
)


# ---------------------------------------------------------------------------
# Params dict builders — identical structure to reconstruct_job.py
# ---------------------------------------------------------------------------

def _naf_params(dataset: str, dc_type: str, reg_sh: float, reg_tv: float, lr: float) -> dict:
    return dict(
        method="naf",
        dataset=dataset,
        dc_type=dc_type,
        ell_max=_ELL_MAX,
        **_BASE_NAF,
        lr=lr,
        reg_weight_sh=reg_sh,
        reg_weight_tv=reg_tv,
        holdout_frac=_HOLDOUT_FRAC,
        holdout_seed=_HOLDOUT_SEED,
    )


def _mumott_params(method: str, dataset: str, dc_type: str) -> dict:
    return dict(
        method=method,
        dataset=dataset,
        dc_type=dc_type,
        ell_max=_ELL_MAX,
        **_MUMOTT,
        holdout_frac=_HOLDOUT_FRAC,
        holdout_seed=_HOLDOUT_SEED,
    )


def _cache_name(method: str, dataset: str, dc_type: str) -> str:
    return f"{method}_{dataset}_{dc_type}"


# ---------------------------------------------------------------------------
# RunAI command builder
# ---------------------------------------------------------------------------

def _job_name(method: str, dataset: str, dc_type: str, params: dict) -> str:
    """Short DNS-compatible job name with a param hash suffix."""
    h = _param_hash(params)[:6]
    m = {"naf": "naf", "mumott_sh": "msh", "mumott_gk": "mgk"}[method]
    return f"smartt-{m}-{dataset}-{dc_type[:3]}-{h}"


def _runai_cmd(method: str, dataset: str, dc_type: str, params: dict) -> str:
    """Build the full runai workspace submit command string."""
    job = _job_name(method, dataset, dc_type, params)
    resources = _RESOURCES_NAF if method == "naf" else _RESOURCES_MUMOTT

    skip = {"method", "dataset", "dc_type"}
    cli_args = " ".join(
        f"--{k} {v}" for k, v in params.items() if k not in skip
    )

    return (
        f"runai workspace submit {job} "
        f"-i {_RUNAI_IMAGE} -p {_RUNAI_PROJECT} {resources} "
        f"--command -- {_LAUNCHER} {_WORKER} "
        f"--dataset {dataset} --dc_type {dc_type} --method {method} {cli_args}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print RunAI commands for missing SAXS-TT reconstructions"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=sorted(REGISTRY),
        help=f"Datasets to check (default: all — {sorted(REGISTRY)})"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["mumott_sh", "mumott_gk", "naf"],
        choices=["mumott_sh", "mumott_gk", "naf"],
    )
    parser.add_argument(
        "--reg_sh", nargs="+", type=float, default=_SWEEP_REG_SH,
        help="NAF reg_weight_sh sweep values"
    )
    parser.add_argument(
        "--reg_tv", nargs="+", type=float, default=_SWEEP_REG_TV,
        help="NAF reg_weight_tv sweep values"
    )
    parser.add_argument(
        "--lr", nargs="+", type=float, default=_SWEEP_LR,
        help="NAF learning rate sweep values"
    )
    args = parser.parse_args()

    missing_cmds = []
    cached_count = 0

    for ds_name in args.datasets:
        ds = get_dataset(ds_name)
        cache_dir = ds.get_cache_dir()

        for dc_type in ds.available_dc_types():
            for method in args.methods:
                if method == "naf":
                    combos = list(product(args.reg_sh, args.reg_tv, args.lr))
                else:
                    combos = [(None, None, None)]

                for reg_sh, reg_tv, lr in combos:
                    if method == "naf":
                        params = _naf_params(ds_name, dc_type, reg_sh, reg_tv, lr)
                    else:
                        params = _mumott_params(method, ds_name, dc_type)

                    name = _cache_name(method, ds_name, dc_type)
                    hit  = load_recon(cache_dir, name, params)

                    if hit is not None:
                        cached_count += 1
                    else:
                        missing_cmds.append(_runai_cmd(method, ds_name, dc_type, params))

    # Summary
    total = cached_count + len(missing_cmds)
    print(f"# Cached: {cached_count}/{total}   Missing: {len(missing_cmds)}/{total}")
    print()

    if not missing_cmds:
        print("# All reconstructions are cached — nothing to submit.")
        return

    print("# ── Copy-paste these commands into your local terminal ──────────────")
    for cmd in missing_cmds:
        print(cmd)
        print()


if __name__ == "__main__":
    main()
