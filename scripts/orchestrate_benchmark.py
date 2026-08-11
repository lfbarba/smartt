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

# Skip NAF, only baseline methods:
python /myhome/smartt/scripts/orchestrate_benchmark.py --methods mumott_sh mumott_gk

# One job per q-bin for a q-indexed dataset (e.g. all 79 frogbone shells),
# instead of the dataset's single default q-bin:
python /myhome/smartt/scripts/orchestrate_benchmark.py --datasets frogbone \\
    --methods mumott_gk --qbins $(seq 0 78)

# Same, for c4 (uses a 'q' constructor kwarg instead of 'qbin' -- detected
# automatically): a handful of shells, or every available index via
# C4DataContainer.list_qshells() (129, sparse -- not every index in [1,194]):
python /myhome/smartt/scripts/orchestrate_benchmark.py --datasets c4 \\
    --methods mumott_gk --qbins 1 10 17 27 38 39 100 111 121 131 142 152 163 173 184 194

# Disable phase-2 regularisation for a dataset where it hurts (e.g. WAXS):
python /myhome/smartt/scripts/orchestrate_benchmark.py --datasets steel-wire-waxs \\
    --methods naf --reg_sh 0.0 --reg_tv 0.0
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

_HOLDOUT_FRAC = 0.
_HOLDOUT_SEED = 42
_ELL_MAX      = 8

# Standard two-phase NAF recipe (see saxs_naf_two_phase_reconstruction
# docstring for the full rationale) — the single, non-swept configuration
# every dataset is reconstructed with. Phase 1 cold-starts and recovers a
# clean object/background split; phase 2 warm-starts at a much higher LR with
# everything unlocked, guarded against overfitting by mild regularisation
# (reg_target_frac_sh/tv — a *target fraction of phase 1's own data loss*,
# auto-calibrated per dataset from phase 1's output, not a fixed weight; a
# fixed weight was tried first and found to be 50-900x miscalibrated across
# datasets with different coefficient scales — see
# saxs_naf_two_phase_reconstruction docstring) plus held-out-tracked early
# stopping. Not universally positive: even correctly calibrated, regularisation
# slightly hurt steel-wire-waxs (a WAXS dataset) in testing even though it
# helped b411/cf-carolina, so a per-dataset override (--reg_sh/--reg_tv, or
# 0.0 to disable) is expected to be needed occasionally rather than treated
# as a bug in these defaults.
_STANDARD_NAF = dict(
    phase1_n_iterations=2001,
    phase1_lr=2e-4,
    phase1_batch_size=100,
    phase1_spatial_frac=0.5,
    phase1_angular_frac=0.6,
    phase1_stochastic_angular=True,
    phase2_n_iterations=1500,
    phase2_lr=5e-3,
    phase2_batch_size=100,
    phase2_reg_target_frac_sh=0.02,
    phase2_reg_target_frac_tv=0.05,
    phase2_early_stop_patience=8,
    phase2_holdout_eval_every=25,
    n_features_per_level=8,
    hidden_dim=64,
    n_hidden_layers=2,
    grid_lr_multiplier=10.0,
    loss_type="huber",
    huber_delta=1.0,
    normalize_target=True,
)

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

def _naf_params(dataset: str, dc_type: str, reg_sh: float, reg_tv: float, q_kwargs: dict = None) -> dict:
    return dict(
        method="naf",
        dataset=dataset,
        dc_type=dc_type,
        ell_max=_ELL_MAX,
        **{**_STANDARD_NAF, "phase2_reg_target_frac_sh": reg_sh, "phase2_reg_target_frac_tv": reg_tv},
        holdout_frac=_HOLDOUT_FRAC,
        holdout_seed=_HOLDOUT_SEED,
        **(q_kwargs or {}),
    )


def _mumott_params(method: str, dataset: str, dc_type: str, q_kwargs: dict = None) -> dict:
    return dict(
        method=method,
        dataset=dataset,
        dc_type=dc_type,
        ell_max=_ELL_MAX,
        **_MUMOTT,
        holdout_frac=_HOLDOUT_FRAC,
        holdout_seed=_HOLDOUT_SEED,
        **(q_kwargs or {}),
    )


def _cache_name(method: str, dataset: str, dc_type: str) -> str:
    return f"{method}_{dataset}_{dc_type}"


def _q_kwarg_name(ds_name: str) -> str:
    """Which constructor kwarg selects the q-index for this dataset.

    Introspects the registered class's ``__init__`` rather than hard-coding
    a convention, since it differs by dataset: frogbone/cf-carolina use
    ``qbin``, c4/px-chameleon/plastic-plasmonics use ``q`` (see
    ``QIndexedDataContainer`` in ``smartt/data_containers/qindexed_base.py``,
    shared by c4 and the near-identical upcoming c5).
    """
    import inspect
    cls = REGISTRY[ds_name]
    params = inspect.signature(cls.__init__).parameters
    if "q" in params:
        return "q"
    if "qbin" in params:
        return "qbin"
    raise ValueError(
        f"{ds_name!r} ({cls.__name__}) has no q/qbin constructor kwarg — "
        f"not a q-indexed dataset, so --qbins doesn't apply to it."
    )


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

    def _flag(k, v):
        if isinstance(v, bool):
            # reconstruct_job.py declares these via argparse.BooleanOptionalAction.
            return f"--{k}" if v else f"--no-{k}"
        return f"--{k} {v}"

    cli_args = " ".join(_flag(k, v) for k, v in params.items() if k not in skip)

    return (
        f"runai workspace submit {job} "
        f"-i {_RUNAI_IMAGE} -p {_RUNAI_PROJECT} {resources} "
        f"--preemptibility preemptible "
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
        "--reg_sh", nargs="+", type=float, default=[_STANDARD_NAF["phase2_reg_target_frac_sh"]],
        help="NAF phase2_reg_target_frac_sh override(s) — pass 0.0 for datasets "
             "where regularisation hurts more than it helps (see _STANDARD_NAF)"
    )
    parser.add_argument(
        "--reg_tv", nargs="+", type=float, default=[_STANDARD_NAF["phase2_reg_target_frac_tv"]],
        help="NAF phase2_reg_target_frac_tv override(s), same caveat as --reg_sh"
    )
    parser.add_argument(
        "--qbins", nargs="+", type=int, default=None,
        help="For q-indexed datasets (frogbone/cf-carolina use a 'qbin' "
             "constructor kwarg; c4/px-chameleon/plastic-plasmonics use 'q' "
             "-- auto-detected per dataset, see _q_kwarg_name): specific "
             "q-indices to generate one job each for, e.g. --qbins $(seq 0 78) "
             "for every frogbone shell, or --qbins $(cat qbins.txt) for c4's "
             "129 (sparse) indices. Default: just the dataset's single "
             "default q-index (no --q/--qbin flag on the generated command)."
    )
    args = parser.parse_args()

    missing_cmds = []
    cached_count = 0

    for ds_name in args.datasets:
        q_kwarg_name = _q_kwarg_name(ds_name) if args.qbins is not None else None
        qbin_values = args.qbins if args.qbins is not None else [None]

        for qbin in qbin_values:
            q_kwargs = {q_kwarg_name: qbin} if qbin is not None else {}
            ds = get_dataset(ds_name, **q_kwargs)
            cache_dir = ds.get_cache_dir()

            for dc_type in ds.available_dc_types():
                for method in args.methods:
                    combos = list(product(args.reg_sh, args.reg_tv)) if method == "naf" else [(None, None)]

                    for reg_sh, reg_tv in combos:
                        if method == "naf":
                            params = _naf_params(ds_name, dc_type, reg_sh, reg_tv, q_kwargs=q_kwargs)
                        else:
                            params = _mumott_params(method, ds_name, dc_type, q_kwargs=q_kwargs)

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
