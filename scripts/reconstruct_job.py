#!/usr/bin/env python3
"""Run a single reconstruction and save the result to cache.

Launched by RunAI via smartt_launcher.sh.  Each job handles one
(dataset × dc_type × method × hyperparameters) combination.

Example
-------
# NAF with specific regularisation weights
python /myhome/smartt/scripts/reconstruct_job.py \\
    --dataset b411 --dc_type main --method naf \\
    --reg_weight_sh 1e-6 --reg_weight_tv 0.0

# mumott SphericalHarmonics
python /myhome/smartt/scripts/reconstruct_job.py \\
    --dataset b411 --dc_type main --method mumott_sh
"""
import sys
sys.path.insert(0, "/myhome/smartt")

import argparse

from smartt.data_containers import get_dataset
from smartt.saxs_naf.cache import save_recon, load_recon
from smartt.saxs_naf.metrics import split_holdout


# ---------------------------------------------------------------------------
# Parameter dict builders — must stay in sync with orchestrate_benchmark.py
# ---------------------------------------------------------------------------

def _naf_params(args: argparse.Namespace) -> dict:
    return dict(
        method="naf",
        dataset=args.dataset,
        dc_type=args.dc_type,
        ell_max=args.ell_max,
        n_iterations=args.n_iterations,
        lr=args.lr,
        batch_size=args.batch_size,
        reg_weight_sh=args.reg_weight_sh,
        reg_weight_tv=args.reg_weight_tv,
        holdout_frac=args.holdout_frac,
        holdout_seed=args.holdout_seed,
    )


def _mumott_params(args: argparse.Namespace) -> dict:
    return dict(
        method=args.method,
        dataset=args.dataset,
        dc_type=args.dc_type,
        ell_max=args.ell_max,
        mumott_iters=args.mumott_iters,
        laplacian_weight=args.laplacian_weight,
        maxcor=args.maxcor,
        holdout_frac=args.holdout_frac,
        holdout_seed=args.holdout_seed,
    )


def cache_name(args: argparse.Namespace) -> str:
    return f"{args.method}_{args.dataset}_{args.dc_type}"


# ---------------------------------------------------------------------------
# Reconstruction routines
# ---------------------------------------------------------------------------

def _run_naf(train_dc, args):
    from smartt.saxs_naf import saxs_naf_reconstruction
    result = saxs_naf_reconstruction(
        train_dc,
        ell_max=args.ell_max,
        n_iterations=args.n_iterations,
        lr=args.lr,
        batch_size=args.batch_size,
        reg_weight_sh=args.reg_weight_sh,
        reg_weight_tv=args.reg_weight_tv,
    )
    return result["reconstruction"].numpy()


def _run_mumott_sh(train_dc, args):
    from mumott.methods.basis_sets import SphericalHarmonics
    from mumott.methods.projectors import SAXSProjector
    from mumott.methods.residual_calculators import GradientResidualCalculator
    from mumott.optimization.loss_functions import SquaredLoss
    from mumott.optimization.optimizers import LBFGS
    from mumott.optimization.regularizers import Laplacian

    projector = SAXSProjector(train_dc.geometry)
    basis = SphericalHarmonics(
        ell_max=args.ell_max,
        probed_coordinates=train_dc.geometry.probed_coordinates,
    )
    rc = GradientResidualCalculator(
        data_container=train_dc, basis_set=basis, projector=projector
    )
    loss = SquaredLoss(residual_calculator=rc)
    loss.add_regularizer("laplacian", Laplacian(), regularization_weight=args.laplacian_weight)
    result = LBFGS(loss, maxiter=args.mumott_iters, maxcor=args.maxcor).optimize()
    return result["x"].astype("float32")


def _run_mumott_gk(train_dc, args):
    from mumott.methods.basis_sets import GaussianKernels
    from mumott.methods.projectors import SAXSProjector
    from mumott.methods.residual_calculators import GradientResidualCalculator
    from mumott.optimization.loss_functions import SquaredLoss
    from mumott.optimization.optimizers import LBFGS
    from mumott.optimization.regularizers import Laplacian
    from smartt.saxs_naf.metrics import to_sh_coefficients

    projector = SAXSProjector(train_dc.geometry)
    basis = GaussianKernels(probed_coordinates=train_dc.geometry.probed_coordinates)
    rc = GradientResidualCalculator(
        data_container=train_dc, basis_set=basis, projector=projector
    )
    loss = SquaredLoss(residual_calculator=rc)
    loss.add_regularizer("laplacian", Laplacian(), regularization_weight=args.laplacian_weight)
    result = LBFGS(loss, maxiter=args.mumott_iters, maxcor=args.maxcor).optimize()
    return to_sh_coefficients(basis, ell_max=args.ell_max, coefficients=result["x"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Single SAXS-TT reconstruction job")

    # Dataset / job identity
    parser.add_argument("--dataset",  required=True, help="Dataset name, e.g. 'b411'")
    parser.add_argument("--dc_type",  required=True,
                        choices=["main", "remount", "combined"],
                        help="Which DataContainer to reconstruct")
    parser.add_argument("--method",   required=True,
                        choices=["naf", "mumott_sh", "mumott_gk"],
                        help="Reconstruction method")
    parser.add_argument("--force",    action="store_true",
                        help="Recompute even if cached result exists")

    # Holdout split (must match what metrics uses)
    parser.add_argument("--holdout_frac", type=float, default=0.10)
    parser.add_argument("--holdout_seed", type=int,   default=42)

    # Shared
    parser.add_argument("--ell_max",  type=int,   default=8)

    # NAF-specific
    parser.add_argument("--n_iterations", type=int,   default=2000)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--batch_size",   type=int,   default=40)
    parser.add_argument("--reg_weight_sh", type=float, default=1e-6)
    parser.add_argument("--reg_weight_tv", type=float, default=0.0)

    # Mumott-specific
    parser.add_argument("--mumott_iters",   type=int,   default=20)
    parser.add_argument("--laplacian_weight", type=float, default=0.1)
    parser.add_argument("--maxcor",         type=int,   default=5)

    args = parser.parse_args()

    # Build params dict and check cache
    params = _naf_params(args) if args.method == "naf" else _mumott_params(args)
    name   = cache_name(args)
    ds     = get_dataset(args.dataset)
    cache_dir = ds.get_cache_dir()

    if not args.force and load_recon(cache_dir, name, params) is not None:
        print(f"[skip] Already cached: {name} in {cache_dir}")
        return

    # Build DC and split holdout
    print(f"[info] Building {args.dc_type} DC for dataset '{args.dataset}'...")
    dc = ds.get_dc(args.dc_type)
    train_dc, _ = split_holdout(dc, fraction=args.holdout_frac, seed=args.holdout_seed)

    # Run reconstruction
    print(f"[info] Running {args.method} reconstruction...")
    if args.method == "naf":
        coeffs = _run_naf(train_dc, args)
    elif args.method == "mumott_sh":
        coeffs = _run_mumott_sh(train_dc, args)
    elif args.method == "mumott_gk":
        coeffs = _run_mumott_gk(train_dc, args)

    # Save
    path = save_recon(cache_dir, name, coeffs, params)
    print(f"[done] Saved {coeffs.shape} array → {path}")


if __name__ == "__main__":
    main()
