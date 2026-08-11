# SAXS-NAF Two-Phase Recipe — Results

This is the results record for the two-phase NAF reconstruction recipe (see
`smartt/saxs_naf/reconstruct.py::saxs_naf_two_phase_reconstruction`, now the
standard `--method naf` behavior in `scripts/reconstruct_job.py` /
`scripts/orchestrate_benchmark.py`, and project memory
`project_saxs_naf_standard_recipe` for the full narrative). For what each
metric means, see `BENCHMARK_METRICS.md`. To browse reconstructions visually
rather than as numbers, use `notebooks/reconstruction_viewer.ipynb` — the
cache chooser there lists every method below (`naf_stochangular(_p1)`,
`naf_antioverfit_reg/earlystop/combined`) per dataset.

All numbers here are **held-out reprojection NRMSE** (`holdout_nrmse`, lower
is better): the reconstruction is reprojected through a disjoint set of
measurement angles never used in training, and compared to the real
measurement. It's a self-supervised metric — no ground truth required — so
it's the one metric available uniformly across every dataset, including the
real (no-GT) ones. All tables below were computed fresh from the real cache
at time of writing, not carried over from earlier chat estimates.

## 1. Registry-wide: does phase 2 help, dataset by dataset?

Phase 1 = cold-start, stochastic-angular reveal (2001 iters). Phase 2 =
warm-started, fully unlocked, high LR, **no regularization, no early
stopping** (1500 iters) — this is the recipe *before* the anti-overfitting
fixes in §2, i.e. it isolates "does unlocking full capacity at a high LR even
help" from "how do we stop it from overfitting."

| dataset | ell_max | phase 1 | phase 2 (unregularized) | Δ (phase2 − phase1) |
|---|---|---|---|---|
| nielsen-m | 12 | 0.0908 | 0.0577 | **−0.0331** |
| steel-wire-waxs | 8 | 0.6395 | 0.5497 | **−0.0898** |
| auditory-ossicle | 8 | 0.5349 | 0.4890 | −0.0459 |
| zenodo | 8 | 0.1381 | 0.1125 | −0.0256 |
| nielsen-mammoth | 8 | 0.0547 | 0.0392 | −0.0155 |
| px-chameleon | 8 | 0.3193 | 0.3060 | −0.0134 |
| c4 | 8 | 0.7677 | 0.7587 | −0.0091 |
| fiber-synthetic-full | 8 | 0.1919 | 0.1862 | −0.0057 |
| plastic-plasmonics | 8 | 0.4391 | 0.4340 | −0.0051 |
| nielsen-t | 2 | 0.0302 | 0.0257 | −0.0045 |
| b411 | 8 | 0.2280 | 0.2251 | −0.0029 |
| frogbone | 8 | 0.1303 | 0.1319 | +0.0016 |
| cf-peek | 8 | 0.7228 | 0.7282 | +0.0053 |
| cf-carolina | 8 | 0.1121 | 0.1202 | **+0.0082** |
| fiber-synthetic | 8 | 0.4901 | 0.5526 | **+0.0624** |

11/15 datasets improve or hold roughly steady in phase 2; 4 (frogbone,
cf-peek, cf-carolina, fiber-synthetic) regress — this is the "phase 2
overfits" symptom that motivated §2. `fiber-synthetic` is the largest
regression and hasn't been re-tested with the anti-overfitting recipe yet —
a natural next candidate if this keeps being a problem.

## 2. Anti-overfitting ablation (b411, steel-wire-waxs, cf-carolina)

These three were flagged as showing clear phase-2 overfitting ("grain" on
noiseless synthetics too, though the grain observation itself was visual, not
captured by holdout NRMSE — see caveat below). Tested four phase-2 variants,
all warm-started from the *same* cached phase-1 checkpoint so the comparison
isolates phase-2 changes only:

- **reg**: turn on `reg_weight_tv`/`reg_weight_sh` (mean-normalized, see
  `SaxsNafField.sh_regularization`/`tv_regularization`), no early stopping.
- **earlystop**: track holdout NRMSE live, keep the best checkpoint, no
  regularization.
- **combined**: both.

| dataset | phase 1 | phase 2 (baseline) | +reg | +earlystop | +combined |
|---|---|---|---|---|---|
| b411 | 0.2280 | 0.2251 | **0.2123** | 0.2249 | 0.2148 |
| steel-wire-waxs | 0.6395 | 0.5497 | 0.5756 (worse) | **0.5484** | 0.5756 (worse) |
| cf-carolina | 0.1121 | 0.1202 | 0.1109 | 0.1157 | **0.1105** |

Takeaway: **no single lever wins everywhere.**
- cf-carolina: clean win, combining both levers beats even phase 1.
- b411: regularization is the effective lever; early stopping barely moves
  the needle (baseline hadn't drifted far from optimal here).
- steel-wire-waxs (the one WAXS dataset in the ablation): the calibrated
  regularization weight actively *hurts* — likely oversmoothing genuine
  high-frequency signal. Early stopping alone gives a small, safe win.

This is why the standard recipe's `phase2_reg_weight_sh=10.0`/
`phase2_reg_weight_tv=100.0` defaults are a starting point, not a tuned
optimum — `orchestrate_benchmark.py --reg_sh 0.0 --reg_tv 0.0 --datasets
<name>` disables them per-dataset. Early stopping (`phase2_early_stop_patience`)
never hurt in this testing, so it's on unconditionally whenever a held-out
split is available.

## Caveat

Holdout NRMSE measures consistency across *unseen viewing angles* of the same
acquisition — a solid proxy for "fitting real signal vs. noise," but it can't
fully see spatial texture/grain that doesn't change the projected value. Where
"grain" is a visual concern, cross-check in `reconstruction_viewer.ipynb`
rather than relying on this table alone.

## Reproducing / extending this report

`scripts/naf_holdout_report.py` regenerates §1 (read-only against the cache —
safe to rerun any time). §2 additionally requires the
`naf_antioverfit_reg/earlystop/combined` cache entries, which currently exist
only for b411/steel-wire-waxs/cf-carolina — there's no standing script for
that ablation yet since it hasn't been extended to other datasets.
