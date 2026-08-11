# NAF vs. mumott (SH/GK) — Paper Results

Comparison of the standard two-phase NAF recipe (`saxs_naf_two_phase_reconstruction`,
see `NAF_TWO_PHASE_RESULTS.md` / project memory `project_saxs_naf_standard_recipe`)
against the two mumott baselines (`mumott_sh` = SphericalHarmonics+LBFGS,
`mumott_gk` = GaussianKernels+LBFGS) for the datasets selected for the paper:
**nielsen-m, nielsen-t, nielsen-mammoth, zenodo, steel-wire-waxs, fiber-synthetic-full**.

All three methods use the *same* `ell_max` per dataset (12 for nielsen-m, 2 for
nielsen-t, 8 for everything else) and the *same* training data per row (see each
section for the exact split). Metric definitions are in `BENCHMARK_METRICS.md`.

**With vs. without phase-2 regularization**: every NAF row below reports *both*
variants of the standard recipe — `NAF (reg)` uses the auto-calibrated
phase-2 regularization (`phase2_reg_target_frac_sh=0.02`,
`phase2_reg_target_frac_tv=0.05` — see mechanism below), `NAF (no-reg)` is the
identical recipe with both target fractions set to `0.0`. **Regularization
benefit is dataset-dependent, not uniformly positive**: it clearly helps
zenodo (both mounts, GT and cross-mount), is roughly a wash on nielsen-m/
nielsen-t, and clearly *hurts* nielsen-mammoth, fiber-synthetic-full, and
steel-wire-waxs (most dramatically fiber-synthetic-full's psnr: 5.82 → 19.04
with regularization off). There is no simple synthetic-vs-real or
missing-wedge-vs-not split that explains the pattern — it appears tied to how
much genuine measurement noise a given dataset has for the unregularized fit
to latch onto, which regularization can only help with if there's noise to
guard against in the first place. Readers should treat the no-reg column as
the more relevant one for datasets marked hurt above; use `--reg_sh 0.0
--reg_tv 0.0` in `orchestrate_benchmark.py` to reproduce.

**Regularization auto-calibration mechanism**: an earlier version of this
report used a *fixed* regularization weight (`reg_weight_sh=10.0`,
`reg_weight_tv=100.0`, tuned on b411/steel-wire-waxs/cf-carolina), which
turned out to be 50-900× too strong for several datasets here because the
coefficient field's absolute scale varies ~200× across datasets — mean-
normalizing the regularizer (an earlier fix) corrects for volume/channel size
but not for this. The recipe now auto-calibrates the weight from phase 1's
own data-loss magnitude right after phase 1 finishes (target *fraction* of
phase 1's loss, not a fixed weight — see `project_saxs_naf_standard_recipe`
memory). A second, independent bug was fixed in the same pass:
`split_holdout(fraction=0.0)` floors at 1 held-out projection, and that single
near-pure-noise projection was truncating phase 2's early-stopping on every
"no-holdout, GT-available" dataset. **All numbers below are post-fix** — the
clearest symptom of the pre-fix bugs was nielsen-t, whose phase 2 used to stop
at 15% of its intended iterations, reading `rsm_corr=0.83` instead of the 0.99
it's actually capable of.

**Mid-report correctness fix (mumott)**: `reconstruct_job.py`'s mumott
baselines were found to call `SquaredLoss` without `use_weights=True`, so
every mumott_sh/gk result *ever cached* (registry-wide, not just here)
silently ignored the per-pixel validity mask, letting invalid/masked detector
pixels corrupt the fit. All mumott numbers below use the corrected
(`use_weights=True`) fits. The rest of the registry's mumott cache (datasets
beyond this paper's 6) is still stale — see project memory
`project_mumott_weights_registry_regen` for the pending follow-up.

**Data-integrity note**: over the course of generating this report, three
cache files were deleted by what appears to be concurrent activity on the same
shared cache directories: zenodo's `ground_truth`, steel-wire-waxs's
`ell_max=8` NAF result, and fiber-synthetic-full's `ground_truth`. All three
were regenerated before finishing this report. The fiber-synthetic-full GT in
particular was **verified**, not just recomputed from the generating notebook
as-is — that notebook (`notebooks/SyntheticDataContainers.ipynb`) has a known
history of producing two orientation-mismatched ground-truth tensors for the
same saved dataset (see `project_synthetic_dc_gt_transpose` memory), so the
regenerated tensor was forward-projected through the existing dataset's real
geometry and checked against its real saved data before being trusted
(matched to relative error ~8e-16).

**Infra note**: the local server alternated between GPU and CPU-only several
times while generating this report, and the CPU-only state was additionally
flaky — background metric-gathering runs twice crashed mid-computation with
an uncaught `cupy`/`numba` CUDA-driver-probe error (no Python traceback, just
`Error: maxBlockDimension getDevice: CUDA error 35`), both when run
concurrently and sequentially. Re-running the same unmodified scripts as
RunAI jobs (`scripts/paper_submit_ablation_gather_runai.py`) succeeded cleanly
every time — if a local metrics run ever dies the same way, prefer RunAI over
retrying locally.

---

## 1. GT-available, single-mount datasets

Trained on the **full dataset, no holdout split** (GT makes a holdout check
unnecessary) — all methods see identical data.

### nielsen-m (ell_max=12, zonal phantom, l≤12)

| method | rsm_corr↑ | ra_mae↓ | orient_err (deg)↓ | psnr↑ | ssim↑ | nrmse↓ |
|---|---|---|---|---|---|---|
| **NAF (reg)** | **0.9233** | 0.0465 | 4.36 | 7.47 | 0.6454 | 3.9508 |
| NAF (no-reg) | 0.9212 | **0.0451** | 4.53\* | **8.29** | **0.6576** | **3.5923** |
| mumott_sh | 0.8387 | 0.1019 | 17.97 | -3.24 | 0.3139 | 13.5487 |
| mumott_gk | 0.8631 | 0.0504 | 9.40 | 1.28 | 0.3511 | 8.0573 |

\* orient_err is the one metric where reg edges ahead (4.36 vs 4.53) — a
near-wash overall, both NAF variants comfortably beat both mumott baselines.

### nielsen-t (ell_max=2, fibre-like phantom, pure rank-2 tensors)

| method | rsm_corr↑ | ra_mae↓ | orient_err (deg)↓ | psnr↑ | ssim↑ | nrmse↓ |
|---|---|---|---|---|---|---|
| **NAF (reg)** | **0.9915** | 0.0068 | **8.62** | 25.63 | **0.9698** | **0.2152** |
| NAF (no-reg) | 0.9899 | 0.0074\* | 9.51 | **25.67** | 0.9687 | 0.2142 |
| mumott_sh | 0.7822 | 0.0516 | 34.19 | 7.13 | 0.3820 | 1.8097 |
| mumott_gk | 0.7146 | 0.0665 | 36.12 | 5.64 | 0.3578 | 2.1496 |

\* ra_mae listed backwards on purpose: 0.0068 (reg) is actually *lower* (better)
than 0.0074 (no-reg) — the split here is genuinely a wash, both variants
essentially tied and both far ahead of mumott.

### nielsen-mammoth (ell_max=8, unrestricted symmetries, 60×60×80)

| method | rsm_corr↑ | ra_mae↓ | orient_err (deg)↓ | psnr↑ | ssim↑ | nrmse↓ |
|---|---|---|---|---|---|---|
| NAF (reg) | 0.9409 | 0.0244 | 14.45 | 17.52 | 0.8261 | 0.9274 |
| **NAF (no-reg)** | **0.9475** | **0.0198** | **12.46** | **21.65** | **0.8519** | **0.5768** |
| mumott_sh | 0.6743 | 0.0275 | 39.09 | 4.19 | 0.2719 | 4.3017 |
| mumott_gk | 0.7759 | 0.0504 | 33.94 | 7.30 | 0.3491 | 3.0078 |

Regularization clearly hurts here — no-reg wins every metric outright, on top
of NAF's already-large margin over both mumott baselines.

### fiber-synthetic-full (ell_max=8, full-sample, no missing wedge)

| method | rsm_corr↑ | ra_mae↓ | orient_err (deg)↓ | psnr↑ | ssim↑ | nrmse↓ |
|---|---|---|---|---|---|---|
| NAF (reg) | 0.9668 | 0.0606 | 3.73 | 5.82 | 0.7211 | 1.2645 |
| **NAF (no-reg)** | **0.9855** | **0.0404** | **2.30** | **19.04** | **0.8954** | **0.2759** |
| mumott_sh | 0.9379 | 0.0838 | 12.56 | 3.59 | 0.4454 | 1.6339 |
| mumott_gk | 0.8956 | 0.1931 | 15.77 | 5.92 | 0.5176 | 1.2490 |

The largest with/without gap in the whole report — psnr more than triples
(5.82 → 19.04) with regularization off. No missing wedge and no measurement
noise here (purely synthetic, exact forward model), so there's nothing for
the regularizer to guard against — it just biases an otherwise-excellent fit.

---

## 2. zenodo (GT-available + cross-mount)

Two independent DataContainers exist for this sample (`main` = data_set_1,
`remount` = data_set_2); ground truth is a mumott reconstruction
(`gt_method='gk', ell_max=8`) on the **combined** (near-full-coverage) data.
Each method is trained on **one full mount at a time** (no internal holdout —
the *other* mount serves as an independent, much stronger consistency check).

### Trained on `main`, evaluated against GT

| method | rsm_corr↑ | ra_mae↓ | orient_err (deg)↓ | psnr↑ | ssim↑ | nrmse↓ |
|---|---|---|---|---|---|---|
| **NAF (reg)** | 0.9558 | 0.0604 | 9.25 | **20.58** | 0.7510 | **0.5432** |
| NAF (no-reg) | 0.9299 | 0.1230 | 12.46 | 15.42 | 0.6949 | 0.9840 |
| mumott_sh | 0.8514 | 0.1064 | 17.72 | 14.64 | 0.6207 | 1.0771 |
| mumott_gk | **0.9510**\* | **0.0478** | **12.55** | 18.19 | **0.7409**\* | 0.7154 |

### Trained on `remount`, evaluated against GT

| method | rsm_corr↑ | ra_mae↓ | orient_err (deg)↓ | psnr↑ | ssim↑ | nrmse↓ |
|---|---|---|---|---|---|---|
| **NAF (reg)** | **0.9520** | 0.0660 | **9.68** | **19.85** | **0.7774** | 0.5911 |
| NAF (no-reg) | 0.9319 | 0.1057 | 12.13 | 15.42 | 0.7509 | 0.9841 |
| mumott_sh | 0.8572 | 0.0960 | 18.48 | 15.47 | 0.6372 | 0.9793 |
| mumott_gk | 0.9589 | **0.0433** | 11.84 | 20.58 | 0.7623 | **0.5433** |

\* mumott_gk is genuinely competitive with NAF(reg) on zenodo's GT metrics
(unlike every other GT dataset above). Regularization is unambiguously the
right call here — NAF(reg) beats NAF(no-reg) on every metric, both mounts.

### Cross-mount held-out reprojection NRMSE (no GT needed)

Reconstruction trained on one mount, reprojected through the *other* mount's
real measured geometry/projections — the strongest generalization test here
since the two mounts are physically independent measurements.

| method | main→remount | remount→main |
|---|---|---|
| **NAF (reg)** | **0.1709** | **0.1807** |
| NAF (no-reg) | 0.1924 | 0.1958 |
| mumott_sh | 0.3152 | 0.3315 |
| mumott_gk | 0.2139 | 0.2133 |

Regularization helps cross-mount generalization too, consistent with the GT
results above — both NAF variants still generalize far better than either
mumott baseline.

---

## 3. steel-wire-waxs (no ground truth — WAXS, real single-mount data)

15% held-out projections (seed 42), held-out reprojection NRMSE only.

| method | holdout_nrmse↓ |
|---|---|
| NAF (reg) | 0.6318 |
| **NAF (no-reg)** | **0.5558** |
| mumott_sh | 0.5742 |
| mumott_gk | 0.6119 |

With regularization off, NAF beats *both* mumott baselines; with the standard
recipe's auto-calibrated regularization on, it's worse than mumott_sh. This is
the one WAXS dataset in the report (every other dataset is SAXS) — combined
with fiber-synthetic-full and nielsen-mammoth also preferring no-reg, this is
one more data point against "regularization always helps," not evidence that
NAF is weaker than mumott on WAXS data in general. Also note NAF training uses
no fixed random seed by default, so exact holdout NRMSE varies run-to-run on
this dataset more than on the better-conditioned ones above.

---

## Reproducing

- All 7 dataset/dc_type combinations, both regularization settings: `scripts/paper_run_naf_datasets.sh <gpu_id>` (local GPU, reg-on defaults) / `scripts/paper_submit_remaining_runai.py` and `scripts/paper_submit_noreg_runai.py` (RunAI, reg-on and reg-off respectively) — all reuse `orchestrate_benchmark.py`'s param/command builders with this report's exact per-dataset ell_max/holdout_frac overrides.
- Metrics (reg-on only): `scripts/paper_gather_gt_metrics.py`, `scripts/paper_gather_zenodo_metrics.py`, `scripts/paper_gather_steelwire_metrics.py`.
- Metrics (both reg-on and reg-off, i.e. the tables above): `scripts/paper_gather_gt_metrics_ablation.py`, `scripts/paper_gather_zenodo_metrics_ablation.py`, `scripts/paper_gather_steelwire_metrics_ablation.py` — or `scripts/paper_submit_ablation_gather_runai.py` to run them on RunAI instead of locally. All read directly from the reconstruction cache via `smartt.saxs_naf.cache.load_recon`, no retraining needed if already cached.
- mumott baselines: `scripts/reconstruct_job.py --method {mumott_sh,mumott_gk} --dataset <name> --dc_type <main|remount> --ell_max <N> --holdout_frac <0.0|0.15>`.
