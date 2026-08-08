"""Dataset-agnostic loading/comparison/plotting for a trained joint multi-q
:class:`SaxsNafField`.

Every function here is parametrized by a :class:`QIndexedDataContainer`
subclass (``C4DataContainer``, ``C5DataContainer``, ...) and a results
directory, instead of being written per-dataset -- adding a new q-resolved
dataset needs zero new plotting code, just pointing these functions at its
class + folder. Supersedes the frogbone/c4-specific one-off comparison
scripts under ``multiq_diagnostics/`` for any *future* dataset (those are
left in place as historical artifacts, not migrated).

Typical use (see ``notebooks/multiq_viewer.ipynb``)::

    from smartt.data_containers.c5 import C5DataContainer
    from smartt.saxs_naf.viz_multiq import run_full_comparison

    result = run_full_comparison(
        C5DataContainer,
        "/myhome/data/smartt/shared/results/c5_benchmark/multiq_diagnostics",
        save_plot_path="/myhome/smartt/notebooks/figures_wandb/c5_multiq_diagnostics/full_baseline_vs_qres.png",
    )
    result["summary"]   # log-log corr, median/max relative deviation, ...
    result["rows"]      # per-qbin table
    result["model"], result["meta"]   # for further interactive exploration
"""
from __future__ import annotations

import glob
import inspect
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch

from .model import SaxsNafField
from .eval_multiq import sample_qshells_physical
from .metrics import split_holdout, _holdout_nrmse, _rsm_volumes, _auto_mask


def _q_kwarg_name(dataset_cls) -> str:
    """Whether *dataset_cls* takes ``q=`` or ``qbin=`` for its q-index."""
    params = inspect.signature(dataset_cls.__init__).parameters
    if "q" in params:
        return "q"
    if "qbin" in params:
        return "qbin"
    raise ValueError(f"{dataset_cls.__name__} has no q/qbin constructor kwarg.")


def _make_dataset(dataset_cls, qbin: int):
    return dataset_cls(**{_q_kwarg_name(dataset_cls): qbin})


def load_multiq_model(results_dir: str, filename: str = "multiq_qres_final.pt", device=None):
    """Load a trained joint multi-q :class:`SaxsNafField` + its query metadata.

    Falls back to the rolling phase-2 then phase-1 checkpoint (same
    directory, ``multiq_qres_phase2_ckpt.pt`` / ``multiq_qres_phase1_ckpt.pt``)
    if the final save isn't there yet -- lets you inspect an in-progress run,
    though only the final save carries the architecture/normalisation
    metadata needed for :func:`compare_multiq_vs_baselines` (a rolling
    checkpoint raises with a pointer to the training script's own
    ``NEW_FIELD``/``Q_FIELD`` instead of guessing).

    Returns
    -------
    model : SaxsNafField, in eval mode.
    meta : dict with ``q_index``, ``q_values``, ``log_q_min``, ``log_q_max``,
        ``target_scale_by_q``, ``ell_max``, ``source_path``, ``is_final``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = [
        os.path.join(results_dir, filename),
        os.path.join(results_dir, "multiq_qres_phase2_ckpt.pt"),
        os.path.join(results_dir, "multiq_qres_phase1_ckpt.pt"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"No multiq checkpoint found in {results_dir!r} (tried {candidates})")

    raw = torch.load(path, map_location=device, weights_only=False)
    is_final = "model_state_dict" in raw
    if not is_final:
        raise ValueError(
            f"{path} is a rolling checkpoint without saved architecture metadata "
            "(field_kwargs/volume_shape/q_values/target_scale_by_q) -- load it "
            "manually with the same NEW_FIELD/Q_FIELD the training script used, "
            "or wait for the final save."
        )

    model = SaxsNafField(raw["volume_shape"], ell_max=raw["ell_max"], **raw["field_kwargs"]).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    meta = dict(
        q_index=raw["q_index"], q_values=raw["q_values"],
        log_q_min=raw["log_q_min"], log_q_max=raw["log_q_max"],
        target_scale_by_q=raw["target_scale_by_q"], ell_max=raw["ell_max"],
        source_path=path, is_final=True,
    )
    return model, meta


def find_baseline_reconstructions(
    dataset_cls, method: str = "mumott_gk", dc_type: str = "main"
) -> Dict[int, str]:
    """``{qbin: npy_path}`` for every completed *method* reconstruction under
    ``dataset_cls._CACHE_DIR_ROOT`` -- one entry per ``q_<N>/`` cache
    subdirectory holding a ``{method}_{name_prefix}_{dc_type}_*.npy``.

    Relies only on the shared ``QIndexedDataContainer`` cache-directory
    convention (``_CACHE_DIR_ROOT/q_<N>/``, from ``get_cache_dir()``), so it
    works for any q-indexed dataset without dataset-specific code.
    """
    root = dataset_cls._CACHE_DIR_ROOT
    name = dataset_cls._NAME_PREFIX
    out = {}
    for d in sorted(glob.glob(os.path.join(str(root), "q_*"))):
        m = re.search(r"q_(\d+)$", d)
        if not m:
            continue
        matches = glob.glob(os.path.join(d, f"{method}_{name}_{dc_type}_*.npy"))
        if matches:
            out[int(m.group(1))] = matches[0]
    return out


def compare_multiq_vs_baselines(
    dataset_cls,
    model,
    meta: dict,
    baseline_paths: Optional[Dict[int, str]] = None,
    method: str = "mumott_gk",
    dc_type: str = "main",
    ell_max: Optional[int] = None,
    holdout_frac: float = 0.15,
    holdout_seed: int = 42,
    n_directions: int = 30,
    half_space: str = "y",
    device=None,
) -> List[dict]:
    """Per-qbin comparison of the joint model against independent baseline
    reconstructions -- mean-c00-over-mask, full RSM correlation, and each
    method's own held-out-projection NRMSE (the real quality signal, needs
    no cross-method comparison).

    If *baseline_paths* isn't given, auto-discovers every completed *method*
    reconstruction via :func:`find_baseline_reconstructions`.
    """
    if device is None:
        device = next(model.parameters()).device
    if ell_max is None:
        ell_max = meta["ell_max"]
    if baseline_paths is None:
        baseline_paths = find_baseline_reconstructions(dataset_cls, method=method, dc_type=dc_type)
    if not baseline_paths:
        raise FileNotFoundError(
            f"No {method!r} baselines found under {dataset_cls._CACHE_DIR_ROOT} -- "
            "pass baseline_paths={qbin: npy_path} explicitly if they live elsewhere."
        )

    from smartt.saxs_fbp import fibonacci_hemisphere
    dirs = fibonacci_hemisphere(n_directions, half_space=half_space)

    rows = []
    for qbin in sorted(baseline_paths):
        ref_coeffs = np.load(baseline_paths[qbin])   # (X,Y,Z,C), physical units

        ds = _make_dataset(dataset_cls, qbin)
        q_phys = ds.get_q_value()

        with torch.no_grad():
            pred_coeffs = sample_qshells_physical(
                model, [q_phys], meta["log_q_min"], meta["log_q_max"],
                meta["target_scale_by_q"], meta["q_values"],
            )[0].numpy()

        c00_ref, c00_pred = ref_coeffs[..., 0], pred_coeffs[..., 0]
        nrmse_c00 = float(np.sqrt(((c00_pred - c00_ref) ** 2).mean()) / (c00_ref.std() + 1e-8))
        corr_c00 = float(np.corrcoef(c00_ref.ravel(), c00_pred.ravel())[0, 1])

        mask = _auto_mask(ref_coeffs.astype(np.float32))
        mean_c00_ref = float(c00_ref[mask].mean())
        mean_c00_pred = float(c00_pred[mask].mean())

        rsm_ref = _rsm_volumes(ref_coeffs.astype(np.float32), dirs, ell_max)
        rsm_pred = _rsm_volumes(pred_coeffs.astype(np.float32), dirs, ell_max)
        a = rsm_ref - rsm_ref.mean(0, keepdims=True)
        b = rsm_pred - rsm_pred.mean(0, keepdims=True)
        num = (a * b).sum(0)
        den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)).clip(1e-8)
        rsm_corr_mean = float((num / den)[mask].mean())

        dc = ds.get_main_dc()
        _, held_dc = split_holdout(dc, fraction=holdout_frac, seed=holdout_seed)
        multiq_holdout = _holdout_nrmse(pred_coeffs.astype(np.float32), held_dc, ell_max, device)
        baseline_holdout = _holdout_nrmse(ref_coeffs.astype(np.float32), held_dc, ell_max, device)

        rows.append(dict(
            qbin=qbin, q=q_phys,
            mean_c00_ref=mean_c00_ref, mean_c00_pred=mean_c00_pred,
            nrmse_c00=nrmse_c00, corr_c00=corr_c00, rsm_corr_mean=rsm_corr_mean,
            baseline_holdout_nrmse=baseline_holdout, multiq_holdout_nrmse=multiq_holdout,
        ))
    return rows


def summarize_comparison(rows: List[dict]) -> dict:
    """Headline numbers from :func:`compare_multiq_vs_baselines`'s rows."""
    ref = np.array([r["mean_c00_ref"] for r in rows])
    pred = np.array([r["mean_c00_pred"] for r in rows])
    rel_dev = np.abs(pred - ref) / np.abs(ref)
    worst = rows[int(np.argmax(rel_dev))]["qbin"]
    return dict(
        n_qshells=len(rows),
        loglog_corr=float(np.corrcoef(np.log(ref), np.log(pred))[0, 1]),
        median_rel_dev=float(np.median(rel_dev)),
        max_rel_dev=float(rel_dev.max()),
        worst_qbin=worst,
        mean_rsm_corr=float(np.mean([r["rsm_corr_mean"] for r in rows])),
        mean_baseline_holdout=float(np.mean([r["baseline_holdout_nrmse"] for r in rows])),
        mean_multiq_holdout=float(np.mean([r["multiq_holdout_nrmse"] for r in rows])),
    )


def plot_baseline_vs_qres(
    rows: List[dict], dataset_name: str = "", baseline_label: str = "baseline", save_path: Optional[str] = None
):
    """Log-log (q vs mean c00) + (baseline vs qres) scatter, mirroring the
    original frogbone/c4 ``full_baseline_vs_qres.png`` comparison plots."""
    import matplotlib
    import matplotlib.pyplot as plt

    q = np.array([r["q"] for r in rows])
    ref = np.array([r["mean_c00_ref"] for r in rows])
    pred = np.array([r["mean_c00_pred"] for r in rows])
    order = np.argsort(q)
    summary = summarize_comparison(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(q[order], ref[order], "o-", label=baseline_label, color="tab:blue")
    axes[0].loglog(q[order], pred[order], "s--", label="Joint multi-q NAF", color="tab:orange")
    axes[0].set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)")
    axes[0].set_ylabel(r"mean $c_{00}$ over object mask")
    title = f"{dataset_name}: full baseline vs joint qres model" if dataset_name else "full baseline vs joint qres model"
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].loglog(ref, pred, "o", color="tab:green")
    lims = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
    axes[1].loglog(lims, lims, "k--", alpha=0.5, label="y=x")
    axes[1].set_xlabel(f"{baseline_label} mean $c_{{00}}$")
    axes[1].set_ylabel("Joint multi-q NAF mean $c_{00}$")
    axes[1].set_title(f"log-log corr = {summary['loglog_corr']:.4f}")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, summary


def run_full_comparison(
    dataset_cls,
    results_dir: str,
    method: str = "mumott_gk",
    dc_type: str = "main",
    save_plot_path: Optional[str] = None,
    device=None,
) -> dict:
    """One-call driver: load the model, compare against every available
    baseline, plot, summarize. This is the generic replacement for the
    per-dataset ``compare_full_sweep_*.py`` scripts -- pass any
    ``QIndexedDataContainer`` subclass and its results folder.
    """
    model, meta = load_multiq_model(results_dir, device=device)
    rows = compare_multiq_vs_baselines(dataset_cls, model, meta, method=method, dc_type=dc_type, device=device)
    fig, summary = plot_baseline_vs_qres(
        rows,
        dataset_name=dataset_cls._NAME_PREFIX,
        baseline_label=method.replace("_", " ").upper(),
        save_path=save_plot_path,
    )
    return dict(model=model, meta=meta, rows=rows, summary=summary, fig=fig)
