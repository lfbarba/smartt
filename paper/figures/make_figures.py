#!/usr/bin/env python3
"""Generate the qualitative figures for the SAXS-TT NAF paper.

Each figure shows one *reciprocal-space-map (RSM) direction*: the SH
coefficient field ``(X, Y, Z, C)`` is contracted with the real-SH basis
evaluated at a single unit vector ``q``, which collapses it to a scalar volume
``I_q(x, y, z)``.  That volume is the spatially resolved scattered intensity
that would be seen looking along ``q`` — the natural object to compare across
reconstruction methods, because a missing-wedge direction is exactly one whose
scalar volume the measurement never constrains directly.

The script is deliberately spec-driven so the paper's figure list can grow or
shrink without editing code: every ``--fig`` argument produces exactly one PDF.

Examples
--------
List the RSM directions for a dataset, sorted by how badly the missing wedge
hits them (use this to choose ``k``)::

    python make_figures.py --list-directions dataset=zenodo

Render two figures::

    python make_figures.py \\
        --fig dataset=zenodo:dc_type=main:k=12:x=32:y=33:z=32 \\
        --fig dataset=nielsen-mammoth:k=7:x=30:y=30:z=40

Spec keys (all optional except ``dataset``)::

    dataset     registry key (zenodo, nielsen-m, nielsen-t, nielsen-mammoth,
                fiber-synthetic-full, steel-wire-waxs)
    dc_type     main | remount            (default main)
    ell_max     SH band limit             (default per-dataset, see _ELL_MAX)
    holdout     holdout fraction used at reconstruction time (default per-dataset)
    k           index of the RSM direction to render (default: the direction
                with the largest missing arc, i.e. the worst-case one)
    x, y, z     slice indices             (default: volume centre)
    planes      comma-separated subset of xy,xz,yz   (default all three)
    methods     comma-separated subset of the method labels (default all found)
    name        output file stem          (default auto from dataset/dc/k)
    cmap        matplotlib colormap       (default inferno)
    diff        1 to add a signed-difference row against GT (default 0)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/myhome/smartt")
sys.path.insert(0, "/myhome/smartt/scripts")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from smartt.data_containers import get_dataset
from smartt.saxs_naf.cache import load_recon
from smartt.saxs_naf.eval import evaluate_real_sh
from smartt.saxs_fbp import fibonacci_hemisphere
import orchestrate_benchmark as ob


# --------------------------------------------------------------------------
# Per-dataset acquisition parameters, matching PAPER_RESULTS.md exactly.
# --------------------------------------------------------------------------

_ELL_MAX = {
    "nielsen-m": 12,
    "nielsen-t": 2,
    "nielsen-mammoth": 8,
    "fiber-synthetic-full": 8,
    "zenodo": 8,
    "steel-wire-waxs": 8,
}
_HOLDOUT = {
    "steel-wire-waxs": 0.15,
}
# Goniometer tilt half-angle (deg) bounding the reachable sample orientations.
# Everything beyond this is the missing wedge.  ``None`` = full coverage.
_ALPHA_DEG = {
    "nielsen-m": 45.0,
    "nielsen-t": 45.0,
    "nielsen-mammoth": 45.0,
    "zenodo": 45.0,
    "steel-wire-waxs": 45.0,
    "fiber-synthetic-full": None,   # full-sample control, no missing wedge
}

_MUMOTT_FIXED = dict(mumott_iters=20, laplacian_weight=0.1, maxcor=5)

# The dataset registry key stays "zenodo" (matches the data-container class
# and its on-disk cache directory), but that name is just the hosting website
# for the raw data, not a description of the sample — the paper never uses it
# and instead displays this dataset as "trabecular-bone" everywhere a human
# reads the label (figure titles, filenames, captions).
_DISPLAY_NAME = {
    "zenodo": "trabecular-bone",
}

# zenodo has no independent ground truth: the "combined" reference is itself a
# mumott-GK reconstruction, so scoring reconstruction methods against it is
# circular for any metric that would favour a matching inductive bias. Shown
# here for visual/qualitative orientation only — label it honestly rather
# than as "GT" (the internal dict key stays "GT" for the shared render logic
# below; only the displayed title changes).
_GT_DISPLAY_OVERRIDE = {
    "zenodo": "Combined (GK), not GT",
}

# Display label -> (cache method name, params-builder key).  The NAF row is the
# no-regularisation variant, which is what the paper presents (see
# PAPER_RESULTS.md: regularisation is available but off by default).
_METHODS = [
    ("GT", "ground_truth"),
    ("NAF (ours)", "naf"),
    ("mumott SH", "mumott_sh"),
    ("mumott GK", "mumott_gk"),
]

K_DIRECTIONS = 30
HALF_SPACE = "y"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _naf_params(dataset, dc_type, ell_max, holdout):
    """Params dict for the no-regularisation NAF run (paper default)."""
    return dict(
        method="naf", dataset=dataset, dc_type=dc_type, ell_max=ell_max,
        **{**ob._STANDARD_NAF,
           "phase2_reg_target_frac_sh": 0.0,
           "phase2_reg_target_frac_tv": 0.0},
        holdout_frac=holdout, holdout_seed=42,
    )


def _mumott_params(method, dataset, dc_type, ell_max, holdout):
    return dict(
        method=method, dataset=dataset, dc_type=dc_type, ell_max=ell_max,
        **_MUMOTT_FIXED, holdout_frac=holdout, holdout_seed=42,
    )


# zenodo has no exact ground truth: its reference is a mumott GK reconstruction
# on the *combined* (both-mount, near-full-coverage) data, cached under the
# name "ground_truth".  These params must match paper_gather_zenodo_metrics.py.
_ZENODO_GT_PARAMS = dict(gt_method="gk", ell_max=8, n_iterations=20,
                         laplacian_weight=0.1, maxcor=5)


def _load_ground_truth(ds, dataset, cache_dir):
    """Exact GT via the dataset API, or the cached combined-mount reference."""
    if dataset == "zenodo":
        gt = load_recon(cache_dir, "ground_truth", _ZENODO_GT_PARAMS)
        if gt is None:
            print(f"    [no GT] {dataset}: cached combined-mount reference not found")
            return None
        return np.asarray(gt, dtype=np.float32)
    try:
        return np.asarray(ds.get_ground_truth(), dtype=np.float32)
    except (AttributeError, FileNotFoundError, RuntimeError) as exc:
        print(f"    [no GT] {dataset}: {exc}")
        return None


def load_volumes(dataset, dc_type, ell_max, holdout, wanted=None):
    """Return ``label -> (X, Y, Z, C)`` for every method found in the cache."""
    ds = get_dataset(dataset)
    cache_dir = ds.get_cache_dir()
    out = {}

    for label, method in _METHODS:
        if wanted is not None and label not in wanted:
            continue
        if method == "ground_truth":
            gt = _load_ground_truth(ds, dataset, cache_dir)
            if gt is None:
                continue
            out[label] = gt
            continue

        if method == "naf":
            params = _naf_params(dataset, dc_type, ell_max, holdout)
        else:
            params = _mumott_params(method, dataset, dc_type, ell_max, holdout)
        coeffs = load_recon(cache_dir, f"{method}_{dataset}_{dc_type}", params)
        if coeffs is None:
            print(f"    [missing] {label} ({method})")
            continue
        out[label] = np.asarray(coeffs, dtype=np.float32)

    # The Nielsen phantoms store a GT whose band limit is phantom-specific; the
    # reconstructions all use `ell_max`.  Align by truncating / zero-padding —
    # valid because mumott orders even-l real SH in nested increasing-l blocks,
    # so the leading coefficients are the same basis functions either way.
    if out:
        target_c = max(v.shape[-1] for k, v in out.items() if k != "GT") \
            if any(k != "GT" for k in out) else next(iter(out.values())).shape[-1]
        for label, vol in list(out.items()):
            c = vol.shape[-1]
            if c > target_c:
                out[label] = vol[..., :target_c]
            elif c < target_c:
                out[label] = np.pad(vol, ((0, 0), (0, 0), (0, 0), (0, target_c - c)))
    return out


# --------------------------------------------------------------------------
# RSM evaluation
# --------------------------------------------------------------------------

def rsm_volume(coeffs, direction, ell_max):
    """Contract an SH coefficient field with the basis at one direction.

    ``I_q(x,y,z) = sum_c  B[c](q) * coeffs[x,y,z,c]``  ->  scalar volume.
    """
    B = evaluate_real_sh(
        torch.tensor(np.asarray(direction, dtype=np.float32)[None, :]), ell_max
    ).numpy()[0]                                    # (C,)
    C = coeffs.shape[-1]
    return np.tensordot(coeffs, B[:C], axes=([3], [0]))


def missing_arcs(directions, alpha_deg):
    """Missing arc length (deg) for each RSM direction, or zeros if no wedge."""
    if alpha_deg is None:
        return np.zeros(len(directions))
    from smartt.saxs_isonet.wedge import all_missing_arcs
    return np.degrees(all_missing_arcs(directions, alpha_deg, None))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

_PLANE_INFO = {
    # key: (slicer, title fmt, xlabel, ylabel)
    "yz": (lambda v, x, y, z: v[x, :, :].T, "YZ  (x={x})", "Y", "Z"),
    "xz": (lambda v, x, y, z: v[:, y, :].T, "XZ  (y={y})", "X", "Z"),
    "xy": (lambda v, x, y, z: v[:, :, z].T, "XY  (z={z})", "X", "Y"),
}


def render_figure(spec, outdir, dpi=300):
    dataset = spec["dataset"]
    dc_type = spec.get("dc_type", "main")
    ell_max = int(spec.get("ell_max", _ELL_MAX.get(dataset, 8)))
    holdout = float(spec.get("holdout", _HOLDOUT.get(dataset, 0.0)))
    planes = [p.strip() for p in spec.get("planes", "yz,xz,xy").split(",") if p.strip()]
    cmap = spec.get("cmap", "inferno")
    want_diff = str(spec.get("diff", "0")) not in ("0", "false", "False", "")
    wanted = None
    if "methods" in spec:
        wanted = [m.strip() for m in spec["methods"].split(",")]

    print(f"  loading {dataset}/{dc_type} (ell_max={ell_max}, holdout={holdout}) ...")
    vols = load_volumes(dataset, dc_type, ell_max, holdout, wanted)
    if not vols:
        print(f"  [skip] {dataset}: nothing loaded")
        return None

    directions = fibonacci_hemisphere(K_DIRECTIONS, half_space=HALF_SPACE)
    arcs = missing_arcs(directions, _ALPHA_DEG.get(dataset))

    k = int(spec["k"]) if "k" in spec else int(np.argmax(arcs))
    q = directions[k]

    shape = next(iter(vols.values())).shape[:3]
    x = int(spec.get("x", shape[0] // 2))
    y = int(spec.get("y", shape[1] // 2))
    z = int(spec.get("z", shape[2] // 2))

    # Contract every method's coefficient field along the same direction.
    scalar = {label: rsm_volume(c, q, ell_max) for label, c in vols.items()}

    labels = [lab for lab, _ in _METHODS if lab in scalar]
    ref = "GT" if "GT" in scalar else labels[0]

    # Shared colour scale, anchored on the reference volume's displayed slices
    # so the comparison is honest across columns.
    ref_slices = [_PLANE_INFO[p][0](scalar[ref], x, y, z) for p in planes]
    lo = min(float(np.percentile(s, 0.5)) for s in ref_slices)
    hi = max(float(np.percentile(s, 99.5)) for s in ref_slices)

    show_diff = want_diff and ref == "GT" and len(labels) > 1
    n_rows = len(planes) * (2 if show_diff else 1)
    n_cols = len(labels)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.35 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )

    dlim = 0.5 * (hi - lo)
    im_abs = im_dif = None

    for pi, plane in enumerate(planes):
        slicer, tfmt, xl, yl = _PLANE_INFO[plane]
        for ci, label in enumerate(labels):
            ax = axes[pi if not show_diff else 2 * pi][ci]
            img = slicer(scalar[label], x, y, z)
            im_abs = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi,
                               origin="lower", aspect="equal", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            if pi == 0:
                title = _GT_DISPLAY_OVERRIDE.get(dataset, label) if label == "GT" else label
                ax.set_title(title, fontsize=10)
            if ci == 0:
                ax.set_ylabel(tfmt.format(x=x, y=y, z=z), fontsize=8)

            if show_diff:
                axd = axes[2 * pi + 1][ci]
                if label == ref:
                    # The reference minus itself is identically zero — an empty
                    # panel would just waste a column, so drop the axis.
                    axd.set_axis_off()
                    continue
                d = img - slicer(scalar[ref], x, y, z)
                im_dif = axd.imshow(d, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                                    origin="lower", aspect="equal", rasterized=True)
                axd.set_xticks([]); axd.set_yticks([])
                if ci == 1:
                    ref_disp = _GT_DISPLAY_OVERRIDE.get(dataset, ref) if ref == "GT" else ref
                    axd.set_ylabel(f"residual vs {ref_disp}", fontsize=8)

    arc_txt = f",  missing arc {arcs[k]:.0f}$^\\circ$" if _ALPHA_DEG.get(dataset) else ",  full coverage"
    dataset_disp = _DISPLAY_NAME.get(dataset, dataset)
    fig.suptitle(
        f"{dataset_disp} / {dc_type} — RSM direction $k$={k}  "
        f"$q$=({q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}){arc_txt}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    cb = fig.colorbar(im_abs, ax=axes[:, :].ravel().tolist(),
                      shrink=0.6, pad=0.015, fraction=0.025)
    cb.set_label("$I_q$ (a.u.)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    stem = spec.get("name", f"rsm_{dataset}_{dc_type}_k{k}")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.pdf"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def list_directions(spec):
    dataset = spec["dataset"]
    directions = fibonacci_hemisphere(K_DIRECTIONS, half_space=HALF_SPACE)
    arcs = missing_arcs(directions, _ALPHA_DEG.get(dataset))
    order = np.argsort(-arcs)
    print(f"\nRSM directions for {dataset} "
          f"(alpha={_ALPHA_DEG.get(dataset)}), sorted by missing arc:\n")
    print(f"  {'k':>3}  {'qx':>7} {'qy':>7} {'qz':>7}   missing arc")
    for k in order:
        q = directions[k]
        print(f"  {k:>3}  {q[0]:>7.3f} {q[1]:>7.3f} {q[2]:>7.3f}   {arcs[k]:>7.1f} deg")


def parse_spec(s):
    spec = {}
    for part in s.split(":"):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad spec fragment {part!r} (expected key=value)")
        k, v = part.split("=", 1)
        spec[k.strip()] = v.strip()
    if "dataset" not in spec:
        raise ValueError(f"spec {s!r} has no dataset= key")
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", action="append", default=[],
                    help="figure spec, key=value pairs separated by ':' "
                         "(repeat for more figures)")
    ap.add_argument("--list-directions", default=None,
                    help="print the RSM directions for a spec and exit")
    ap.add_argument("--outdir", default=str(Path(__file__).parent),
                    help="output directory for the PDFs")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if args.list_directions:
        list_directions(parse_spec(args.list_directions))
        return

    if not args.fig:
        ap.error("no --fig specs given (and no --list-directions)")

    outdir = Path(args.outdir)
    written = []
    for s in args.fig:
        spec = parse_spec(s)
        print(f"[figure] {s}")
        p = render_figure(spec, outdir, dpi=args.dpi)
        if p:
            written.append(p)

    print(f"\n{len(written)}/{len(args.fig)} figures written to {outdir}")


if __name__ == "__main__":
    main()
