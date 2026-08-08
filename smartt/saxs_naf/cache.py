"""Parameter-aware reconstruction cache for SAXS NAF benchmarks.

Each cached reconstruction is stored as a pair:
  ``{name}_{hash8}.npy``   — the (X, Y, Z, C) coefficient array
  ``{name}_{hash8}.json``  — sidecar with the full parameter dict

The 8-char hex hash is derived from the MD5 of the JSON-serialised parameter
dict (keys sorted, values coerced to str for non-JSON-serialisable types).
Different parameter dicts produce different filenames automatically, so
multiple runs with different settings coexist in the same cache directory
without overwriting each other.

Typical usage
-------------
>>> params = dict(ell_max=8, n_iterations=2000, gt_method='sh')
>>> coeffs = load_recon(CACHE_DIR, 'ground_truth', params)
>>> if coeffs is None:
...     coeffs = run_expensive_reconstruction(...)
...     save_recon(CACHE_DIR, 'ground_truth', coeffs, params)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _param_hash(params: Dict[str, Any]) -> str:
    """Return an 8-char hex hash of the sorted, JSON-serialised params dict."""
    serialised = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(serialised.encode()).hexdigest()[:8]


def cache_stem(name: str, params: Dict[str, Any]) -> str:
    """Return the filename stem (without extension) for a name+params pair."""
    return f"{name}_{_param_hash(params)}"


def npy_path(cache_dir, name: str, params: Dict[str, Any]) -> Path:
    """Full path to the .npy file for a given name+params."""
    return Path(cache_dir) / f"{cache_stem(name, params)}.npy"


def save_recon(
    cache_dir,
    name: str,
    coeffs: np.ndarray,
    params: Dict[str, Any],
) -> Path:
    """Save ``coeffs`` and a JSON sidecar under ``cache_dir``.

    Parameters
    ----------
    cache_dir : path-like
        Directory to write into (created if absent).
    name : str
        Logical name, e.g. ``'ground_truth'``, ``'naf_b411R'``.
    coeffs : ``(X, Y, Z, C)`` float32 array.
    params : dict
        All parameters used to produce ``coeffs``.  Stored verbatim in the
        sidecar so the provenance is always recoverable from disk.

    Returns
    -------
    Path to the saved .npy file.
    """
    path = npy_path(cache_dir, name, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, coeffs)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(params, indent=2, default=str))
    return path


def load_recon(
    cache_dir,
    name: str,
    params: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Load a cached reconstruction if it exists, else return ``None``.

    The lookup is purely hash-based: if the .npy file exists it is loaded
    without any further parameter comparison (the sidecar is for human
    inspection, not for runtime validation).

    Parameters
    ----------
    cache_dir : path-like
    name : str
    params : dict
        Must be identical (same keys, same values) to the dict used when
        :func:`save_recon` was called.

    Returns
    -------
    ``(X, Y, Z, C)`` float32 array, or ``None`` if not cached.
    """
    path = npy_path(cache_dir, name, params)
    if path.exists():
        try:
            arr = np.load(path)
        except (ValueError, OSError):
            return None
        if arr.dtype == object:
            return None
        # Validate SH coefficient count against ell_max when available.
        ell_max = params.get("ell_max")
        if ell_max is not None and arr.ndim == 4:
            expected_c = sum(2 * ell + 1 for ell in range(0, int(ell_max) + 1, 2))
            if arr.shape[-1] != expected_c:
                import warnings
                warnings.warn(
                    f"Stale cache {path.name}: has {arr.shape[-1]} coefficients "
                    f"but ell_max={ell_max} expects {expected_c}. Recomputing.",
                    stacklevel=2,
                )
                return None
        return arr
    return None


def model_path(cache_dir, name: str, params: Dict[str, Any]) -> Path:
    """Full path to the ``.model.pt`` checkpoint for a given name+params.

    Uses a ``.model.pt`` extension (not ``.pt``) so the model checkpoint and its
    sidecar never collide with the ``.npy``/``.json`` pair written by
    :func:`save_recon` for the same name+params.
    """
    return Path(cache_dir) / f"{cache_stem(name, params)}.model.pt"


def save_model(
    cache_dir,
    name: str,
    model,
    params: Dict[str, Any],
) -> Path:
    """Persist a trained :class:`~smartt.saxs_naf.model.SaxsNafField`.

    Stores a checkpoint holding the model's construction ``config`` (from
    ``model.get_config()``) and ``state_dict``, so :func:`load_model` can rebuild
    an identical architecture and restore the weights — enabling super-resolution
    querying of the field long after the training run.

    Parameters
    ----------
    cache_dir : path-like
    name : str
        Logical name, matching the one used with :func:`save_recon`.
    model : SaxsNafField
        Must expose ``get_config()`` and ``state_dict()``.
    params : dict
        Same parameter dict used for the reconstruction (drives the hash).

    Returns
    -------
    Path to the saved ``.model.pt`` file.
    """
    import torch
    path = model_path(cache_dir, name, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": model.get_config(), "state_dict": model.state_dict()}, path
    )
    sidecar = path.with_suffix(".json")   # -> {stem}.model.json
    sidecar.write_text(json.dumps(params, indent=2, default=str))
    return path


def load_model(
    cache_dir,
    name: str,
    params: Dict[str, Any],
    device=None,
):
    """Load a cached :class:`~smartt.saxs_naf.model.SaxsNafField`, else ``None``.

    Rebuilds the field from the stored ``config`` and loads its ``state_dict``.

    Parameters
    ----------
    cache_dir : path-like
    name : str
    params : dict
        Must be identical to the dict passed to :func:`save_model`.
    device : torch.device or str, optional
        If given, the loaded model is moved to this device (and the checkpoint is
        mapped there); defaults to CPU.

    Returns
    -------
    A ``SaxsNafField`` in eval-ready state, or ``None`` if not cached.
    """
    import torch
    from .model import SaxsNafField

    path = model_path(cache_dir, name, params)
    if not path.exists():
        return None
    try:
        ckpt = torch.load(
            path, map_location=device or "cpu", weights_only=False
        )
    except (ValueError, OSError, RuntimeError):
        return None
    model = SaxsNafField(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model = model.to(device)
    return model


def save_metrics(
    cache_dir,
    dc_type: str,
    metrics: dict,
    params: Dict[str, Any],
) -> "Path":
    """Persist the output of :func:`~smartt.saxs_naf.metrics.compute_metrics`.

    Parameters
    ----------
    cache_dir : path-like
    dc_type : str
        Partition label (e.g. ``'b411R'``), used as part of the filename.
    metrics : dict
        ``method → {metric_name → scalar_or_array}`` as returned by
        :func:`compute_metrics`.
    params : dict
        All parameters that determine the metric values (reconstruction params,
        ell_max, K, half_space, …).  Changes to any param produce a new cache
        file; the old one is left on disk.

    Returns
    -------
    Path to the saved ``.pkl`` file.
    """
    import pickle
    path = Path(cache_dir) / f"metrics_{dc_type}_{_param_hash(params)}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(metrics, fh, protocol=4)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(params, indent=2, default=str))
    return path


def load_metrics(
    cache_dir,
    dc_type: str,
    params: Dict[str, Any],
) -> "Optional[dict]":
    """Load cached metrics if present, else return ``None``.

    Parameters
    ----------
    cache_dir : path-like
    dc_type : str
    params : dict
        Must be identical to the dict passed to :func:`save_metrics`.

    Returns
    -------
    The metrics dict, or ``None`` if no matching cache file exists.
    """
    import pickle
    path = Path(cache_dir) / f"metrics_{dc_type}_{_param_hash(params)}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def project_params(row: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``{k: row[k]}`` for every key that ``target`` already defines.

    The canonical way to pull a selected sidecar back into a notebook param dict:
    a sidecar is a *superset* of the notebook's param dict (it also carries
    ``method``/``dataset``/``dc_type``/``holdout_*``), so we project onto the
    keys the target already owns and leave everything else alone.

    >>> project_params(sidecar, NAF_PARAMS)   # -> only the 6 NAF keys
    """
    return {k: row[k] for k in target if k in row}


def load_selected(cache_dir, sidecar: Dict[str, Any]) -> Optional[np.ndarray]:
    """Load the exact reconstruction a chooser row points at.

    ``sidecar`` is a full sidecar param dict — e.g. an entry of
    :attr:`CacheChooser.selection`.  It carries ``method``/``dataset``/``dc_type``
    alongside the hyper-parameters, so it alone determines both the file *name*
    and the content *hash*.  Loading with it verbatim guarantees a cache hit.

    Prefer this over merging a sidecar into a notebook default dict and rebuilding
    the params: the merge silently keeps default keys the sidecar never had (an
    older sidecar lacking ``holdout_*`` is the canonical case), and those extra
    keys change the hash into a miss even though the .npy is right there on disk.

    Returns the ``(X, Y, Z, C)`` array, or ``None`` if the file is absent.
    """
    name = f"{sidecar['method']}_{sidecar['dataset']}_{sidecar['dc_type']}"
    return load_recon(cache_dir, name, sidecar)


# Categories whose reconstructions carry no tunable parameters — nothing to choose.
_CHOOSER_SKIP = {"fbp"}
# Columns that are bookkeeping, not selectable hyper-parameters.
_CHOOSER_META_COLS = {"name", "hash", "shape", "mtime", "dataset", "method"}


def _entry_category(entry: Dict[str, Any]) -> str:
    """Map a cache entry to its chooser category.

    ``ground_truth`` sidecars carry no ``method`` field (they are keyed on
    ``gt_method``); everything else is grouped by its ``method`` string.
    """
    params = entry.get("params") or {}
    if entry.get("name") == "ground_truth" or "method" not in params:
        return "ground_truth"
    return params.get("method", entry.get("name", "?"))


def cache_table(cache_dir) -> "Any":
    """Return a :class:`pandas.DataFrame` summarising every cached reconstruction.

    One row per ``.npy``/sidecar pair.  Columns are ``name``, ``hash``, ``shape``,
    ``mtime`` followed by the union of all sidecar parameter keys.  Intended for
    eyeballing what has already been computed in a ``cache_dir`` so param sets can
    be reloaded instead of retyped (see :class:`CacheChooser`).
    """
    import pandas as pd

    cache_dir = Path(cache_dir)
    rows = []
    for entry in list_cache(cache_dir):
        row: Dict[str, Any] = {
            "name": entry["name"],
            "hash": entry["hash"],
            "shape": str(entry.get("shape")),
            "category": _entry_category(entry),
        }
        npy = cache_dir / f"{entry['name']}_{entry['hash']}.npy"
        try:
            row["mtime"] = int(npy.stat().st_mtime)
        except OSError:
            row["mtime"] = None
        row.update(entry.get("params") or {})
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    lead = ["category", "name", "hash", "shape", "mtime"]
    rest = [c for c in df.columns if c not in lead]
    return df[[c for c in lead if c in df.columns] + rest]


class CacheChooser:
    """Interactive picker over a reconstruction ``cache_dir``.

    Renders one grid per method category (``mumott_sh``, ``mumott_gk``, ``naf``,
    plus a ``ground_truth`` grid), each with a row-dropdown; ``fbp`` is skipped
    (no tunable params).  A top-level ``dc_type`` dropdown filters every
    method grid so the shared keys (``ell_max``/``holdout_*``/``dc_type``) can
    never disagree across categories.  Selecting *none* for a category leaves the
    notebook's own default in place.

    Clicking **Load** populates :attr:`selection` — a ``{category: params}`` dict
    holding the full sidecar dict for each chosen row — and prints it.  The
    consuming cell must be re-run afterwards to pick up the new params.

    Notebook usage::

        chooser = CacheChooser(cache_dir); chooser        # renders the grids
        # ... click Load ...
        sel = chooser.selection
        if 'naf' in sel:
            NAF_PARAMS.update(project_params(sel['naf'], NAF_PARAMS))
    """

    #: display order for the category grids
    ORDER = ["mumott_sh", "mumott_gk", "naf", "naf_twophase_p1", "naf_twophase",
              "naf_bgmask", "naf_stochangular_p1", "naf_stochangular",
              "naf_antioverfit_reg", "naf_antioverfit_earlystop", "naf_antioverfit_combined",
              "sh_sa", "ground_truth"]

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.df = cache_table(self.cache_dir)
        self.selection: Dict[str, Dict[str, Any]] = {}
        self._build()

    @property
    def dc_type(self):
        """The currently-selected ``dc_type`` (``None`` if the cache is empty)."""
        return getattr(self, "_dc_dd", None) and self._dc_dd.value

    # -- construction -----------------------------------------------------
    def _dc_types(self):
        if self.df.empty or "dc_type" not in self.df:
            return []
        vals = [v for v in self.df["dc_type"].dropna().unique().tolist()]
        return sorted(vals)

    def _build(self):
        import ipywidgets as widgets

        if self.df.empty:
            self._box = widgets.HTML(
                f"<i>Cache directory {self.cache_dir} is empty.</i>"
            )
            return

        dc_types = self._dc_types()
        self._dc_dd = widgets.Dropdown(
            options=dc_types, value=dc_types[0] if dc_types else None,
            description="dc_type:", style={"description_width": "initial"},
        )
        self._dc_dd.observe(self._on_dc_change, names="value")

        self._grids_box = widgets.VBox([])
        self._load_btn = widgets.Button(
            description="Load", button_style="primary", icon="download"
        )
        self._load_btn.on_click(self._on_load)
        self._out = widgets.Output()
        self._dropdowns: Dict[str, "widgets.Dropdown"] = {}

        self._refresh_grids()
        self._box = widgets.VBox(
            [self._dc_dd, self._grids_box, self._load_btn, self._out]
        )

    def _row_label(self, sub, idx):
        """Compact one-line summary of a row: hash + the columns that vary."""
        def _nunique(col):
            # Sidecar values can include unhashable types (e.g. voxel_indices
            # lists), which crash Series.nunique(); stringify first since this
            # is only used to decide whether a column varies, not to display it.
            try:
                return col.nunique(dropna=False)
            except TypeError:
                return col.astype(str).nunique(dropna=False)

        varying = [
            c for c in sub.columns
            if c not in _CHOOSER_META_COLS and c not in ("category", "dc_type")
            and _nunique(sub[c]) > 1
        ]
        row = sub.loc[idx]
        parts = [f"{c}={row[c]}" for c in varying]
        return f"{row['hash']}  " + " ".join(parts) if parts else f"{row['hash']}"

    def _refresh_grids(self):
        import ipywidgets as widgets
        from IPython.display import display

        dc = self._dc_dd.value
        boxes = []
        self._dropdowns = {}
        for cat in self.ORDER:
            sub = self.df[self.df["category"] == cat]
            # GT has no dc_type; every other category filters to the selected one.
            if cat != "ground_truth" and "dc_type" in sub:
                sub = sub[sub["dc_type"] == dc]
            sub = sub.dropna(axis=1, how="all")
            if sub.empty:
                continue
            options = [("— none —", None)] + [
                (self._row_label(sub, i), sub.loc[i, "hash"]) for i in sub.index
            ]
            dd = widgets.Dropdown(
                options=options, value=None, description=f"{cat}:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="auto"),
            )
            self._dropdowns[cat] = dd
            grid = widgets.Output()
            with grid:
                cols = [c for c in sub.columns if c not in ("category", "mtime")]
                display(sub[cols].reset_index(drop=True))
            boxes.append(widgets.VBox([widgets.HTML(f"<b>{cat}</b>"), grid, dd]))
        self._grids_box.children = boxes

    # -- callbacks --------------------------------------------------------
    def _on_dc_change(self, _change):
        self._refresh_grids()

    def _params_for_hash(self, h):
        # Read the raw sidecar rather than the DataFrame row: pandas coerces
        # mixed columns to float (n_iterations -> 2000.0), which would re-hash
        # differently than the original int and turn a cache hit into a silent
        # recompute.  The sidecar preserves the exact original types.
        row = self.df[self.df["hash"] == h].iloc[0]
        sidecar = self.cache_dir / f"{row['name']}_{h}.json"
        return json.loads(sidecar.read_text())

    def _on_load(self, _btn):
        self.selection = {}
        for cat, dd in self._dropdowns.items():
            if dd.value is not None:
                self.selection[cat] = self._params_for_hash(dd.value)
        self._out.clear_output()
        with self._out:
            if not self.selection:
                print("No rows selected — notebook defaults unchanged.")
            for cat, params in self.selection.items():
                print(f"[{cat}] {params}")

    def _ipython_display_(self):
        from IPython.display import display
        display(self._box)


def list_cache(cache_dir) -> list[dict]:
    """Return a summary of all cached reconstructions in ``cache_dir``.

    Each entry in the returned list is a dict with keys ``name``, ``hash``,
    ``shape`` (if loadable), and ``params`` (from the sidecar if present).
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    entries = []
    for npy in sorted(cache_dir.glob("*.npy")):
        stem = npy.stem          # e.g. "ground_truth_a3f2c1d0"
        parts = stem.rsplit("_", 1)
        entry: dict = {"name": parts[0], "hash": parts[1] if len(parts) == 2 else stem}
        sidecar = npy.with_suffix(".json")
        if sidecar.exists():
            try:
                entry["params"] = json.loads(sidecar.read_text())
            except Exception:
                entry["params"] = None
        try:
            entry["shape"] = np.load(npy, mmap_mode="r").shape
        except Exception:
            entry["shape"] = None
        entries.append(entry)
    return entries
