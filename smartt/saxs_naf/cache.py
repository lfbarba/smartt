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
