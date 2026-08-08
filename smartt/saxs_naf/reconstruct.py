"""Per-scan self-supervised NAF reconstruction for SAXS tensor tomography.

``saxs_naf_reconstruction`` mirrors the signature/spirit of
``smartt.saxs_fbp.reconstruction.saxs_gd_reconstruction`` so it is a drop-in in
the comparison notebook, but the ``(X, Y, Z, C)`` SH-coefficient field is the
output of a coordinate network (:class:`SaxsNafField`) rather than a free
parameter.  The forward model (``build_mumott_projector`` + ``forward_quadrature``)
is reused unchanged.

Key mechanics (see project memory ``project_saxs_naf_design``):
* **Cached fixed-partition projector pool** — projections are partitioned once
  into chunks, one projector built per chunk, cycled; the pool is reshuffled and
  rebuilt every ``reshuffle_every`` steps.  Avoids the per-step ``deepcopy``.
* **Cold start** with data-calibrated ``c00`` so the isotropic component starts
  at the right scale.
* **Linear coarse-to-fine** spatial + angular annealing via :class:`Annealer`.
* **Boolean-masked Huber (default) or MSE** data term + ℓ(ℓ+1) angular penalty.
  Huber downweights the gradient contribution of outlier pixels (measurement
  spikes an order of magnitude above the local signal are common in these
  detectors) without needing a hard percentile cutoff.
"""

from __future__ import annotations

# torch._dynamo is imported lazily the first time any torch.optim optimizer is
# constructed (via @_disable_dynamo on Optimizer.add_param_group).  In some
# torch 2.x + distributed environments the import fails with:
#   RuntimeError: …kernel registered…wait_tensor…_c10d_functional namespace
# because _c10d_functional has already registered the same op from C++.
# We pre-empt the crash by stubbing torch._dynamo if the real import fails.
# The stub only needs disable() to work as a pass-through so that
# _disable_dynamo's inner() can wrap its target without TorchDynamo compilation.
import sys as _sys
import types as _types

if "torch._dynamo" not in _sys.modules:
    try:
        import torch._dynamo  # noqa: F401
    except RuntimeError:
        def _noop(*a, **kw): pass  # noqa: E704
        def _passthrough(fn=None, recursive=True, wrapping=True):  # noqa: E306
            return (lambda f: f) if fn is None else fn
        _stub = _types.ModuleType("torch._dynamo")
        _stub.disable = _passthrough          # used by @_disable_dynamo
        _stub.graph_break = _noop            # used by _use_grad_for_differentiable
        _stub.__getattr__ = lambda _name: _noop  # catch-all for any other attr
        _sys.modules["torch._dynamo"] = _stub

import copy
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import SaxsNafField
from .schedule import Annealer


def _cosine_warmup_lr(step: int, total: int, warmup: int) -> float:
    """Multiplicative LR factor: linear warmup then cosine decay to ~0."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def _full_dataset_Y_int(
    dc,
    ell_max: int,
    device: torch.device,
    cache_dir=None,
) -> torch.Tensor:
    """Compute the ``(N, M, C)`` SH integration matrix once for the whole dataset.

    Uses the fast flat-plane quadrature (:func:`precompute_Y_int`) for SAXS
    geometries, and mumott's own curvature-correct basis set
    (:func:`mumott_projection_matrix`) whenever the geometry carries a nonzero
    ``two_theta`` (WAXS) — see project memory / plan for why the two must not
    be conflated. Computing this once per dataset (rather than once per
    training chunk, as before) is what keeps the WAXS path tractable: mumott's
    adaptive quadrature is CPU-only and must not run on every pool rebuild.
    """
    from smartt.shutils.evaulate_sh import precompute_Y_int, mumott_projection_matrix

    geometry = dc.geometry
    is_waxs = bool(np.any(np.asarray(geometry.two_theta) != 0))
    if not is_waxs:
        return precompute_Y_int(geometry.probed_coordinates, ell_max=ell_max, device=device)

    cache_path = None
    if cache_dir is not None:
        from pathlib import Path
        cache_path = Path(cache_dir) / f"mumott_Y_int_ellmax{ell_max}_hash{hash(geometry) & 0xFFFFFFFF:08x}.npy"
    return mumott_projection_matrix(
        geometry.probed_coordinates, ell_max=ell_max, device=device, cache_path=cache_path,
    )


def _build_projector_pool(
    dc,
    perm: np.ndarray,
    batch_size: int,
    full_target: torch.Tensor,
    full_weights: torch.Tensor,
    device: torch.device,
    ell_max: int,
    Y_int_full: torch.Tensor,
) -> List[dict]:
    """Partition projections into chunks; build one projector each (once).

    The SH integration matrix ``Y_int`` (shape ``(N, M, C)``) is computed once
    for the *full* dataset by :func:`_full_dataset_Y_int` and sliced by
    projection index here — chunks share the same underlying matrix, so
    rebuilding the pool (e.g. on reshuffle) never re-runs SH integration.
    During training each step uses a single ``torch.einsum`` instead of
    re-running the per-step Python loop inside ``forward_quadrature``.
    """
    from smartt.projectors import build_mumott_projector

    n = len(perm)
    pool: List[dict] = []
    for start in range(0, n, batch_size):
        idx = np.sort(perm[start : start + batch_size])
        sub = copy.deepcopy(dc)
        keep = set(int(i) for i in idx)
        for j in sorted(range(len(dc.projections)), reverse=True):
            if j not in keep:
                del sub.projections[j]
        pool.append(
            {
                "projector": build_mumott_projector(sub.geometry, device=device),
                "Y_int": Y_int_full[idx],
                "target": full_target[idx],
                "mask": full_weights[idx],
            }
        )
    return pool


def saxs_naf_reconstruction(
    dc,
    ell_max: int = 8,
    n_iterations: int = 2000,
    lr: float = 1e-2,
    batch_size: int = 40,
    reshuffle_every: int = 200,
    warmup_steps: int = 50,
    reg_weight_sh: float = 1e-6,
    reg_weight_tv: float = 0.0,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    anneal_spatial: bool = True,
    anneal_angular: bool = True,
    spatial_frac: float = 0.25,
    angular_frac: float = 0.35,
    cold_start: bool = True,
    calibrate_c00: bool = True,
    normalize_target: bool = True,
    n_features_per_level: int = 8,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
    head_init_std: float = 1e-3,
    grid_lr_multiplier: float = 10.0,
    compute_rank_diagnostics: bool = True,
    warm_start_state_dict: Optional[dict] = None,
    background_mask: Optional[np.ndarray] = None,
    reg_weight_background: float = 0.0,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 200,
    resume: bool = False,
    reset_grid_optimizer_state_on_unlock: bool = False,
    stochastic_angular: bool = False,
    held_out_dc=None,
    holdout_eval_every: int = 100,
    early_stop_patience: Optional[int] = None,
    use_best_checkpoint: bool = False,
    field_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Reconstruct an ``(X, Y, Z, C)`` SH-coefficient field with a NAF model.

    A coordinate network (multiresolution hash encoding + MLP) is optimised
    self-supervisedly against the measured SAXS projections in ``dc``.  The
    forward model is the unmodified mumott pipeline:
    ``build_mumott_projector`` (linear X-ray transform) followed by
    ``forward_quadrature`` (SH angular contraction), so gradients flow
    end-to-end through the field.

    Parameters
    ----------
    dc : mumott DataContainer
        Must have ``dc.geometry`` (provides ``volume_shape`` and
        ``probed_coordinates``) and ``dc.projections`` (provides ``.data``
        and ``.weights``).  Pass the training subset; the full/remounted DC
        is only needed for post-hoc comparison via ``evaluate_models``.
    ell_max : int
        Maximum even SH degree.  ``ell_max=8`` → 45 coefficients.
    n_iterations : int
        Total gradient steps.
    lr : float
        Peak learning rate for AdamW (after cosine warmup).
    batch_size : int
        Number of projections per gradient step.  One projector is built per
        chunk at pool-construction time (avoids per-step deepcopy).
    reshuffle_every : int
        Rebuild the projector pool with a fresh random partition every this
        many steps (0 to disable).
    warmup_steps : int
        Number of steps for the linear LR warmup before cosine decay.
    reg_weight_sh : float
        Weight for the ℓ(ℓ+1) angular regulariser (Laplace-Beltrami).
        Set to 0 to disable.
    reg_weight_tv : float
        Weight for the squared total-variation spatial regulariser, summed
        over all SH channels and all three axes.  Off by default (0.0);
        enable (e.g. ``1e-7``) if spatial noise or ringing appears in the
        reconstruction.
    loss_type : str
        Data term: ``'huber'`` (default) or ``'mse'``.  SAXS projections
        routinely carry outlier pixels an order of magnitude above the local
        signal (detector spikes); Huber caps their gradient contribution
        instead of letting them dominate the squared-error term. Set to
        ``'mse'`` to recover the previous behaviour.
    huber_delta : float
        Transition point between the quadratic and linear regimes of the
        Huber loss, in the same units as the (optionally normalised) target —
        see ``normalize_target``. With ``normalize_target=True`` (default)
        the target is rescaled to O(1), so the default ``1.0`` is a
        reasonable starting point; tune down if outliers still dominate,
        or up to behave closer to MSE. Ignored when ``loss_type='mse'``.
    anneal_spatial : bool
        Linearly reveal hash-encoding levels coarse→fine over
        ``spatial_frac`` of training (FreeNeRF-style).
    anneal_angular : bool
        Linearly reveal SH degrees ℓ=0→2→4→6→8 over ``angular_frac`` of
        training, suppressing high-frequency angular content early on.
    spatial_frac : float
        Fraction of training over which spatial levels are fully revealed.
    angular_frac : float
        Fraction of training over which all SH degrees are fully revealed
        (should be ≥ ``spatial_frac`` to stagger the two ramps).
    cold_start : bool
        Start from all-zero anisotropy (only c₀₀ non-zero).  Warm-start
        from an FBP reconstruction is deferred to a future version.
    calibrate_c00 : bool
        Before training, scale the c₀₀ bias so the isotropic prediction
        matches the mean of the valid target pixels.  Requires
        ``cold_start=True``.
    normalize_target : bool
        Divide the target projections by the mean of their valid pixels
        before training, so the data loss (and its gradients) sit at O(1)
        regardless of a dataset's raw intensity scale — different q-shells
        span wildly different magnitudes (e.g. [0, 50] vs [0, 1e6]), and
        without this the loss/gradient scale — and therefore how ``lr``,
        ``reg_weight_sh``/``reg_weight_tv``, and any fixed gradient-clip
        threshold behave — would vary per dataset.  The returned
        ``reconstruction`` is rescaled back to the original (unnormalised)
        units, so this is transparent to callers.  Default on.
    n_features_per_level, hidden_dim, n_hidden_layers : int
        Forwarded to :class:`SaxsNafField` — see its docstring for why the
        defaults favour grid capacity (``n_features_per_level=8``) over a
        lean shared trunk (``hidden_dim=64``, ``n_hidden_layers=2``).
    head_init_std : float
        Forwarded to :class:`SaxsNafField`; see its docstring. ``0.0``
        reproduces the old zero-init (starts the readout at exact rank 1).
    grid_lr_multiplier : float
        The hash-grid tables are optimised at ``lr * grid_lr_multiplier``;
        the trunk/head at ``lr``. Grid tables start at a much smaller scale
        (Instant-NGP init, ``std=1e-4``) than a freshly-initialised MLP's
        activations, so a shared learning rate under-trains the grid relative
        to the trunk — the opposite of what a hash-grid encoding is meant to
        rely on (most of the representational capacity should live in the
        per-voxel table, not the shared decoder).
    compute_rank_diagnostics : bool
        After training, compute :func:`~smartt.saxs_naf.diagnostics.full_rank_report`
        (decoder/encoder effective-rank + the model-agnostic per-voxel shape
        rank) and include it in the returned dict as ``rank_diagnostics``.
        Cheap (bounded-size SVDs on subsampled voxels); set ``False`` to skip.
    field_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`SaxsNafField`
        (e.g. ``n_levels``, ``base_resolution``). Takes precedence over the
        named ``n_features_per_level``/``hidden_dim``/``n_hidden_layers``/
        ``head_init_std`` arguments above if the same key is also passed here.
    warm_start_state_dict : dict, optional
        A ``SaxsNafField.state_dict()`` from a *previous* call with the SAME
        architecture (``field_kwargs``/``ell_max``/``volume_shape``) as this
        one. When given, the model loads these weights instead of the usual
        cold-start (``cold_start``/``calibrate_c00`` are ignored) and trains
        with a *fresh* optimiser/scheduler at this call's ``lr``/``n_iterations``
        — i.e. a second, independently-paced phase continuing from the first
        phase's solution. Typical use: phase 1 with ``head_init_std=0`` and the
        old, slower annealing (good at recovering a clean object/background
        split, per empirical observation); phase 2 warm-started from phase 1
        with ``anneal_spatial=False, anneal_angular=False`` (everything already
        unlocked) and a larger ``lr``, to aggressively fit per-voxel RSM shape
        on top of the already-correct object silhouette. Any baked-in
        ``output_scale`` from the source run is reset to 1.0 before continuing
        (this call re-normalises/re-bakes it the usual way), so ``normalize_target``
        behaves identically to a fresh run.
    background_mask : ``(X, Y, Z)`` bool array, optional
        Voxels *outside* this mask (i.e. background/air) are penalised toward
        zero every step via ``reg_weight_background``. Intended to be derived
        from a *different*, already-trained reconstruction's ``c00`` (e.g.
        Otsu-threshold the phase-1 / zero-init run, which empirically gives a
        cleaner object/air split than the higher-capacity config) and fed in
        here to suppress missing-wedge background artefacts that the data
        alone under-constrains, independent of which config produced the mask.
    reg_weight_background : float
        Weight for the background-suppression penalty (mean squared
        coefficient value over ``~background_mask`` voxels, all channels
        including ``c00``). ``0.0`` (default) disables it; ignored if
        ``background_mask`` is ``None``.
    checkpoint_path : str, optional
        If given, ``model``/``optimizer`` state + current step are saved here
        (atomically — written to ``{checkpoint_path}.tmp`` then renamed) every
        ``checkpoint_every`` steps and once more at the final step. Makes a
        long run resumable after a crash or an environment restart, at the
        cost of losing at most ``checkpoint_every`` steps of progress.
    checkpoint_every : int
        How often to write the checkpoint (steps). Ignored if
        ``checkpoint_path`` is ``None``.
    resume : bool
        If ``True`` and ``checkpoint_path`` exists on disk, restore
        ``model``/``optimizer`` state and continue from the saved step instead
        of cold/warm-starting — the LR/annealing schedules are pure functions
        of step number, so they resume correctly with no extra state needed.
        ``cold_start``/``calibrate_c00``/``warm_start_state_dict`` are ignored
        when an actual resume happens. Requires identical ``field_kwargs``
        (architecture) to the run that wrote the checkpoint, same as
        ``warm_start_state_dict``.
    reset_grid_optimizer_state_on_unlock : bool
        Spatial annealing (``anneal_spatial``) masks each hash level's
        features to exactly ``0`` before it starts revealing (see
        ``level_weights`` in ``MultiResolutionHashEncoding.forward``), so that
        level's table receives an exact-zero gradient every step until then —
        but Adam's per-tensor ``step`` counter keeps incrementing regardless
        (``.grad`` is not ``None``, just zero-valued), so by the time a level
        is revealed its bias-correction denominator is already ~1, losing the
        usual early-training boost a freshly-initialised parameter gets. Each
        ``GridLevel.table`` is its own parameter tensor (unlike the shared
        ``head.weight``, whose per-channel angular reveal can't be reset this
        way without splitting it into one tensor per SH degree — not done),
        so resetting is exact here: when ``True``, the optimizer's
        ``step``/``exp_avg``/``exp_avg_sq`` for a level's table are zeroed the
        first step its ``level_weights`` entry becomes nonzero, giving it a
        genuinely fresh Adam start at that point. No effect when
        ``anneal_spatial=False`` (nothing ever transitions) or on resume
        before the checkpoint's step (state for already-unlocked levels is
        left alone).
    stochastic_angular : bool
        Replaces the deterministic per-channel angular reveal with a
        per-voxel stochastic one (see ``Annealer.ell_mask`` docstring in
        ``schedule.py``): each voxel independently sees either every SH
        degree at full strength, or the scheduled partial truncation, with
        the "full" probability ramping from 0 to 1 over ``angular_frac`` of
        training. A cleaner alternative to
        ``reset_grid_optimizer_state_on_unlock`` for the angular ramp
        specifically (that trick doesn't cleanly apply here since
        ``head.weight``'s rows share one parameter tensor): every degree gets
        some real, full-strength gradient from step 0, so no
        ``head.weight`` row is ever frozen at exact zero waiting to unlock.
        No effect when ``anneal_angular=False``.
    held_out_dc : mumott DataContainer, optional
        A held-out split (e.g. from ``metrics.split_holdout``), disjoint from
        ``dc``. When given, every ``holdout_eval_every`` steps (and the final
        step) the *current*, unmasked full-ℓ model is re-projected through
        this DC's own geometry/projector and its NRMSE against the held-out
        measurements is recorded — a live proxy for "is this still improving
        the reconstruction, or just fitting the training projections' noise
        (or, for noiseless data, the null space of an under-determined
        missing-wedge inverse)". Intended for a high-LR, unregularised
        fine-tune phase (e.g. the ``warm_start_state_dict`` phase-2 pattern)
        that has no other mechanism to notice it has started overfitting.
        The held-out target is divided by the same ``target_scale`` as the
        training target so the two NRMSEs are comparable. ``None`` disables
        all of this (no extra compute).
    holdout_eval_every : int
        How often (steps) to run the held-out evaluation above. Ignored if
        ``held_out_dc`` is ``None``.
    early_stop_patience : int, optional
        If given (and ``held_out_dc`` is not ``None``), stop training early
        once ``holdout_nrmse`` has failed to improve on its best-seen value
        for this many *consecutive evaluations* (not steps). ``None``
        disables early stopping — the held-out curve is still recorded, just
        not acted on.
    use_best_checkpoint : bool
        If ``True`` (and ``held_out_dc`` is not ``None``), the weights from
        the step with the lowest recorded ``holdout_nrmse`` are reloaded
        before computing the returned ``reconstruction`` — i.e. the result
        reflects the best point on the held-out curve, not whatever the
        training loop happened to end on. Requires ``held_out_dc``; a no-op
        if no evaluation ever ran (shouldn't happen since step 0 always
        evaluates).
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    verbose : bool
        Print calibration info and show a tqdm progress bar.
    seed : int, optional
        Seed for NumPy and PyTorch RNGs (for reproducible pool shuffles).
    cache_dir : str, optional
        Directory to cache the SH integration matrix in for WAXS datasets
        (``dc.geometry.two_theta`` nonzero) — mumott's curvature-correct
        quadrature is CPU-only and re-running it every call is wasteful once
        the geometry is fixed. Ignored for SAXS datasets. See
        ``SmarttDataContainer.get_cache_dir()``.

    Returns
    -------
    dict
        ``reconstruction`` — ``(X, Y, Z, C)`` float32 CPU tensor of SH
        coefficients, evaluated without annealing masks.

        ``model`` — the trained :class:`SaxsNafField` (still on ``device``).

        ``losses`` — list of per-step scalar loss values (data + reg).

        ``time`` — wall-clock seconds for the training loop.

        ``iterations`` — number of steps completed.

        ``encoding`` — human-readable description of the hash-grid levels.

        ``rank_diagnostics`` — decoder/encoder/shape effective-rank report
        (see :mod:`smartt.saxs_naf.diagnostics`), or omitted if
        ``compute_rank_diagnostics=False``.
    """
    import time as _time
    if loss_type not in ("huber", "mse"):
        raise ValueError(f"loss_type must be 'huber' or 'mse', got {loss_type!r}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    geometry = dc.geometry
    volume_shape = tuple(geometry.volume_shape)
    field_kwargs = dict(field_kwargs or {})

    # Full target data + boolean validity mask (matches the notebook).
    full_target = torch.tensor(
        dc.projections.data, device=device, dtype=torch.float32
    )
    full_weights = torch.tensor(
        dc.projections.weights, device=device, dtype=torch.float32
    ).bool()
    n_proj = full_target.shape[0]

    # Rescale so the data loss is O(1) regardless of this dataset's raw
    # intensity range (q-shells span [0, ~50] up to [0, ~1e6]).  The forward
    # model is linear in the SH coefficients, so training against a scaled
    # target is equivalent to training against the original one up to a
    # global rescale of the output, which is undone below.
    target_scale = 1.0
    if normalize_target:
        target_scale = float(full_target[full_weights].abs().mean().clamp_min(1e-8))
        full_target = full_target / target_scale

    held_eval = None
    if held_out_dc is not None:
        from smartt.projectors import build_mumott_projector
        held_target = torch.tensor(
            held_out_dc.projections.data, device=device, dtype=torch.float32
        )
        held_weights = torch.tensor(
            held_out_dc.projections.weights, device=device, dtype=torch.float32
        ).bool()
        if normalize_target:
            held_target = held_target / target_scale
        held_eval = {
            "projector": build_mumott_projector(held_out_dc.geometry, device=device),
            "Y_int": _full_dataset_Y_int(held_out_dc, ell_max, device, cache_dir=cache_dir),
            "target": held_target,
            "mask": held_weights,
        }

    # Build the field (cold start). Named capacity/init args are defaults that
    # an explicit field_kwargs entry (same key) overrides.
    field_kwargs = {
        "n_features_per_level": n_features_per_level,
        "hidden_dim": hidden_dim,
        "n_hidden_layers": n_hidden_layers,
        "head_init_std": head_init_std,
        **field_kwargs,
    }
    model = SaxsNafField(volume_shape, ell_max=ell_max, **field_kwargs).to(device)

    if warm_start_state_dict is not None:
        # Reset any baked-in output_scale from the source run so this call's
        # normalize_target/set_output_scale bookkeeping is self-consistent
        # (see warm_start_state_dict docstring) — training resumes against a
        # freshly-(re)normalised target either way.
        sd = dict(warm_start_state_dict)
        sd["output_scale"] = torch.tensor(1.0)
        model.load_state_dict(sd)

    # Resume takes priority over warm_start_state_dict if both are given (it
    # means this exact run was already partway through). Optimizer state is
    # restored once the optimizer exists, further down.
    _ckpt = None
    start_step = 0
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        _ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model"])
        start_step = int(_ckpt["step"]) + 1
        if verbose:
            print(f"[resume] Restoring {checkpoint_path} at step {_ckpt['step']}; "
                  f"continuing at step {start_step}/{n_iterations}.")

    # SH integration matrix, computed once for the full dataset (curvature-aware
    # for WAXS geometries; see _full_dataset_Y_int) and sliced per chunk below.
    Y_int_full = _full_dataset_Y_int(dc, ell_max, device, cache_dir=cache_dir)

    # Initial projector pool.
    perm = np.random.permutation(n_proj)
    pool = _build_projector_pool(
        dc, perm, batch_size, full_target, full_weights, device, ell_max, Y_int_full
    )
    n_chunks = len(pool)

    # Calibrate the cold-start c00 so the isotropic component matches the data scale.
    # Skipped entirely when warm-starting: the loaded weights already encode a
    # correctly-scaled object, and recalibrating would clobber it.
    if warm_start_state_dict is None and start_step == 0 and cold_start and calibrate_c00:
        with torch.no_grad():
            ch = pool[0]
            coeffs = model()                      # c00 = softplus(bias) (=1), rest ~0
            spatial = ch["projector"](coeffs)
            pred = torch.einsum("nijc,nmc->nijm", spatial, ch["Y_int"])
            m = ch["mask"]
            pred_mean = float(pred[m].mean().clamp_min(1e-8))
            tgt_mean = float(full_target[full_weights].mean())
            # Forward is linear in c00 (others≈0) ⇒ scale c00 to match means.
            model.set_c00_init(max(tgt_mean / pred_mean, 1e-6))
        if verbose:
            print(f"Calibrated c00 init (mean target/pred = {tgt_mean:.3e}/{pred_mean:.3e}).")

    # Two LR groups: hash-grid tables (fast) vs trunk+head (slow). The grid
    # tables start at Instant-NGP scale (std=1e-4, see GridLevel), much
    # smaller than a freshly-initialised MLP's activations, so a shared LR
    # under-trains the grid relative to the trunk — see grid_lr_multiplier
    # docstring.
    grid_param_ids = {id(p) for p in model.encoding.parameters()}
    grid_params = [p for p in model.parameters() if id(p) in grid_param_ids]
    other_params = [p for p in model.parameters() if id(p) not in grid_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": grid_params, "lr": lr * grid_lr_multiplier},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=0.0,
    )
    if _ckpt is not None:
        optimizer.load_state_dict(_ckpt["optimizer"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _cosine_warmup_lr(s, n_iterations, warmup_steps)
    )
    for _ in range(start_step):
        scheduler.step()
    annealer = Annealer(
        n_levels=model.encoding.n_levels,
        lm_list=model.lm_list,
        total_steps=n_iterations,
        spatial_on=anneal_spatial,
        angular_on=anneal_angular,
        spatial_frac=spatial_frac,
        angular_frac=angular_frac,
        stochastic_angular=stochastic_angular,
    )
    n_voxels_total = int(np.prod(volume_shape))

    # Seed "already unlocked" from one step before the resume point (a level
    # revealed by then stays revealed — the ramp is monotone), so resuming
    # mid-anneal doesn't re-reset a level that has already been training a
    # while post-unlock.
    grid_unlocked = [False] * model.encoding.n_levels
    if reset_grid_optimizer_state_on_unlock and start_step > 0:
        _prev_lw = annealer.level_weights(start_step - 1)
        if _prev_lw is not None:
            grid_unlocked = [bool(_prev_lw[i] > 0) for i in range(model.encoding.n_levels)]

    bg_mask_t = None
    if background_mask is not None and reg_weight_background > 0:
        bg_mask_t = torch.tensor(
            ~np.asarray(background_mask, dtype=bool), dtype=torch.bool, device=device
        )

    if verbose:
        print(f"SAXS-NAF: vol={volume_shape} C={model.num_coeffs} "
              f"proj={n_proj} chunks={n_chunks} | {model.encoding.describe()}")

    losses: List[float] = []
    holdout_curve: List[Tuple[int, float]] = []
    best_holdout_nrmse = float("inf")
    best_step = -1
    best_state_dict = None
    _no_improve_evals = 0
    _stopped_early = False
    try:
        from tqdm import tqdm
        iterator = tqdm(range(start_step, n_iterations), initial=start_step,
                         total=n_iterations, disable=not verbose)
    except Exception:
        iterator = range(start_step, n_iterations)

    t0 = _time.time()
    for step in iterator:
        # Periodically reshuffle the partition and rebuild the projector pool.
        if step > 0 and reshuffle_every and step % reshuffle_every == 0:
            perm = np.random.permutation(n_proj)
            pool = _build_projector_pool(
                dc, perm, batch_size, full_target, full_weights, device, ell_max, Y_int_full
            )
            n_chunks = len(pool)

        ch = pool[step % n_chunks]
        level_w, ell_mask = annealer.masks(step, model.num_coeffs, n_voxels_total, device)
        if level_w is not None:
            level_w = level_w.to(device)
        if ell_mask is not None:
            ell_mask = ell_mask.to(device)

        if reset_grid_optimizer_state_on_unlock and level_w is not None:
            for _i in range(model.encoding.n_levels):
                if not grid_unlocked[_i] and float(level_w[_i]) > 0.0:
                    grid_unlocked[_i] = True
                    _tbl = model.encoding.levels[_i].table
                    _st = optimizer.state.get(_tbl)
                    if _st:
                        _st["step"] = torch.zeros_like(_st["step"]) if torch.is_tensor(_st["step"]) else 0
                        _st["exp_avg"].zero_()
                        _st["exp_avg_sq"].zero_()

        optimizer.zero_grad(set_to_none=True)
        coeffs = model(level_weights=level_w, ell_mask=ell_mask)           # (X,Y,Z,C)
        spatial = ch["projector"](coeffs)                                  # (I,J,K,C)
        pred = torch.einsum("nijc,nmc->nijm", spatial, ch["Y_int"])        # (I,J,K,M)

        m = ch["mask"]
        if loss_type == "huber":
            data_loss = torch.nn.functional.huber_loss(pred[m], ch["target"][m], delta=huber_delta)
        else:
            data_loss = torch.nn.functional.mse_loss(pred[m], ch["target"][m])
        reg_loss = reg_weight_sh * model.sh_regularization(coeffs) if reg_weight_sh > 0 else 0.0
        tv_loss = reg_weight_tv * model.tv_regularization(coeffs) if reg_weight_tv > 0 else 0.0
        bg_loss = (
            reg_weight_background * (coeffs[bg_mask_t] ** 2).mean()
            if bg_mask_t is not None else 0.0
        )
        loss = data_loss + reg_loss + tv_loss + bg_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if checkpoint_path is not None and (
            step % checkpoint_every == 0 or step == n_iterations - 1
        ):
            tmp_path = f"{checkpoint_path}.tmp"
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                tmp_path,
            )
            os.replace(tmp_path, checkpoint_path)  # atomic — never leaves a half-written file

        if held_eval is not None and (
            step % holdout_eval_every == 0 or step == n_iterations - 1
        ):
            with torch.no_grad():
                full_coeffs = model()  # unmasked, full ell — the "deployable" state right now
                held_spatial = held_eval["projector"](full_coeffs)
                held_pred = torch.einsum("nijc,nmc->nijm", held_spatial, held_eval["Y_int"])
                hm = held_eval["mask"]
                h_diff = held_pred[hm] - held_eval["target"][hm]
                h_nrmse = float(
                    torch.sqrt((h_diff ** 2).mean()) / (held_eval["target"][hm].std() + 1e-8)
                )
            holdout_curve.append((step, h_nrmse))
            if h_nrmse < best_holdout_nrmse:
                best_holdout_nrmse = h_nrmse
                best_step = step
                best_state_dict = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                _no_improve_evals = 0
            else:
                _no_improve_evals += 1
            if early_stop_patience is not None and _no_improve_evals >= early_stop_patience:
                if verbose:
                    print(
                        f"[early-stop] holdout_nrmse hasn't improved in "
                        f"{early_stop_patience} evals (best={best_holdout_nrmse:.4f} "
                        f"@ step {best_step}); stopping at step {step}."
                    )
                _stopped_early = True
                break

        losses.append(float(loss.detach()))
        if verbose and hasattr(iterator, "set_postfix"):
            last_lr = scheduler.get_last_lr()
            postfix = {
                "loss": f"{float(loss.detach()):.3e}",
                "data": f"{float(data_loss.detach()):.3e}",
                "lr_grid": f"{last_lr[0]:.2e}",
                "lr_mlp": f"{last_lr[1]:.2e}",
            }
            if reg_weight_sh > 0:
                postfix["sh"] = f"{float(reg_loss.detach()):.3e}"
            if reg_weight_tv > 0:
                postfix["tv"] = f"{float(tv_loss.detach()):.3e}"
            if bg_mask_t is not None:
                postfix["bg"] = f"{float(bg_loss.detach()):.3e}"
            if holdout_curve:
                postfix["holdout"] = f"{holdout_curve[-1][1]:.4f}"
                postfix["best"] = f"{best_holdout_nrmse:.4f}"
            iterator.set_postfix(**postfix)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = _time.time() - t0

    if use_best_checkpoint and best_state_dict is not None:
        if verbose:
            print(f"[best-checkpoint] Loading step {best_step} "
                  f"(holdout_nrmse={best_holdout_nrmse:.4f}) over the final step's weights.")
        model.load_state_dict(best_state_dict)

    if normalize_target:
        # Bake the rescale into the model itself so it natively emits
        # physical-unit coefficients for any later caller (checkpointing,
        # super-resolution querying), not just the ``reconstruction`` below.
        model.set_output_scale(target_scale)

    with torch.no_grad():
        final = model().detach().cpu()      # (X, Y, Z, C), full ℓ, no masks

    result = {
        "reconstruction": final,
        "model": model,
        "losses": losses,
        "time": total_time,
        "iterations": n_iterations,
        "encoding": model.encoding.describe(),
        "target_scale": target_scale,
    }
    if held_eval is not None:
        result["holdout_curve"] = holdout_curve
        result["best_step"] = best_step
        result["best_holdout_nrmse"] = best_holdout_nrmse
        result["stopped_early"] = _stopped_early
    if compute_rank_diagnostics:
        from .diagnostics import full_rank_report
        try:
            result["rank_diagnostics"] = full_rank_report(model, final.numpy())
        except np.linalg.LinAlgError as e:
            # SVD can occasionally fail to converge on a pathological/degenerate
            # slice of voxels (e.g. a near-singular subsample); this is a
            # diagnostic-only computation, not the reconstruction itself, so a
            # numerical hiccup here must not crash an otherwise-complete,
            # multi-hour run (and, worse, checkpoint_path/resume would just
            # recompute the exact same failing SVD on every retry forever).
            if verbose:
                print(f"[warning] rank_diagnostics failed ({e}); leaving it out of the result.")
            result["rank_diagnostics"] = None
    return result


def saxs_naf_two_phase_reconstruction(
    dc,
    ell_max: int = 8,
    held_out_dc=None,
    # Phase 1: cold start, recovers a clean object/background split before
    # any per-voxel RSM shape is asked of the model.
    phase1_n_iterations: int = 2001,
    phase1_lr: float = 2e-4,
    phase1_batch_size: int = 100,
    phase1_spatial_frac: float = 0.5,
    phase1_angular_frac: float = 0.6,
    phase1_head_init_std: float = 0.0,
    phase1_stochastic_angular: bool = True,
    # Phase 2: warm start, full capacity unlocked, aggressively fits per-voxel
    # RSM shape on top of phase 1's object/background split. Mild regularisers
    # and held-out-tracked early stopping/best-checkpoint keep it from
    # overfitting the training projections' noise (or, for noiseless data, the
    # missing-wedge inverse problem's null space) once it's past the point of
    # genuine improvement — see project memory ``project_saxs_naf_design``.
    phase2_n_iterations: int = 1500,
    phase2_lr: float = 5e-3,
    phase2_batch_size: int = 100,
    phase2_reg_target_frac_sh: float = 0.02,
    phase2_reg_target_frac_tv: float = 0.05,
    phase2_holdout_eval_every: int = 25,
    phase2_early_stop_patience: Optional[int] = 8,
    # Shared field capacity / training knobs
    n_features_per_level: int = 8,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
    grid_lr_multiplier: float = 10.0,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    normalize_target: bool = True,
    compute_rank_diagnostics: bool = True,
    field_kwargs: Optional[dict] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 100,
    resume: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Standard two-phase NAF recipe: cold-start phase 1 + regularised, early-stopped phase 2.

    This is the recommended entry point for SAXS-NAF reconstruction — see
    project memory for the empirical trail (capacity-matched two-phase →
    stochastic angular reveal → phase-2 anti-overfitting). ``saxs_naf_reconstruction``
    (a single training run) is the primitive this is built from; call it
    directly only if you need a bespoke recipe (e.g. :mod:`reconstruct_multiq`).

    Phase 1 cold-starts a NEW-capacity model (``n_features_per_level=8``,
    ``hidden_dim=64``, ``n_hidden_layers=2``, ``grid_lr_multiplier=10``) with
    ``head_init_std=0`` (exact rank-1 start) and slow spatial+angular
    annealing, using the stochastic per-voxel angular reveal (see
    ``Annealer.ell_mask``) rather than the deterministic per-channel one —
    every SH degree gets a real, full-strength gradient from step 0, avoiding
    the "stale Adam step counter" problem a hard per-channel unlock has.  This
    combination reliably breaks the shared-``head.weight`` rank-collapse
    degenerate optimum (same RSM shape at every voxel) that a single-phase run
    is prone to.

    Phase 2 warm-starts from phase 1's weights with a *fresh* optimiser at a
    much higher LR (25× phase 1's, the value found necessary to actually move
    off the phase-1 solution) and every annealing constraint lifted — full
    spatial + angular capacity is available from phase 2's step 0.  Left
    unchecked this aggressively fits per-voxel RSM shape onto the projections'
    noise (or, for noiseless synthetic data, the missing-wedge inverse
    problem's null space) once the genuine signal is already captured — the
    "grain"/overfitting failure mode this recipe exists to prevent.  Two
    independent, complementary levers guard against it:

    * ``phase2_reg_target_frac_sh``/``phase2_reg_target_frac_tv`` — mild
      ℓ(ℓ+1) angular and spatial-TV smoothness priors
      (:meth:`SaxsNafField.sh_regularization`/``tv_regularization``).  These
      are mean- (not sum-) normalised over voxels/channels so they don't
      scale with volume size, but that alone doesn't make a *fixed* weight
      portable across datasets — the coefficient field's own absolute scale
      also varies a lot (c00 ranges from ~1e-3 to ~2e-1 across datasets
      depending on projector geometry/path length), and since both
      regularisers are quadratic in the coefficients, a fixed weight can be
      2-3 orders of magnitude too strong or too weak depending on that scale
      (observed: a weight tuned on b411/steel-wire-waxs/cf-carolina was
      50-900× too strong on nielsen-t/nielsen-mammoth/zenodo). So instead of
      a fixed weight, phase 2 auto-calibrates: right after phase 1 finishes,
      it evaluates the *raw* (unweighted) regulariser value on phase 1's own
      output and phase 1's own trailing data loss, then solves for the
      weight that makes each regulariser contribute exactly
      ``phase2_reg_target_frac_{sh,tv}`` of that data loss — a dataset-scale-
      invariant target instead of a dataset-scale-dependent raw weight. The
      defaults (0.02 / 0.05) reproduce roughly the same *relative* strength
      that was manually tuned on b411/cf-carolina. Even correctly calibrated
      this lever isn't uniformly positive — it still slightly hurt
      steel-wire-waxs (a WAXS dataset) in testing — so it's a starting point,
      not a guarantee; set the target fraction to ``0.0`` to disable if a
      dataset shows that pattern (regularised phase 2 doing worse than
      unregularised on held-out reprojection).
    * ``held_out_dc`` + ``phase2_early_stop_patience`` — if a held-out split is
      given, phase 2 tracks held-out reprojection NRMSE live and keeps the
      best-seen checkpoint (see ``saxs_naf_reconstruction``'s
      ``held_out_dc``/``use_best_checkpoint`` docstring) instead of blindly
      running to ``phase2_n_iterations``. Unlike the regularisers, this lever
      was never observed to hurt in testing (worst case: it's a no-op because
      the last step happens to be the best) — pass ``held_out_dc=None`` only
      if you have no holdout split to spare, e.g. an evaluation-critical
      dataset.

    Parameters
    ----------
    dc : mumott DataContainer
        Training subset (already holdout-split if ``held_out_dc`` is given
        from the same split — pass the *train* half here).
    held_out_dc : mumott DataContainer, optional
        Disjoint held-out split for live phase-2 monitoring — see
        ``saxs_naf_reconstruction``'s docstring. Strongly recommended; pass
        ``None`` only if no holdout can be spared.
    checkpoint_dir : str, optional
        If given, phase 1 and phase 2 each get their own checkpoint file
        (``{checkpoint_dir}/phase1_ckpt.pt``, ``phase2_ckpt.pt``) so a crash
        resumes from the last completed step of whichever phase was running
        (``resume=True``) — including the degenerate case where phase 1 had
        already finished, in which case its "resume" is a near-instant
        state-dict load, not a re-run. Note: a resume that lands mid-phase-2
        restarts held-out best-tracking from the resume point, not from
        phase 2's true start — the recorded "best" is only best among
        evaluations since the resume.
    Other parameters mirror ``saxs_naf_reconstruction`` — see its docstring
    for anything not phase-prefixed here (``ell_max``, field capacity,
    ``loss_type``/``huber_delta``/``normalize_target``, ``seed``, ``device``,
    ``verbose``, ``cache_dir``).

    Returns
    -------
    dict
        ``reconstruction``, ``model`` — phase 2's final (or best-checkpoint)
        output, in the same units/shape as ``saxs_naf_reconstruction``'s.

        ``phase1``, ``phase2`` — the full result dict from each phase's
        ``saxs_naf_reconstruction`` call (rank diagnostics, losses, holdout
        curve if applicable, etc.).

        ``time`` — combined wall-clock seconds for both phases.

        ``phase2_reg_weight_sh``, ``phase2_reg_weight_tv`` — the actual
        (auto-calibrated) weights used, for logging/inspection.

        ``phase2_reg_calibration`` — the raw ingredients behind that
        calibration (``raw_sh``, ``raw_tv``, ``phase1_data_loss``).
    """
    import time as _time
    t0 = _time.time()

    ckpt1 = f"{checkpoint_dir}/phase1_ckpt.pt" if checkpoint_dir else None
    ckpt2 = f"{checkpoint_dir}/phase2_ckpt.pt" if checkpoint_dir else None

    if verbose:
        print(f"=== Phase 1: cold start, stochastic-angular reveal, "
              f"{phase1_n_iterations} iters ===")
    phase1 = saxs_naf_reconstruction(
        dc, ell_max=ell_max,
        n_iterations=phase1_n_iterations, lr=phase1_lr, batch_size=phase1_batch_size,
        reg_weight_sh=0.0, reg_weight_tv=0.0,
        loss_type=loss_type, huber_delta=huber_delta, normalize_target=normalize_target,
        head_init_std=phase1_head_init_std,
        spatial_frac=phase1_spatial_frac, angular_frac=phase1_angular_frac,
        anneal_spatial=True, anneal_angular=True,
        stochastic_angular=phase1_stochastic_angular,
        n_features_per_level=n_features_per_level, hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers, grid_lr_multiplier=grid_lr_multiplier,
        compute_rank_diagnostics=compute_rank_diagnostics,
        field_kwargs=field_kwargs,
        checkpoint_path=ckpt1, checkpoint_every=checkpoint_every, resume=resume,
        device=device, verbose=verbose, seed=seed, cache_dir=cache_dir,
    )

    # Auto-calibrate phase 2's regularisation weights from phase 1's own
    # scale, rather than using a fixed weight tuned on a different dataset's
    # coefficient scale (see docstring). Evaluated with output_scale
    # temporarily reset to 1.0 to match the *training*-scale coefficients
    # phase 2 will actually see when it warm-starts (warm_start_state_dict
    # always resets output_scale to 1.0 — see that param's docstring), not
    # phase 1's returned physical-unit reconstruction.
    with torch.no_grad():
        _orig_scale = float(phase1["model"].output_scale)
        phase1["model"].set_output_scale(1.0)
        _calib_coeffs = phase1["model"]()
        _raw_sh = float(phase1["model"].sh_regularization(_calib_coeffs))
        _raw_tv = float(phase1["model"].tv_regularization(_calib_coeffs))
        phase1["model"].set_output_scale(_orig_scale)
    _calib_data_loss = float(np.mean(phase1["losses"][-20:]))
    # A degenerate raw value (e.g. phase 1 barely trained, so tv_regularization
    # is still ~0 at near-cold-start) would blow the weight up arbitrarily —
    # clamp to a range wide enough for every real calibration seen so far
    # (weights spanning roughly 1e-1 to 1e3 across very different datasets)
    # with headroom, rather than trusting an unbounded division.
    _eps = 1e-8
    _max_weight = 1e4
    phase2_reg_weight_sh = min(phase2_reg_target_frac_sh * _calib_data_loss / max(_raw_sh, _eps), _max_weight)
    phase2_reg_weight_tv = min(phase2_reg_target_frac_tv * _calib_data_loss / max(_raw_tv, _eps), _max_weight)

    if verbose:
        print(f"=== Phase 2: warm start, full unlock, lr={phase2_lr}, "
              f"{phase2_n_iterations} iters, auto-calibrated reg_sh={phase2_reg_weight_sh:.4g} "
              f"(raw={_raw_sh:.4g}, target_frac={phase2_reg_target_frac_sh}), "
              f"reg_tv={phase2_reg_weight_tv:.4g} (raw={_raw_tv:.4g}, "
              f"target_frac={phase2_reg_target_frac_tv}), phase1_data_loss={_calib_data_loss:.4g}, "
              f"early_stop={phase2_early_stop_patience if held_out_dc is not None else 'off (no held_out_dc)'} ===")
    phase2 = saxs_naf_reconstruction(
        dc, ell_max=ell_max,
        n_iterations=phase2_n_iterations, lr=phase2_lr, batch_size=phase2_batch_size,
        reg_weight_sh=phase2_reg_weight_sh, reg_weight_tv=phase2_reg_weight_tv,
        loss_type=loss_type, huber_delta=huber_delta, normalize_target=normalize_target,
        anneal_spatial=False, anneal_angular=False,
        warm_start_state_dict=phase1["model"].state_dict(),
        head_init_std=phase1_head_init_std,
        n_features_per_level=n_features_per_level, hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers, grid_lr_multiplier=grid_lr_multiplier,
        compute_rank_diagnostics=compute_rank_diagnostics,
        field_kwargs=field_kwargs,
        held_out_dc=held_out_dc,
        holdout_eval_every=phase2_holdout_eval_every,
        early_stop_patience=(phase2_early_stop_patience if held_out_dc is not None else None),
        use_best_checkpoint=(held_out_dc is not None),
        checkpoint_path=ckpt2, checkpoint_every=checkpoint_every, resume=resume,
        device=device, verbose=verbose,
        seed=(None if seed is None else seed + 1),
        cache_dir=cache_dir,
    )

    return {
        "reconstruction": phase2["reconstruction"],
        "model": phase2["model"],
        "phase1": phase1,
        "phase2": phase2,
        "time": phase1["time"] + phase2["time"],
        "phase2_reg_weight_sh": phase2_reg_weight_sh,
        "phase2_reg_weight_tv": phase2_reg_weight_tv,
        "phase2_reg_calibration": {
            "raw_sh": _raw_sh, "raw_tv": _raw_tv, "phase1_data_loss": _calib_data_loss,
        },
    }
