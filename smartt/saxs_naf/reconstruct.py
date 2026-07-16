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
* **Boolean-masked MSE** data term + ℓ(ℓ+1) angular penalty.
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


def _build_projector_pool(
    dc,
    perm: np.ndarray,
    batch_size: int,
    full_target: torch.Tensor,
    full_weights: torch.Tensor,
    device: torch.device,
    ell_max: int,
) -> List[dict]:
    """Partition projections into chunks; build one projector each (once).

    The SH integration matrix ``Y_int`` (shape ``(N, M, C)``) is precomputed
    once per chunk from the fixed probed coordinates and stored as a GPU
    tensor.  During training each step uses a single ``torch.einsum`` instead
    of re-running the per-step Python loop inside ``forward_quadrature``.
    """
    from smartt.projectors import build_mumott_projector
    from smartt.shutils.evaulate_sh import precompute_Y_int

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
                "Y_int": precompute_Y_int(
                    sub.geometry.probed_coordinates,
                    ell_max=ell_max,
                    device=device,
                ),
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
    anneal_spatial: bool = True,
    anneal_angular: bool = True,
    spatial_frac: float = 0.5,
    angular_frac: float = 0.6,
    cold_start: bool = True,
    calibrate_c00: bool = True,
    normalize_target: bool = True,
    field_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
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
    field_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`SaxsNafField`
        (e.g. ``hidden_dim``, ``n_hidden_layers``, ``n_levels``).
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.
    verbose : bool
        Print calibration info and show a tqdm progress bar.
    seed : int, optional
        Seed for NumPy and PyTorch RNGs (for reproducible pool shuffles).

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
    """
    import time as _time
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

    # Build the field (cold start).
    model = SaxsNafField(volume_shape, ell_max=ell_max, **field_kwargs).to(device)

    # Initial projector pool.
    perm = np.random.permutation(n_proj)
    pool = _build_projector_pool(dc, perm, batch_size, full_target, full_weights, device, ell_max)
    n_chunks = len(pool)

    # Calibrate the cold-start c00 so the isotropic component matches the data scale.
    if cold_start and calibrate_c00:
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _cosine_warmup_lr(s, n_iterations, warmup_steps)
    )
    annealer = Annealer(
        n_levels=model.encoding.n_levels,
        lm_list=model.lm_list,
        total_steps=n_iterations,
        spatial_on=anneal_spatial,
        angular_on=anneal_angular,
        spatial_frac=spatial_frac,
        angular_frac=angular_frac,
    )

    if verbose:
        print(f"SAXS-NAF: vol={volume_shape} C={model.num_coeffs} "
              f"proj={n_proj} chunks={n_chunks} | {model.encoding.describe()}")

    losses: List[float] = []
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_iterations), disable=not verbose)
    except Exception:
        iterator = range(n_iterations)

    t0 = _time.time()
    for step in iterator:
        # Periodically reshuffle the partition and rebuild the projector pool.
        if step > 0 and reshuffle_every and step % reshuffle_every == 0:
            perm = np.random.permutation(n_proj)
            pool = _build_projector_pool(
                dc, perm, batch_size, full_target, full_weights, device, ell_max
            )
            n_chunks = len(pool)

        ch = pool[step % n_chunks]
        level_w, ell_mask = annealer.masks(step, model.num_coeffs)
        if level_w is not None:
            level_w = level_w.to(device)
        if ell_mask is not None:
            ell_mask = ell_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        coeffs = model(level_weights=level_w, ell_mask=ell_mask)           # (X,Y,Z,C)
        spatial = ch["projector"](coeffs)                                  # (I,J,K,C)
        pred = torch.einsum("nijc,nmc->nijm", spatial, ch["Y_int"])        # (I,J,K,M)

        m = ch["mask"]
        data_loss = torch.nn.functional.mse_loss(pred[m], ch["target"][m])
        reg_loss = reg_weight_sh * model.sh_regularization(coeffs) if reg_weight_sh > 0 else 0.0
        tv_loss = reg_weight_tv * model.tv_regularization(coeffs) if reg_weight_tv > 0 else 0.0
        loss = data_loss + reg_loss + tv_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(float(loss.detach()))
        if verbose and hasattr(iterator, "set_postfix"):
            postfix = {
                "loss": f"{float(loss.detach()):.3e}",
                "data": f"{float(data_loss.detach()):.3e}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if reg_weight_sh > 0:
                postfix["sh"] = f"{float(reg_loss.detach()):.3e}"
            if reg_weight_tv > 0:
                postfix["tv"] = f"{float(tv_loss.detach()):.3e}"
            iterator.set_postfix(**postfix)

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = _time.time() - t0

    if normalize_target:
        # Bake the rescale into the model itself so it natively emits
        # physical-unit coefficients for any later caller (checkpointing,
        # super-resolution querying), not just the ``reconstruction`` below.
        model.set_output_scale(target_scale)

    with torch.no_grad():
        final = model().detach().cpu()      # (X, Y, Z, C), full ℓ, no masks

    return {
        "reconstruction": final,
        "model": model,
        "losses": losses,
        "time": total_time,
        "iterations": n_iterations,
        "encoding": model.encoding.describe(),
        "target_scale": target_scale,
    }
