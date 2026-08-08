"""Joint multi-q-shell NAF reconstruction — the full 3D RSM, ``I(x, y, z, q)``.

Generalises :func:`smartt.saxs_naf.reconstruct.saxs_naf_reconstruction` from a
single q-shell to many simultaneously. Instead of one ``DataContainer``, the
caller passes a ``{qbin: DataContainer}`` dict (one per q-shell, e.g. all 79
frogbone bins). Every shell shares the *same* tomography geometry (only
scattering intensity differs per shell — see ``FrogboneDataContainer``), so:

* The projector and the SH integration matrix (``Y_int``) are built **once**
  from any one shell's geometry and reused for every shell — no per-shell
  duplication of the expensive setup.
* The field is :class:`~smartt.saxs_naf.model.SaxsNafField` with
  ``n_qshells > 1``: the same ``(x, y, z)`` hash-grid encoder as the
  single-shell model, plus a small separate 1-D hash encoder over q, whose
  features are concatenated before the shared trunk/head (see
  ``SaxsNafField`` docstring). ``forward_at_q(q_norm_batch)`` evaluates the
  full spatial grid at a handful of q-coordinates in one batched pass.
* Each training step samples a random *subset* of q-shells (``q_batch_size``
  of them) crossed with the usual random projection chunk, so one gradient
  step sees several shells' data at once instead of alternating between
  fully independent per-shell models — the whole point being that the shared
  spatial encoder (and the q encoder's interpolation) can borrow strength
  across nearby voxels *and* nearby q, rather than reconstructing each shell
  in isolation.

Setting ``q_batch_size=1`` with a single-entry ``dcs`` dict degenerates
mechanically to the single-shell recipe (mirroring ``SaxsNafField``'s exact
fallback at ``n_qshells=1``), though for a genuinely single-shell run you
should just call :func:`~smartt.saxs_naf.reconstruct.saxs_naf_reconstruction`
directly — it is simpler and has no q-encoder overhead at all.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import SaxsNafField
from .schedule import Annealer
from .reconstruct import _cosine_warmup_lr, _full_dataset_Y_int


def _build_shared_pool(
    ref_dc,
    perm: np.ndarray,
    batch_size: int,
    device: torch.device,
    ell_max: int,
    Y_int_full: torch.Tensor,
) -> List[dict]:
    """Partition projections into chunks; one projector per chunk (shared across q-shells).

    Unlike :func:`smartt.saxs_naf.reconstruct._build_projector_pool`, chunk
    entries do NOT bake in a target/mask — those differ per q-shell and are
    looked up from the big ``(n_qshells, N, J, K)`` tensors at train time —
    only ``idx`` (the projection indices in this chunk, needed for that
    lookup), the shared ``projector``, and the shared ``Y_int`` slice.
    """
    from smartt.projectors import build_mumott_projector

    n = len(perm)
    pool: List[dict] = []
    for start in range(0, n, batch_size):
        idx = np.sort(perm[start : start + batch_size])
        sub = copy.deepcopy(ref_dc)
        keep = set(int(i) for i in idx)
        for j in sorted(range(len(ref_dc.projections)), reverse=True):
            if j not in keep:
                del sub.projections[j]
        pool.append(
            {
                "projector": build_mumott_projector(sub.geometry, device=device),
                "Y_int": Y_int_full[idx],
                "idx": idx,
            }
        )
    return pool


def saxs_naf_reconstruction_multiq(
    dcs: Dict[int, object],
    q_values: Dict[int, float],
    ell_max: int = 8,
    n_iterations: int = 2000,
    lr: float = 1e-2,
    q_batch_size: int = 4,
    batch_size: int = 40,
    reshuffle_every: int = 200,
    warmup_steps: int = 50,
    reg_weight_sh: float = 0.0,
    reg_weight_tv: float = 0.0,
    reg_weight_q_tv: float = 0.0,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    anneal_spatial: bool = True,
    anneal_angular: bool = True,
    spatial_frac: float = 0.25,
    angular_frac: float = 0.35,
    stochastic_angular: bool = False,
    cold_start: bool = True,
    calibrate_c00: bool = True,
    normalize_target: bool = True,
    n_features_per_level: int = 8,
    hidden_dim: int = 64,
    n_hidden_layers: int = 2,
    head_init_std: float = 1e-3,
    grid_lr_multiplier: float = 10.0,
    q_n_levels: int = 6,
    q_n_features_per_level: int = 4,
    q_base_resolution: int = 4,
    warm_start_state_dict: Optional[dict] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 200,
    resume: bool = False,
    held_out_dcs: Optional[Dict[int, object]] = None,
    holdout_eval_every: int = 100,
    holdout_eval_qshells: int = 8,
    early_stop_patience: Optional[int] = None,
    use_best_checkpoint: bool = False,
    field_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dict:
    """Reconstruct a joint ``(X, Y, Z, Q, C)`` SH-coefficient field across q-shells.

    Parameters mirror :func:`~smartt.saxs_naf.reconstruct.saxs_naf_reconstruction`
    (same annealing/regularisation/checkpointing/early-stopping semantics — see
    its docstring for what each one does); only the multi-q-specific additions
    are documented here.

    Parameters
    ----------
    dcs : ``{qbin: DataContainer}``
        One DataContainer per q-shell used for training. All must share the
        same geometry (volume_shape, number/orientation of projections) —
        only scattering intensity differs. This is a real assumption of the
        q-resolved datasets built so far (cf-carolina, plastic-plasmonics,
        frogbone); it is checked (volume_shape + projection count) and raises
        ``ValueError`` if violated.
    q_values : ``{qbin: q}``
        Physical q value for every key in ``dcs`` (and ``held_out_dcs`` if
        given). q-shells are log-spaced (see ``FrogboneDataContainer``), so
        the q axis is normalised via ``log(q)``, not q itself, to keep
        neighbouring shells roughly equidistant in the encoder's coordinate.
    q_batch_size : int
        Number of q-shells sampled (without replacement) per training step,
        crossed with the usual random projection chunk. Training cost per
        step scales ~linearly with this (each sampled shell adds another full
        ``C``-channel projector pass), so this trades wall-clock for how many
        shells' gradients mix per step.
    held_out_dcs : ``{qbin: DataContainer}``, optional
        Held-out projection splits (e.g. one per training q-shell, from
        ``metrics.split_holdout``) for live NRMSE tracking / early stopping,
        exactly like ``held_out_dc`` in the single-shell function. A *subset*
        of ``holdout_eval_qshells`` shells (fixed once at setup, not resampled
        per eval) is used at each evaluation to bound its cost.
    q_n_levels, q_n_features_per_level, q_base_resolution :
        Forwarded to :class:`SaxsNafField`'s q-encoder — see its docstring.

    Returns
    -------
    dict
        ``model`` — the trained :class:`SaxsNafField` (``n_qshells=len(dcs)``).

        ``q_index`` — sorted list of qbins, the order used for every q-indexed
        array below and for ``model.forward_at_q``'s q-coordinate ordering.

        ``q_norm`` — ``(len(q_index),)`` normalised q-coordinates, same order.

        Other keys mirror the single-shell function (``losses``, ``time``,
        ``iterations``, ``target_scale`` — one global scale, computed across
        all shells jointly so shells stay on a shared relative footing —
        ``holdout_curve``/``best_step``/``best_holdout_nrmse``/``stopped_early``
        if ``held_out_dcs`` was given). There is no ``reconstruction`` key (a
        full ``(X,Y,Z,Q,C)`` grid is rarely all you want at once and can be
        large); call ``model.forward_at_q(q_norm)`` yourself for whichever
        q-shells you need, or use the ``sample_qshell`` helper in
        ``smartt.saxs_naf.eval_multiq``.
    """
    import time as _time

    if loss_type not in ("huber", "mse"):
        raise ValueError(f"loss_type must be 'huber' or 'mse', got {loss_type!r}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    q_index = sorted(dcs.keys())
    n_qshells = len(q_index)
    if q_batch_size > n_qshells:
        raise ValueError(f"q_batch_size={q_batch_size} > number of training q-shells ({n_qshells})")

    ref_dc = dcs[q_index[0]]
    geometry = ref_dc.geometry
    volume_shape = tuple(geometry.volume_shape)
    n_proj = len(ref_dc.projections)
    for qb in q_index[1:]:
        g = dcs[qb].geometry
        if tuple(g.volume_shape) != volume_shape or len(dcs[qb].projections) != n_proj:
            raise ValueError(
                f"q-shell {qb} geometry mismatch (expected volume_shape={volume_shape}, "
                f"n_proj={n_proj}) — all q-shells must share the same acquisition geometry."
            )

    field_kwargs = dict(field_kwargs or {})

    # log-spaced q -> [0, 1], shared for training AND held-out shells (so a
    # held-out q value that happens to coincide with a training one — or an
    # interpolation test where it deliberately does not — normalises
    # consistently either way).
    log_q = np.log(np.array([q_values[qb] for qb in q_index], dtype=np.float64))
    log_q_min, log_q_max = float(log_q.min()), float(log_q.max())
    q_span = max(log_q_max - log_q_min, 1e-12)

    def _q_norm(qb: int) -> float:
        return float((np.log(q_values[qb]) - log_q_min) / q_span)

    q_norm_t = torch.tensor([_q_norm(qb) for qb in q_index], dtype=torch.float32, device=device)

    # Stack every shell's target once: (n_qshells, N_proj, J, K, M).
    # Frogbone-scale data (79 shells x 240 proj x 73x100x8) is ~0.5 GB per
    # tensor — cheap to keep resident on GPU.
    all_target = torch.stack(
        [torch.tensor(dcs[qb].projections.data, device=device, dtype=torch.float32) for qb in q_index],
        dim=0,
    )
    all_weights = torch.stack(
        [torch.tensor(dcs[qb].projections.weights, device=device, dtype=torch.float32).bool() for qb in q_index],
        dim=0,
    )

    # Normalise PER Q-SHELL, not globally: SAXS intensity falls off steeply
    # with q (frogbone spans ~4 orders of magnitude from the lowest to the
    # highest shell), so a single global scale would squash high-q shells to
    # a sliver of the Huber loss's O(1) working range and the model would
    # effectively never see their gradient. Dividing each shell by its own
    # mean(|target|) puts every shell on equal O(1) footing for the loss,
    # leaving the (physically real, but scientifically uninteresting here)
    # overall q-decay to be recovered afterwards from ``target_scale_by_q``
    # rather than asking the shared trunk/head to represent 4 decades of
    # scale on top of learning shape. See ``fit_scale_trend`` in
    # ``eval_multiq.py`` for recovering this trend at an arbitrary/unseen q.
    target_scale_by_q: Dict[int, float] = {qb: 1.0 for qb in q_index}
    if normalize_target:
        for i, qb in enumerate(q_index):
            s = float(all_target[i][all_weights[i]].abs().mean().clamp_min(1e-8))
            target_scale_by_q[qb] = s
            all_target[i] = all_target[i] / s

    held_eval = None
    if held_out_dcs is not None:
        held_index = sorted(held_out_dcs.keys())
        rng = np.random.default_rng(0)
        eval_index = sorted(rng.choice(held_index, size=min(holdout_eval_qshells, len(held_index)), replace=False).tolist())
        held_target_by_q = {}
        held_weights_by_q = {}
        held_Y_int_by_q = {}
        held_projector_by_q = {}
        from smartt.projectors import build_mumott_projector
        for qb in eval_index:
            hdc = held_out_dcs[qb]
            t = torch.tensor(hdc.projections.data, device=device, dtype=torch.float32)
            if normalize_target:
                # Same q-bin as a training shell (just a disjoint projection
                # split) -> reuse that shell's own scale, not a global one.
                t = t / target_scale_by_q.get(qb, 1.0)
            held_target_by_q[qb] = t
            held_weights_by_q[qb] = torch.tensor(hdc.projections.weights, device=device, dtype=torch.float32).bool()
            held_Y_int_by_q[qb] = _full_dataset_Y_int(hdc, ell_max, device, cache_dir=cache_dir)
            held_projector_by_q[qb] = build_mumott_projector(hdc.geometry, device=device)
        held_eval = dict(
            eval_index=eval_index, target=held_target_by_q, weights=held_weights_by_q,
            Y_int=held_Y_int_by_q, projector=held_projector_by_q,
        )

    field_kwargs = {
        "n_features_per_level": n_features_per_level,
        "hidden_dim": hidden_dim,
        "n_hidden_layers": n_hidden_layers,
        "head_init_std": head_init_std,
        "n_qshells": n_qshells,
        "q_n_levels": q_n_levels,
        "q_n_features_per_level": q_n_features_per_level,
        "q_base_resolution": q_base_resolution,
        **field_kwargs,
    }
    model = SaxsNafField(volume_shape, ell_max=ell_max, **field_kwargs).to(device)

    if warm_start_state_dict is not None:
        sd = dict(warm_start_state_dict)
        sd["output_scale"] = torch.tensor(1.0)
        model.load_state_dict(sd)

    _ckpt = None
    start_step = 0
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        _ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model"])
        start_step = int(_ckpt["step"]) + 1
        if verbose:
            print(f"[resume] Restoring {checkpoint_path} at step {_ckpt['step']}; "
                  f"continuing at step {start_step}/{n_iterations}.")

    Y_int_full = _full_dataset_Y_int(ref_dc, ell_max, device, cache_dir=cache_dir)
    perm = np.random.permutation(n_proj)
    pool = _build_shared_pool(ref_dc, perm, batch_size, device, ell_max, Y_int_full)
    n_chunks = len(pool)

    if warm_start_state_dict is None and start_step == 0 and cold_start and calibrate_c00:
        with torch.no_grad():
            ch = pool[0]
            q0 = q_norm_t[:1]
            coeffs = model.forward_at_q(q0)[0]           # (X,Y,Z,C), c00=softplus(bias)≈1, rest≈0
            spatial = ch["projector"](coeffs)
            pred = torch.einsum("nijc,nmc->nijm", spatial, ch["Y_int"])
            idx0 = ch["idx"]
            m0 = all_weights[0][idx0]
            pred_mean = float(pred[m0].mean().clamp_min(1e-8))
            # Shell 0's sampled chunk only -- NOT all_target[all_weights] over
            # every shell. Boolean-indexing the full (n_qshells, N, J, K, M)
            # stack at once makes PyTorch materialise an internal index buffer
            # sized off the TOTAL element count, not the masked-True count;
            # for frogbone (~0.5 GB stacked) that was negligible, but for c4's
            # larger/more-numerous shells it tried to allocate ~183 GiB and
            # OOM'd. Since normalize_target already rescales every shell to
            # its own masked mean ~1.0, shell 0 alone is representative.
            tgt_mean = float(all_target[0][idx0][m0].mean().clamp_min(1e-8))
            model.set_c00_init(max(tgt_mean / pred_mean, 1e-6))
        if verbose:
            print(f"Calibrated c00 init (mean target/pred = {tgt_mean:.3e}/{pred_mean:.3e}).")

    grid_param_ids = {id(p) for p in model.encoding.parameters()}
    if model.q_encoding is not None:
        grid_param_ids |= {id(p) for p in model.q_encoding.parameters()}
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
    n_spatial_voxels = int(np.prod(volume_shape))

    if verbose:
        print(f"SAXS-NAF multi-q: vol={volume_shape} C={model.num_coeffs} n_qshells={n_qshells} "
              f"q_batch={q_batch_size} proj={n_proj} chunks={n_chunks} | spatial {model.encoding.describe()} "
              f"| q {model.q_encoding.describe() if model.q_encoding is not None else '(none)'}")

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
        if step > 0 and reshuffle_every and step % reshuffle_every == 0:
            perm = np.random.permutation(n_proj)
            pool = _build_shared_pool(ref_dc, perm, batch_size, device, ell_max, Y_int_full)
            n_chunks = len(pool)

        ch = pool[step % n_chunks]
        q_batch_pos = np.sort(np.random.choice(n_qshells, size=q_batch_size, replace=False))
        q_batch_norm = q_norm_t[q_batch_pos]

        level_w, ell_mask = annealer.masks(
            step, model.num_coeffs, q_batch_size * n_spatial_voxels, device
        )
        if level_w is not None:
            level_w = level_w.to(device)
        if ell_mask is not None:
            ell_mask = ell_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        coeffs = model.forward_at_q(q_batch_norm, level_weights=level_w, ell_mask=ell_mask)  # (Qb,X,Y,Z,C)
        X, Y, Z, C = coeffs.shape[1:]
        coeffs_perm = coeffs.permute(1, 2, 3, 0, 4).reshape(X, Y, Z, q_batch_size * C)
        spatial = ch["projector"](coeffs_perm)                                  # (N,J,K,Qb*C)
        Nc, Jc, Kc = spatial.shape[0], spatial.shape[1], spatial.shape[2]
        spatial = spatial.reshape(Nc, Jc, Kc, q_batch_size, C)
        pred = torch.einsum("nijqc,nmc->nijqm", spatial, ch["Y_int"])          # (N,J,K,Qb,M)

        idx = ch["idx"]
        tgt = all_target[q_batch_pos][:, idx].permute(1, 2, 3, 0, 4)            # (N,J,K,Qb,M)
        msk = all_weights[q_batch_pos][:, idx].permute(1, 2, 3, 0, 4)           # (N,J,K,Qb,M)

        if loss_type == "huber":
            data_loss = torch.nn.functional.huber_loss(pred[msk], tgt[msk], delta=huber_delta)
        else:
            data_loss = torch.nn.functional.mse_loss(pred[msk], tgt[msk])
        reg_loss = reg_weight_sh * model.sh_regularization(coeffs) if reg_weight_sh > 0 else 0.0
        tv_loss = reg_weight_tv * model.tv_regularization(coeffs) if reg_weight_tv > 0 else 0.0
        q_tv_loss = (
            reg_weight_q_tv * model.q_tv_regularization(coeffs)
            if reg_weight_q_tv > 0 and q_batch_size > 1 else 0.0
        )
        loss = data_loss + reg_loss + tv_loss + q_tv_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if checkpoint_path is not None and (
            step % checkpoint_every == 0 or step == n_iterations - 1
        ):
            tmp_path = f"{checkpoint_path}.tmp"
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                 "q_index": q_index, "log_q_min": log_q_min, "log_q_max": log_q_max,
                 "target_scale_by_q": target_scale_by_q},
                tmp_path,
            )
            os.replace(tmp_path, checkpoint_path)

        if held_eval is not None and (
            step % holdout_eval_every == 0 or step == n_iterations - 1
        ):
            with torch.no_grad():
                sq_diff_sum, sq_target_sum, n_valid = 0.0, 0.0, 0
                for qb in held_eval["eval_index"]:
                    q_norm_single = torch.tensor([_q_norm(qb)], device=device)
                    full_coeffs = model.forward_at_q(q_norm_single)[0]   # (X,Y,Z,C)
                    hs = held_eval["projector"][qb](full_coeffs)
                    hp = torch.einsum("nijc,nmc->nijm", hs, held_eval["Y_int"][qb])
                    hm = held_eval["weights"][qb]
                    hd = hp[hm] - held_eval["target"][qb][hm]
                    sq_diff_sum += float((hd ** 2).sum())
                    sq_target_sum += float((held_eval["target"][qb][hm] ** 2).sum())
                    n_valid += int(hm.sum())
                h_nrmse = float(np.sqrt(sq_diff_sum / max(n_valid, 1)) /
                                (np.sqrt(sq_target_sum / max(n_valid, 1)) + 1e-8))
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
                losses.append(float(loss.detach()))
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
            if reg_weight_q_tv > 0:
                postfix["qtv"] = f"{float(q_tv_loss.detach()):.3e}" if q_batch_size > 1 else "0"
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

    # NOTE: unlike the single-shell function, ``model.output_scale`` is left
    # at 1.0 here — it is a single global scalar buffer and cannot represent
    # the per-shell scale factors in ``target_scale_by_q`` (a ~4-decade q
    # falloff, per-shell, not one global number). The model's raw
    # ``forward_at_q`` output is therefore in PER-SHELL-NORMALISED units;
    # multiply by ``target_scale_by_q[qb]`` (exact training q-bin) or
    # ``eval_multiq.fit_scale_trend(...)`` (interpolated/unseen q) to recover
    # physical units.

    result = {
        "model": model,
        "q_index": q_index,
        "q_norm": q_norm_t.detach().cpu().numpy(),
        "target_scale_by_q": target_scale_by_q,
        "log_q_min": log_q_min,
        "log_q_max": log_q_max,
        "losses": losses,
        "time": total_time,
        "iterations": start_step + len(losses),
    }
    if held_eval is not None:
        result["holdout_curve"] = holdout_curve
        result["best_step"] = best_step
        result["best_holdout_nrmse"] = best_holdout_nrmse
        result["stopped_early"] = _stopped_early
        result["holdout_eval_qshells"] = held_eval["eval_index"]
    return result
