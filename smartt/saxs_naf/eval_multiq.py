"""Query helpers for a trained multi-q :class:`SaxsNafField`.

Public API
----------
q_to_norm(q, log_q_min, log_q_max)
    Map a physical q value to the model's normalised q-coordinate.

sample_qshells(model, q_values, log_q_min, log_q_max, ...)
    Evaluate the trained field at one or more (possibly untrained,
    interpolated) q values -> ``(Q, X, Y, Z, C)`` coefficients, in
    PER-SHELL-NORMALISED units (see ``fit_scale_trend`` to recover physical
    units).

fit_scale_trend(target_scale_by_q)
    Piecewise-log-log interpolator recovering the (~4-decade, per-shell) SAXS
    intensity falloff at an arbitrary q, including q values never trained on.

sample_qshells_physical(model, q_values, log_q_min, log_q_max, target_scale_by_q, ...)
    ``sample_qshells`` rescaled back to physical units via ``fit_scale_trend``.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


def q_to_norm(q: float, log_q_min: float, log_q_max: float) -> float:
    """Physical q -> normalised ``[0, 1]`` coordinate (log-spaced, matches training)."""
    span = max(log_q_max - log_q_min, 1e-12)
    return float((np.log(q) - log_q_min) / span)


@torch.no_grad()
def sample_qshells(
    model,
    q_values: Sequence[float],
    log_q_min: float,
    log_q_max: float,
    device=None,
) -> torch.Tensor:
    """Evaluate a trained multi-q field at arbitrary physical q value(s).

    Parameters
    ----------
    model : SaxsNafField
        Trained field with ``n_qshells > 1`` (has a ``q_encoding``).
    q_values : sequence of float
        Physical q values to query — need not be q-bins the model was
        trained on; values between training shells exercise the q-encoder's
        interpolation (the core claim of the joint 3D-RSM model: it should
        predict a *sensible*, smoothly-varying shell even at a q it never
        saw a projection for).
    log_q_min, log_q_max : float
        From the training run's result dict (``r['log_q_min']``,
        ``r['log_q_max']``) — the SAME normalisation used during training.

    Returns
    -------
    ``(len(q_values), X, Y, Z, C)`` CPU tensor.
    """
    if device is None:
        device = next(model.parameters()).device
    q_norm = torch.tensor(
        [q_to_norm(q, log_q_min, log_q_max) for q in q_values],
        dtype=torch.float32, device=device,
    )
    return model.forward_at_q(q_norm).detach().cpu()


def fit_scale_trend(target_scale_by_q: Dict[int, float], q_values: Dict[int, float]):
    """Piecewise-log-log interpolator for the per-shell scale trend.

    SAXS intensity falls off ~monotonically (often close to a power law) over
    q; ``target_scale_by_q`` (from a multi-q training run's result dict) only
    has one value per *trained* q-bin. This fits nothing fancy — just
    ``np.interp`` in ``(log q, log scale)`` space, which is exact at every
    trained q-bin and a well-behaved, conservative interpolant in between
    (and a flat extrapolation outside the trained range, ``np.interp``'s
    default). That is sufficient for the interpolation test this is meant
    for: recovering physical units at a q the model was never trained on but
    that lies between two it was.

    Returns
    -------
    callable : ``q -> scale`` (physical q in, multiplicative scale out).
    """
    qbins = sorted(target_scale_by_q.keys())
    log_q = np.array([np.log(q_values[qb]) for qb in qbins])
    log_scale = np.array([np.log(target_scale_by_q[qb]) for qb in qbins])
    order = np.argsort(log_q)
    log_q, log_scale = log_q[order], log_scale[order]

    def trend(q):
        return float(np.exp(np.interp(np.log(q), log_q, log_scale)))

    return trend


@torch.no_grad()
def sample_qshells_physical(
    model,
    q_values_query: Sequence[float],
    log_q_min: float,
    log_q_max: float,
    target_scale_by_q: Dict[int, float],
    q_values_trained: Dict[int, float],
    device=None,
) -> torch.Tensor:
    """:func:`sample_qshells`, rescaled to physical units via :func:`fit_scale_trend`.

    Returns ``(len(q_values_query), X, Y, Z, C)`` CPU tensor in the same
    physical units as the original (un-normalised) projections.
    """
    coeffs = sample_qshells(model, q_values_query, log_q_min, log_q_max, device=device)
    trend = fit_scale_trend(target_scale_by_q, q_values_trained)
    scales = torch.tensor([trend(q) for q in q_values_query], dtype=coeffs.dtype)
    return coeffs * scales.view(-1, 1, 1, 1, 1)
