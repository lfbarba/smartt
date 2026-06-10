"""Linear coarse-to-fine annealing schedules for SAXS NAF.

Two independent, staggered, *linear* ramps (each toggleable):

* **Spatial** — reveal hash levels low→high over the first ``spatial_frac`` of
  training.  Returns per-level weights in ``[0, 1]`` with a soft (fractional)
  edge on the level currently being revealed (FreeNeRF-style).
* **Angular** — reveal SH degrees ℓ = 2, 4, 6, 8 over the first
  ``angular_frac`` of training (slightly slower than spatial by default).
  Returns a per-coefficient visibility mask in ``[0, 1]``; ℓ = 0 is always 1.

Both ramps are pure functions of the normalised training progress
``p = step / max(1, total_steps - 1)``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def _linear_reveal(progress: float, n_units: int, frac: float) -> torch.Tensor:
    """Linearly fade in ``n_units`` units over the first ``frac`` of training.

    Returns a ``(n_units,)`` tensor in ``[0, 1]``: fully revealed units are 1,
    the unit currently being revealed gets the fractional remainder, later units
    are 0.  Unit 0 is always 1 (the coarsest content is never masked).
    """
    w = torch.zeros(n_units)
    w[0] = 1.0
    if n_units == 1:
        return w
    if frac <= 0:
        return torch.ones(n_units)
    # Fraction of the (n_units - 1) revealable units that should be visible.
    revealed = min(progress / frac, 1.0) * (n_units - 1)
    full = int(revealed)
    for i in range(1, min(full + 1, n_units)):
        w[i] = 1.0
    if full + 1 < n_units:
        w[full + 1] = revealed - full
    return w


class Annealer:
    """Produces (level_weights, ell_mask) for a given training step.

    Parameters
    ----------
    n_levels : Number of hash-encoding levels.
    lm_list : List of ``(ℓ, m)`` pairs (from the field), used to broadcast the
        per-degree angular ramp onto per-coefficient visibility.
    total_steps : Total training iterations (for progress normalisation).
    spatial_on, angular_on : Enable each ramp.  When off, the corresponding
        mask is all-ones (everything visible from the start).
    spatial_frac : Fraction of training over which spatial levels are revealed.
    angular_frac : Fraction of training over which SH degrees are revealed
        (default slightly larger than ``spatial_frac`` — staggered).
    """

    def __init__(
        self,
        n_levels: int,
        lm_list: List[Tuple[int, int]],
        total_steps: int,
        spatial_on: bool = True,
        angular_on: bool = True,
        spatial_frac: float = 0.5,
        angular_frac: float = 0.6,
    ):
        self.n_levels = n_levels
        self.total_steps = max(int(total_steps), 1)
        self.spatial_on = spatial_on
        self.angular_on = angular_on
        self.spatial_frac = spatial_frac
        self.angular_frac = angular_frac

        ells = sorted({l for l, _ in lm_list})
        self.ells = ells                       # e.g. [0, 2, 4, 6, 8]
        self.n_degrees = len(ells)
        # Map each coefficient to its degree's index in `ells`.
        deg_index = [ells.index(l) for l, _ in lm_list]
        self._deg_index = torch.tensor(deg_index, dtype=torch.long)

    def _progress(self, step: int) -> float:
        return step / max(self.total_steps - 1, 1)

    def level_weights(self, step: int) -> Optional[torch.Tensor]:
        if not self.spatial_on:
            return None
        return _linear_reveal(self._progress(step), self.n_levels, self.spatial_frac)

    def ell_mask(self, step: int, num_coeffs: int) -> Optional[torch.Tensor]:
        if not self.angular_on:
            return None
        deg_w = _linear_reveal(self._progress(step), self.n_degrees, self.angular_frac)
        return deg_w[self._deg_index]          # (num_coeffs,)

    def masks(
        self, step: int, num_coeffs: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.level_weights(step), self.ell_mask(step, num_coeffs)
