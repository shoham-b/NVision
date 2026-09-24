"""OpenEvolve target: Bayesian-SBED's acquisition-point-selection logic.

This is a *standalone copy* of two methods from
``nvision.sim.locs.bayesian.sbed_locator.SequentialBayesianExperimentDesignLocator``
(as of the date this file was created). The evaluator
(``openevolve/sbed_acquisition/evaluator.py``) imports this module and
monkeypatches these three functions onto the real locator class before
running simulations — the production file is never modified by the search,
so nothing here reaches ``nvision/`` until a human reviews and manually
ports back a winning candidate.

Only the EVOLVE-BLOCK region may change. Everything outside it (docstring,
imports) is scaffolding OpenEvolve should leave alone. Function signatures
and names must stay identical — the evaluator binds them by name.

Each function receives ``self`` explicitly (the locator instance) exactly
as the original bound methods did, so the bodies are otherwise unchanged
and free to use ``self.belief``, ``self._acquisition_bounds()``, etc.
Returned frequencies are physical Hz and MUST fall strictly within the
domain bounds used elsewhere in the method (see the existing bounds checks
below) — this codebase fails fast on out-of-domain values instead of
clamping them silently, and the evaluator will score a candidate that
raises on any repeat as a hard failure (score 0), not a partial credit.

See ``openevolve/sbed_acquisition/README.md`` for how to run the search and
interpret results, and ``.claude/skills/locator-evaluation`` for why
``splitting_converged_step`` (not the locator's own stop reason) is the
metric that actually gets optimized.
"""

from __future__ import annotations

import numpy as np

# EVOLVE-BLOCK-START


def _acquire(self) -> float:
    """Select the next measurement point by maximizing EIG over a frequency grid.

    A decaying share of steps instead explores: uniformly over the whole probe window
    (to find dips the posterior has narrowed away from), or near a dip the data already
    show (:meth:`SMCMarginalDistribution.dip_candidates`).
    """
    lo, hi = self._acquisition_bounds()
    if hi <= lo:
        return float(lo)

    # The belief's original (never-narrowed) probe window -- used by the exploration branches
    # so they can still reach a location resampling has already narrowed away from.
    # `_to_experiment_normalized` normalizes against this same full domain, not
    # `_acquisition_bounds()`, so returning a value outside `lo, hi` here is valid.
    orig_lo, orig_hi = self.belief._original_physical_x_bounds

    # The exploration branches are drawn first so the (much more expensive) EIG grid
    # search in _eig_acquire() is skipped entirely on steps where it would be discarded.
    # The uniform exploration probability decays exponentially to focus on EIG as the scan
    # progresses.
    decay = np.exp(-self.inference_step_count / 25.0)
    rand_val = np.random.rand()
    if rand_val < 0.1 * decay:
        return float(np.random.uniform(orig_lo, orig_hi))
    if rand_val < 0.2:
        # Dip-biased sampling: draw within +/-5 MHz of a dip the observations show. This
        # corrects a biased posterior that has drifted away from the true dip location.
        dip_centers = [d.centroid_hz for d in self.belief.dip_candidates if orig_lo <= d.centroid_hz <= orig_hi]
        if dip_centers:
            center = float(np.random.choice(dip_centers))
            return center + float(np.random.uniform(max(-5e6, orig_lo - center), min(5e6, orig_hi - center)))

        # No dip found yet: Thompson sampling of the scanned parameter from the posterior.
        if self._scan_param in self.belief._param_names:
            idx = int(np.random.choice(len(self.belief._weights), p=self.belief._weights))
            p_idx = self.belief._param_names.index(self._scan_param)
            return self.belief._to_physical(self._scan_param, float(self.belief._particles[idx, p_idx]))

    return self._eig_acquire()


def _eig_acquire(self) -> float:
    """Maximize EIG over the belief's slope-targeted candidate grid."""
    # Retrieve candidates directly from the belief (slope-targeted epoch grid)
    candidates = self.belief.get_candidates()

    # Thin candidates to minimum physical step spacing.
    # The epoch grid window is ±3σ_f, so candidate count ≈ 6σ_f / step_hz:
    # many candidates early (large σ_f), few near convergence (σ_f ≈ step_hz).
    candidates = self._thin_candidates_by_step(candidates)

    # Keep the most recently EIG-selected frequency in the candidate set so a
    # second batch there is a legitimate EIG outcome rather than being dropped
    # by minimum-spacing thinning. EIG's diminishing returns decide when
    # re-batching stops paying off (no explicit repeat counter needed).
    lo, hi = self._acquisition_bounds()
    if (
        self._last_eig_physical_x is not None
        and lo <= self._last_eig_physical_x <= hi
        and not np.any(np.isclose(candidates, self._last_eig_physical_x))
    ):
        candidates = np.append(candidates, self._last_eig_physical_x)

    best = self.belief.select_max_information_gain(candidates, 1)
    result = float(best[0]) if len(best) > 0 else float(candidates[len(candidates) // 2])
    self._last_eig_physical_x = result
    return result


# EVOLVE-BLOCK-END
