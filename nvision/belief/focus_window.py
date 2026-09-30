"""The one shared procedure for narrowing a probe window.

Every part of the pipeline that shrinks the frequency axis mid-run -- the
Bayesian SMC belief's particle-percentile narrowing (``UnitCubeSMCMarginalDistribution._resample``)
and the sweep locators' geometric dip-shape narrowing (``StagedSobolSweepLocator`` /
``nvision.sim.locs.refocus``) -- ends up needing the same three guard rails:

1. Clamp the candidate window into the immutable full domain.
2. Optionally enforce a minimum width (a floor below which the window must
   not collapse).
3. Reject candidates that don't actually shrink the window by a meaningful
   amount, so a run doesn't spend cycles re-applying no-op narrowings.

What must stay separate, deliberately, is *how a candidate window is
computed*: the SMC belief has a particle ensemble and estimates a candidate
from percentiles of per-particle active ranges; the sweep locators only have
raw (x, y) sweep observations and detect dips geometrically (see
``nvision.sim.locs.refocus`` for why those two detectors are "not
interchangeable"). This module is exclusively the shared *application* of a
narrowing decision once a candidate ``(lo, hi)`` has been computed -- not the
detector.

``FocusWindow`` is the value object every narrowing decision flows through:
physical ``[lo, hi]`` sub-interval of the frequency axis currently being
probed, plus the immutable ``full_lo``/``full_hi`` domain it can never escape.
Owned by whichever locator or belief is doing the narrowing. Every mutation
returns a new instance -- the object itself never changes in place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_Numeric = float | np.ndarray


def clamp_to_domain(lo: float, hi: float, domain_lo: float, domain_hi: float) -> tuple[float, float]:
    """Clip ``(lo, hi)`` into ``[domain_lo, domain_hi]``, tolerating swapped input order."""
    lo, hi = min(lo, hi), max(lo, hi)
    return max(lo, domain_lo), min(hi, domain_hi)


@dataclass(frozen=True)
class FocusWindow:
    """Physical ``[lo, hi]`` sub-interval of the frequency axis.

    Owned by the locator or belief doing the narrowing. Can shrink (or, via
    :meth:`from_candidate`, grow back) during a run -- every mutation returns
    a *new* ``FocusWindow`` without touching the original. ``full_lo`` /
    ``full_hi`` are set once at construction and never change; they are the
    only values that may be passed to ``CoreExperiment.measure()`` for
    normalisation, and the only ceiling any narrowing may clamp against.

    Parameters
    ----------
    lo, hi:
        Current (possibly narrowed) physical bounds of the probe window.
    full_lo, full_hi:
        Original full-domain physical bounds. Immutable. Used exclusively by
        :meth:`to_measure_x` so that experiment normalisation is always
        relative to the full domain.
    """

    lo: float
    hi: float
    full_lo: float
    full_hi: float

    def __post_init__(self) -> None:
        if self.hi <= self.lo:
            raise ValueError(f"FocusWindow requires lo < hi; got lo={self.lo}, hi={self.hi}")
        if self.full_hi <= self.full_lo:
            raise ValueError(
                f"FocusWindow requires full_lo < full_hi; got full_lo={self.full_lo}, full_hi={self.full_hi}"
            )

    @classmethod
    def from_candidate(
        cls, candidate_lo: float, candidate_hi: float, *, full_lo: float, full_hi: float
    ) -> FocusWindow | None:
        """Build a ``FocusWindow`` from a raw candidate, clamped to ``[full_lo, full_hi]``.

        Returns ``None`` if the clamped candidate collapses (``hi <= lo``)
        instead of raising, so callers that compute a candidate from noisy
        data (e.g. a sweep locator re-inferring its window every few
        observations) can simply keep their previous window on failure.
        """
        lo, hi = clamp_to_domain(candidate_lo, candidate_hi, full_lo, full_hi)
        if hi <= lo:
            return None
        return cls(lo=lo, hi=hi, full_lo=full_lo, full_hi=full_hi)

    def is_full_domain(self, *, eps: float = 1e-9) -> bool:
        """True when this window has not actually narrowed from the full domain."""
        full_width = self.full_hi - self.full_lo
        return (self.hi - self.lo) >= full_width * (1.0 - eps)

    def propose_narrowing(
        self,
        candidate_lo: float,
        candidate_hi: float,
        *,
        min_width: float | None = None,
        min_narrowing_fraction: float = 0.0,
    ) -> FocusWindow | None:
        """Apply the shared narrowing decision to a detector-computed candidate.

        In order:

        1. If ``min_width`` is given and the candidate is narrower than it,
           re-center the candidate to exactly ``min_width`` (never shrink
           past the caller's known floor, e.g. the belief's own active-range
           estimate).
        2. Clamp the (possibly widened) candidate into ``[full_lo, full_hi]``.
        3. Reject (return ``None``) if the clamped candidate has collapsed.
        4. Reject if the candidate doesn't shrink the *current* window by at
           least ``min_narrowing_fraction`` -- avoids churning on noise-level
           narrowings that aren't worth the resample cost.

        Returns the new, narrower ``FocusWindow``, or ``None`` if the
        candidate was rejected at any step (callers keep their current
        window in that case).
        """
        lo, hi = candidate_lo, candidate_hi
        if min_width is not None and (hi - lo) < min_width:
            center = 0.5 * (hi + lo)
            lo = center - min_width / 2.0
            hi = center + min_width / 2.0

        lo, hi = clamp_to_domain(lo, hi, self.full_lo, self.full_hi)
        if hi <= lo:
            return None

        cur_width = self.hi - self.lo
        if min_narrowing_fraction > 0.0 and cur_width > 0.0:
            shrink_frac = (cur_width - (hi - lo)) / cur_width
            if shrink_frac < min_narrowing_fraction:
                return None

        return FocusWindow(lo=lo, hi=hi, full_lo=self.full_lo, full_hi=self.full_hi)

    def to_measure_x(self, phys_freq: _Numeric) -> _Numeric:
        """Map a physical frequency (scalar or array) to [0, 1] for ``CoreExperiment.measure()``.

        Always uses ``full_lo`` / ``full_hi`` -- never the (possibly narrowed)
        ``lo`` / ``hi``. This is the **only** correct path into
        ``CoreExperiment.measure()``.
        """
        return (phys_freq - self.full_lo) / (self.full_hi - self.full_lo)
