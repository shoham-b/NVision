"""The one shared procedure for narrowing a drive-frequency window.

Every part of the pipeline that shrinks the drive-frequency axis mid-run -- the
Bayesian locator's dip-based focus (:func:`next_focus_window`, owned by
the locator, never the belief)
and the sweep locators' geometric dip-shape narrowing (``StagedSobolSweepLocator`` /
``nvision.sim.locs.refocus``) -- ends up needing the same three guard rails:

1. Clamp the candidate window into the immutable full domain.
2. Optionally enforce a minimum width (a floor below which the window must
   not collapse).
3. Reject candidates that don't actually shrink the window by a meaningful
   amount, so a run doesn't spend cycles re-applying no-op narrowings.

What must stay separate, deliberately, is *how a candidate window is
computed*: the Bayesian locator takes the extents of the statistically significant dips
(``find_dips``) in its observations; the sweep locators only have
raw (x, y) sweep observations and detect dips geometrically (see
``nvision.sim.locs.refocus`` for why those two detectors are "not
interchangeable"). This module is exclusively the shared *application* of a
narrowing decision once a candidate ``(lo, hi)`` has been computed -- not the
detector.

``FocusWindow`` is the value object every narrowing decision flows through:
physical ``[lo, hi]`` sub-interval of the drive-frequency axis currently being
probed, plus the immutable ``full_lo``/``full_hi`` domain it can never escape.
Owned by whichever locator or belief is doing the narrowing. Every mutation
returns a new instance -- the object itself never changes in place.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.belief.dip_detection import DipCandidate

_Numeric = float | np.ndarray

# --- drive-frequency focus policy (see next_focus_window) -------------------------------------------
# Narrowing waits this many steps so multi-modal hyperfine ambiguity resolves before the focus
# settles on a (possibly wrong) place.
NVISION_MIN_STEPS_BEFORE_NARROWING: int = int(os.getenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "8"))
# The focus is the significant dips' bounding interval padded by k * hwhm on each side; k is this
# cover factor. Dip extents only span the points measured >= n_sigma below baseline, i.e. the
# line core, so the pad restores the wings (3 HWHM of a Lorentzian carries ~90% of its depth).
NVISION_SMC_FOCUSING_COVER_FACTOR: float = float(os.getenv("NVISION_SMC_FOCUSING_COVER_FACTOR", "3.0"))
# A narrowing is only applied when it shrinks the focus by at least this fraction.
_MIN_NARROWING_FRACTION: float = 0.05


def clamp_to_domain(lo: float, hi: float, domain_lo: float, domain_hi: float) -> tuple[float, float]:
    """Clip ``(lo, hi)`` into ``[domain_lo, domain_hi]``, tolerating swapped input order."""
    lo, hi = min(lo, hi), max(lo, hi)
    return max(lo, domain_lo), min(hi, domain_hi)


@dataclass(frozen=True)
class FocusWindow:
    """Physical ``[lo, hi]`` sub-interval of the drive-frequency axis.

    Owned by the locator or belief doing the narrowing. Can shrink (or, via
    :meth:`from_candidate`, grow back) during a run -- every mutation returns
    a *new* ``FocusWindow`` without touching the original. ``full_lo`` /
    ``full_hi`` are set once at construction and never change; they are the
    only values that may be passed to ``CoreExperiment.measure()`` for
    normalisation, and the only ceiling any narrowing may clamp against.

    Parameters
    ----------
    lo, hi:
        Current (possibly narrowed) physical bounds of the drive-frequency window.
    full_lo, full_hi:
        Original full-domain physical bounds. Immutable. Used exclusively by
        :meth:`to_measure_drive_freq_unit` so that experiment normalisation is always
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

    def to_measure_drive_freq_unit(self, drive_freq_phys: _Numeric) -> _Numeric:
        """Map a physical drive-frequency position (scalar or array) to [0, 1] for ``CoreExperiment.measure()``.

        Always uses ``full_lo`` / ``full_hi`` -- never the (possibly narrowed)
        ``lo`` / ``hi``. This is the **only** correct path into
        ``CoreExperiment.measure()``.
        """
        return (drive_freq_phys - self.full_lo) / (self.full_hi - self.full_lo)


def next_focus_window(
    focus: FocusWindow,
    *,
    dips: Sequence[DipCandidate],
    pad_phys: float,
    step: int,
    last_expansion_step: int,
) -> tuple[FocusWindow, bool]:
    """Decide the drive-frequency focus for the next epoch from the dips the observations show.

    The target window is the bounding interval of every significant dip's ``[f_min, f_max]`` extent
    (see :func:`nvision.belief.dip_detection.find_dips`: measurements ``NVISION_DIP_N_SIGMA`` noise
    sigmas below the baseline that pass the binomial test), padded by ``pad_phys`` on each side and
    clamped to the full axis. It depends on the measurements only, never on the particle belief.

    * Target reaches outside the current focus (a new dip, or a known one growing): the focus
      *expands* to the union of the current focus and the target, immediately.
    * Otherwise, once ``step >= NVISION_MIN_STEPS_BEFORE_NARROWING`` and at least that many steps have
      passed since the last expansion, the focus *narrows* to the target, provided that shrinks it by
      at least ``_MIN_NARROWING_FRACTION``.
    * No dips: the focus is left unchanged (a full axis stays full) -- there is no fact to act on.

    The focus only limits *which candidate x positions may be scanned*; it never changes the belief's
    parameter bounds, particles or the model's drive-frequency axis.

    Args:
        focus: Current focus (sub-interval of the drive-frequency axis).
        dips: Significant dips; each carries ``f_min``/``f_max`` in Hz.
        pad_phys: Margin added on each side of the dips' bounding interval, in Hz (> 0).
        step: Number of observations so far.
        last_expansion_step: ``step`` of the most recent expansion, -1 if none.

    Returns:
        ``(new_focus, expanded)``; ``new_focus is focus`` when nothing changed.
    """
    if not (pad_phys > 0.0) or not np.isfinite(pad_phys):
        raise ValueError(f"next_focus_window: pad_phys must be positive and finite, got {pad_phys!r}")
    if not dips:
        return focus, False

    dips_lo, dips_hi = min(d.f_min for d in dips), max(d.f_max for d in dips)
    target_lo, target_hi = clamp_to_domain(dips_lo - pad_phys, dips_hi + pad_phys, focus.full_lo, focus.full_hi)
    if target_hi <= target_lo:
        raise ValueError(
            f"next_focus_window: dips [{dips_lo}, {dips_hi}] Hz lie outside "
            f"the full drive-frequency axis [{focus.full_lo}, {focus.full_hi}] Hz."
        )

    if target_lo < focus.lo or target_hi > focus.hi:
        new_lo, new_hi = min(focus.lo, target_lo), max(focus.hi, target_hi)
        return FocusWindow(new_lo, new_hi, focus.full_lo, focus.full_hi), True

    # Narrowing delay: let early dips accumulate (and a recent expansion be explored) before committing.
    if step < NVISION_MIN_STEPS_BEFORE_NARROWING:
        return focus, False
    if last_expansion_step >= 0 and (step - last_expansion_step) < NVISION_MIN_STEPS_BEFORE_NARROWING:
        return focus, False

    proposed = focus.propose_narrowing(target_lo, target_hi, min_narrowing_fraction=_MIN_NARROWING_FRACTION)
    return (focus, False) if proposed is None else (proposed, False)
