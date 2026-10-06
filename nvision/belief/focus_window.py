"""The one shared procedure for narrowing a drive-frequency window.

Every part of the pipeline that shrinks the drive-frequency axis mid-run -- the
Bayesian locator's particle-percentile focus (:func:`next_focus_window`, owned by
the locator, never the belief)
and the sweep locators' geometric dip-shape narrowing (``StagedSobolSweepLocator`` /
``nvision.sim.locs.refocus``) -- ends up needing the same three guard rails:

1. Clamp the candidate window into the immutable full domain.
2. Optionally enforce a minimum width (a floor below which the window must
   not collapse).
3. Reject candidates that don't actually shrink the window by a meaningful
   amount, so a run doesn't spend cycles re-applying no-op narrowings.

What must stay separate, deliberately, is *how a candidate window is
computed*: the Bayesian locator has a particle ensemble and estimates a candidate
from percentiles of per-particle active ranges; the sweep locators only have
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
from dataclasses import dataclass

import numpy as np

_Numeric = float | np.ndarray

# --- drive-frequency focus policy (see next_focus_window) -------------------------------------------
# Narrowing waits this many steps so multi-modal hyperfine ambiguity resolves before the focus
# settles on a (possibly wrong) place.
NVISION_MIN_STEPS_BEFORE_NARROWING: int = int(os.getenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "8"))
# Each particle's active range is [center_freq - span, center_freq + span] with
# span = zeeman_split + split + k * hwhm; k is this cover factor.
NVISION_SMC_FOCUSING_COVER_FACTOR: float = float(os.getenv("NVISION_SMC_FOCUSING_COVER_FACTOR", "3.0"))
# The 5th/95th percentiles skip stray low-weight tail particles while barely eating into the true
# dense clusters (which span ~100s of kHz, so losing 5% of their mass barely moves the boundary).
_FOCUSING_TAIL_PERCENTILE: float = 5.0
# A narrowing is only applied when it shrinks the focus by at least this fraction.
_MIN_NARROWING_FRACTION: float = 0.05
# Particles within this fraction of the focus width of its edge (or outside it) signal that the
# true center_freq lies beyond the focus...
_EDGE_BAND_FRACTION: float = 0.05
# ...once more than this fraction of them sit in that band.
_EDGE_PILING_FRACTION: float = 0.15
# The focus never expands/narrows around a dip narrower than this (a floor on omega).
_MIN_OMEGA_PHYS: float = 1.0e5


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

    def to_measure_x(self, x_phys: _Numeric) -> _Numeric:
        """Map a physical drive-frequency position (scalar or array) to [0, 1] for ``CoreExperiment.measure()``.

        Always uses ``full_lo`` / ``full_hi`` -- never the (possibly narrowed)
        ``lo`` / ``hi``. This is the **only** correct path into
        ``CoreExperiment.measure()``.
        """
        return (x_phys - self.full_lo) / (self.full_hi - self.full_lo)


def next_focus_window(
    focus: FocusWindow,
    *,
    center_freq_particles_phys: np.ndarray,
    active_half_span_particles_phys: np.ndarray,
    omega_phys: float,
    zeeman_split_phys: float,
    step: int,
    last_expansion_step: int,
) -> tuple[FocusWindow, bool]:
    """Decide the drive-frequency focus for the next epoch from the posterior over ``center_freq``.

    The focus only limits *which candidate x positions may be scanned*; it never changes the belief's
    parameter bounds, particles or the model's drive-frequency axis. Call it right after a resample, when the
    particle weights are uniform (the percentiles below are unweighted).

    Args:
        focus: Current focus (sub-interval of the drive-frequency axis).
        center_freq_particles_phys: Per-particle ``center_freq`` in Hz. shape: (n_particles,)
        active_half_span_particles_phys: Per-particle half-span of the dips, ``zeeman_split + split +
            NVISION_SMC_FOCUSING_COVER_FACTOR * hwhm``, in Hz. shape: (n_particles,)
        omega_phys: Posterior-mean effective HWHM in Hz (sizes a boundary-escape expansion).
        zeeman_split_phys: Posterior-mean Zeeman split in Hz (sizes a boundary-escape expansion).
        step: Number of observations so far.
        last_expansion_step: ``step`` of the most recent expansion, -1 if none.

    Returns:
        ``(new_focus, expanded)``; ``new_focus is focus`` when nothing changed.
    """
    if center_freq_particles_phys.shape != active_half_span_particles_phys.shape:
        raise ValueError(
            "next_focus_window: particle arrays must match, got "
            f"{center_freq_particles_phys.shape} vs {active_half_span_particles_phys.shape}"
        )

    # Boundary escape: particles piling at/beyond a focus edge mean the true center_freq lies outside
    # the focus, so expand it that way (never past the full drive-frequency axis).
    width = focus.hi - focus.lo
    u = (center_freq_particles_phys - focus.lo) / width
    left_piling = float(np.mean(u < _EDGE_BAND_FRACTION)) > _EDGE_PILING_FRACTION
    right_piling = float(np.mean(u > 1.0 - _EDGE_BAND_FRACTION)) > _EDGE_PILING_FRACTION
    expansion = max(width, 10.0 * max(omega_phys, _MIN_OMEGA_PHYS) + 2.0 * zeeman_split_phys)
    if left_piling and focus.lo > focus.full_lo:
        return FocusWindow(max(focus.lo - expansion, focus.full_lo), focus.hi, focus.full_lo, focus.full_hi), True
    if right_piling and focus.hi < focus.full_hi:
        return FocusWindow(focus.lo, min(focus.hi + expansion, focus.full_hi), focus.full_lo, focus.full_hi), True

    # Narrowing delay: resolve multi-modal ambiguity first, and give a recent expansion time to be explored.
    if step < NVISION_MIN_STEPS_BEFORE_NARROWING:
        return focus, False
    if last_expansion_step >= 0 and (step - last_expansion_step) < NVISION_MIN_STEPS_BEFORE_NARROWING:
        return focus, False

    # Active-range union. Each particle believes its dips span center_freq +/- half_span; the half-span
    # is symmetric, so it is pooled across all particles into one upper quantile. The center term keeps
    # its own percentiles (not the mean) so a wide or not-yet-unimodal posterior is still respected.
    span_q = float(np.percentile(active_half_span_particles_phys, 100.0 - _FOCUSING_TAIL_PERCENTILE))
    f_lo = float(np.percentile(center_freq_particles_phys, _FOCUSING_TAIL_PERCENTILE))
    f_hi = float(np.percentile(center_freq_particles_phys, 100.0 - _FOCUSING_TAIL_PERCENTILE))
    proposed = focus.propose_narrowing(
        f_lo - span_q,
        f_hi + span_q,
        min_width=2.0 * span_q,
        min_narrowing_fraction=_MIN_NARROWING_FRACTION,
    )
    return (focus, False) if proposed is None else (proposed, False)
