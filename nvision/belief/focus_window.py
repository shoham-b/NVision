"""Value objects and policy for narrowing the drive-frequency axis mid-run.

Two kinds of narrowing share only the clamping arithmetic (:func:`clamp_to_domain`):

* The Bayesian locator's dip-based focus is a :class:`FocusRegion` -- one padded section per
  statistically significant dip (``find_dips``), overlapping sections merged -- decided by
  :func:`next_focus_region`. Owned by the locator, never the belief.
* The sweep locators' geometric dip-shape narrowing (``StagedSobolSweepLocator`` /
  ``nvision.sim.locs.refocus``) uses a single-interval :class:`FocusWindow`, built with
  :meth:`FocusWindow.from_candidate` (clamp + reject a collapsed window).

How a candidate is *computed* stays separate, deliberately: the Bayesian locator uses the
extents of significant dips in its observations; the sweep locators detect dips geometrically (see
``nvision.sim.locs.refocus`` for why those two detectors are "not interchangeable").

Both value objects carry the immutable ``full_lo``/``full_hi`` axis they can never escape, and
every mutation returns a new instance.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nvision.belief.dip_detection import DipCandidate

_Numeric = float | np.ndarray

# --- drive-frequency focus policy (see next_focus_region) -------------------------------------------
# Narrowing waits this many steps so multi-modal hyperfine ambiguity resolves before the focus
# settles on a (possibly wrong) place.
NVISION_MIN_STEPS_BEFORE_NARROWING: int = int(os.getenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "8"))
# Each significant dip's extent is padded by k * hwhm on each side to form its focus section; k is
# this cover factor. Dip extents only span the points measured >= n_sigma below baseline, i.e. the
# line core, so the pad restores the wings (3 HWHM of a Lorentzian carries ~90% of its depth).
NVISION_SMC_FOCUSING_COVER_FACTOR: float = float(os.getenv("NVISION_SMC_FOCUSING_COVER_FACTOR", "3.0"))
# A narrowing is only applied when it shrinks the focus's total covered width by at least this fraction.
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

    def to_measure_drive_freq_unit(self, drive_freq_phys: _Numeric) -> _Numeric:
        """Map a physical drive-frequency position (scalar or array) to [0, 1] for ``CoreExperiment.measure()``.

        Always uses ``full_lo`` / ``full_hi`` -- never the (possibly narrowed)
        ``lo`` / ``hi``. This is the **only** correct path into
        ``CoreExperiment.measure()``.
        """
        return (drive_freq_phys - self.full_lo) / (self.full_hi - self.full_lo)


@dataclass(frozen=True)
class FocusRegion:
    """A union of disjoint physical sections of the drive-frequency axis: where candidates may be scanned.

    The Bayesian locator's focus. Each detected dip contributes its own padded section, so the empty
    axis between two separated dips is excluded instead of being swallowed by one bounding interval.
    Sections whose padded extents overlap or touch are merged into one.

    Parameters
    ----------
    sections:
        Sorted, pairwise-disjoint ``(lo_phys, hi_phys)`` intervals in Hz, each with ``lo < hi`` and
        inside ``[full_lo, full_hi]``. shape: (n_sections, 2). Build from raw intervals with
        :meth:`from_intervals`, which clamps and merges; the constructor only validates.
    full_lo, full_hi:
        Immutable full drive-frequency axis (Hz) the sections can never leave.
    """

    sections: tuple[tuple[float, float], ...]
    full_lo: float
    full_hi: float

    def __post_init__(self) -> None:
        if self.full_hi <= self.full_lo:
            raise ValueError(
                f"FocusRegion requires full_lo < full_hi; got full_lo={self.full_lo}, full_hi={self.full_hi}"
            )
        if not self.sections:
            raise ValueError("FocusRegion requires at least one section.")
        prev_hi = -np.inf
        for lo, hi in self.sections:
            if not (lo < hi):
                raise ValueError(f"FocusRegion section requires lo < hi; got ({lo}, {hi})")
            if lo < self.full_lo or hi > self.full_hi:
                raise ValueError(
                    f"FocusRegion section ({lo}, {hi}) lies outside the full axis [{self.full_lo}, {self.full_hi}]"
                )
            if lo <= prev_hi:
                raise ValueError(f"FocusRegion sections must be sorted and disjoint; got {self.sections}")
            prev_hi = hi

    @classmethod
    def full(cls, full_lo: float, full_hi: float) -> FocusRegion:
        """The whole drive-frequency axis as a single section."""
        return cls(sections=((full_lo, full_hi),), full_lo=full_lo, full_hi=full_hi)

    @classmethod
    def from_intervals(cls, intervals: Sequence[tuple[float, float]], *, full_lo: float, full_hi: float) -> FocusRegion:
        """Clamp each raw interval to the full axis, then merge overlapping or touching ones.

        Fails fast (``ValueError``) when an interval lies entirely outside the axis: every interval
        here comes from a measured dip, and a measurement off the axis is an upstream bug.
        """
        if not intervals:
            raise ValueError("FocusRegion.from_intervals needs at least one interval.")
        clamped: list[tuple[float, float]] = []
        for raw_lo, raw_hi in intervals:
            lo, hi = clamp_to_domain(raw_lo, raw_hi, full_lo, full_hi)
            if hi <= lo:
                raise ValueError(
                    f"FocusRegion.from_intervals: interval [{raw_lo}, {raw_hi}] Hz lies outside "
                    f"the full drive-frequency axis [{full_lo}, {full_hi}] Hz."
                )
            clamped.append((lo, hi))
        clamped.sort()
        merged: list[list[float]] = [list(clamped[0])]
        for lo, hi in clamped[1:]:
            if lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return cls(sections=tuple((lo, hi) for lo, hi in merged), full_lo=full_lo, full_hi=full_hi)

    @property
    def lo(self) -> float:
        """Left edge of the leftmost section (Hz)."""
        return self.sections[0][0]

    @property
    def hi(self) -> float:
        """Right edge of the rightmost section (Hz)."""
        return self.sections[-1][1]

    @property
    def width(self) -> float:
        """Total covered width (Hz): the sum of the section widths, gaps excluded."""
        return float(sum(hi - lo for lo, hi in self.sections))

    def is_full_domain(self, *, eps: float = 1e-9) -> bool:
        """True when the region still covers the whole axis."""
        return self.width >= (self.full_hi - self.full_lo) * (1.0 - eps)

    def contains(self, drive_freq_phys: np.ndarray) -> np.ndarray:
        """Boolean mask: which positions (Hz) lie inside some section (edges inclusive).

        shape: same as *drive_freq_phys*.
        """
        x = np.asarray(drive_freq_phys, dtype=np.float64)
        mask = np.zeros(x.shape, dtype=bool)
        for lo, hi in self.sections:
            mask |= (x >= lo) & (x <= hi)
        return mask

    def covers(self, other: FocusRegion) -> bool:
        """True when every section of *other* lies inside a single section of this region."""
        return all(any(lo <= o_lo and o_hi <= hi for lo, hi in self.sections) for o_lo, o_hi in other.sections)

    def union(self, other: FocusRegion) -> FocusRegion:
        """The region covering both this one and *other* (same full axis)."""
        if (other.full_lo, other.full_hi) != (self.full_lo, self.full_hi):
            raise ValueError(
                f"FocusRegion.union: full axes differ: [{self.full_lo}, {self.full_hi}] vs "
                f"[{other.full_lo}, {other.full_hi}]"
            )
        return FocusRegion.from_intervals([*self.sections, *other.sections], full_lo=self.full_lo, full_hi=self.full_hi)


def next_focus_region(
    focus: FocusRegion,
    *,
    dips: Sequence[DipCandidate],
    pad_phys: float,
    step: int,
    last_expansion_step: int,
) -> tuple[FocusRegion, bool]:
    """Decide the drive-frequency focus for the next epoch from the dips the observations show.

    The target region has one section per significant dip: its ``[f_min, f_max]`` extent (see
    :func:`nvision.belief.dip_detection.find_dips`: measurements ``NVISION_DIP_N_SIGMA`` noise sigmas
    below the baseline that pass the binomial test), padded by ``pad_phys`` on each side, clamped to the
    full axis, with overlapping sections merged. It depends on the measurements only, never on the
    particle belief.

    * A target section reaches outside the current focus (a new dip, or a known one growing): the focus
      *expands* to the union of the current focus and the target, immediately.
    * Otherwise (the focus covers the target), once ``step >= NVISION_MIN_STEPS_BEFORE_NARROWING`` and at
      least that many steps have passed since the last expansion, the focus *narrows* to the target,
      provided that shrinks its total covered width by at least ``_MIN_NARROWING_FRACTION``.
    * No dips: the focus is left unchanged (a full axis stays full) -- there is no fact to act on.

    The focus only limits *which candidate x positions may be scanned*; it never changes the belief's
    parameter bounds, particles or the model's drive-frequency axis.

    Args:
        focus: Current focus region.
        dips: Significant dips; each carries ``f_min``/``f_max`` in Hz.
        pad_phys: Margin added on each side of every dip extent, in Hz (> 0).
        step: Number of observations so far.
        last_expansion_step: ``step`` of the most recent expansion, -1 if none.

    Returns:
        ``(new_focus, expanded)``; ``new_focus is focus`` when nothing changed.
    """
    if not (pad_phys > 0.0) or not np.isfinite(pad_phys):
        raise ValueError(f"next_focus_region: pad_phys must be positive and finite, got {pad_phys!r}")
    if not dips:
        return focus, False

    target = FocusRegion.from_intervals(
        [(d.f_min - pad_phys, d.f_max + pad_phys) for d in dips], full_lo=focus.full_lo, full_hi=focus.full_hi
    )

    if not focus.covers(target):
        return focus.union(target), True

    # Narrowing delay: let early dips accumulate (and a recent expansion be explored) before committing.
    if step < NVISION_MIN_STEPS_BEFORE_NARROWING:
        return focus, False
    if last_expansion_step >= 0 and (step - last_expansion_step) < NVISION_MIN_STEPS_BEFORE_NARROWING:
        return focus, False

    # The focus covers the target here, so the target's width can only be <= the focus width.
    if target.width > (1.0 - _MIN_NARROWING_FRACTION) * focus.width:
        return focus, False
    return target, False
