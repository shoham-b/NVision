"""Coordinate-system value objects for the NVision inference pipeline.

Two orthogonal transforms operate on measurement data:

``RescaleMap``
    Fixed [0, 1] <-> physical mapping for one parameter.  Created once at
    belief construction from ``physical_param_bounds`` and never mutated.
    Owned by the belief.  All parameters carry one.

``FocusWindow``
    Physical [lo, hi] sub-interval of the frequency axis that the locator
    currently probes.  Can narrow during a run (always returns a new
    instance).  Owned by the locator.  Carries immutable ``full_lo``/
    ``full_hi`` so ``CoreExperiment.measure()`` normalisation is always
    relative to the original full domain.

These two are **orthogonal**.  Conflating them — e.g. using
``physical_param_bounds["frequency"]`` for both rescaling and acquisition
bounds — is the root coordinate-system defect this module fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_Numeric = float | np.ndarray


@dataclass(frozen=True)
class RescaleMap:
    """Fixed [0, 1] <-> physical mapping for one parameter.

    Created at belief construction.  Never mutated for the lifetime of a run.

    Parameters
    ----------
    lo, hi:
        Physical lower / upper bound of this parameter.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.hi <= self.lo:
            raise ValueError(f"RescaleMap requires lo < hi; got lo={self.lo}, hi={self.hi}")

    def to_phys(self, u: _Numeric) -> _Numeric:
        """Map a unit-cube value (scalar or array) in [0, 1] to physical units."""
        return self.lo + u * (self.hi - self.lo)

    def to_unit(self, phys: _Numeric) -> _Numeric:
        """Map a physical value (scalar or array) to [0, 1]."""
        return (phys - self.lo) / (self.hi - self.lo)

    @property
    def width(self) -> float:
        """Physical width of the parameter range."""
        return self.hi - self.lo


@dataclass(frozen=True)
class FocusWindow:
    """Physical [lo, hi] sub-interval of the frequency axis.

    Owned by the locator.  Can narrow during a run — each call to
    :meth:`narrowed` returns a *new* ``FocusWindow`` without touching the
    original.  ``full_lo`` / ``full_hi`` are set once at construction and
    never change; they are the only values that may be passed to
    ``CoreExperiment.measure()`` for normalisation.

    Parameters
    ----------
    lo, hi:
        Current (possibly narrowed) physical bounds of the probe window.
    full_lo, full_hi:
        Original full-domain physical bounds.  Immutable.  Used exclusively
        by :meth:`to_measure_x` so that experiment normalisation is always
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

    def narrowed(self, new_lo: float, new_hi: float) -> FocusWindow:
        """Return a new ``FocusWindow`` clipped to ``[new_lo, new_hi]``.

        ``full_lo`` / ``full_hi`` are preserved unchanged — they must never
        be narrowed.
        """
        return FocusWindow(
            lo=max(new_lo, self.lo),
            hi=min(new_hi, self.hi),
            full_lo=self.full_lo,
            full_hi=self.full_hi,
        )

    def to_measure_x(self, phys_freq: _Numeric) -> _Numeric:
        """Map a physical frequency (scalar or array) to [0, 1] for ``CoreExperiment.measure()``.

        Always uses ``full_lo`` / ``full_hi`` — never the (possibly narrowed)
        ``lo`` / ``hi``.  This is the **only** correct path into
        ``CoreExperiment.measure()``.
        """
        return (phys_freq - self.full_lo) / (self.full_hi - self.full_lo)
