"""Coordinate-system value objects for the NVision inference pipeline.

Two orthogonal transforms operate on measurement data:

``RescaleMap``
    Fixed [0, 1] <-> physical mapping for one parameter.  Created once at
    belief construction from ``physical_param_bounds`` and never mutated.
    Owned by the belief.  All parameters carry one.

``FocusWindow`` (see ``nvision.belief.focus_window``)
    Physical [lo, hi] sub-interval of the frequency axis that the locator
    currently probes.  Can narrow during a run (always returns a new
    instance).  Owned by the locator.  Carries immutable ``full_lo``/
    ``full_hi`` so ``CoreExperiment.measure()`` normalisation is always
    relative to the original full domain.

These two are **orthogonal**.  Conflating them -- e.g. using
``physical_param_bounds["frequency"]`` for both rescaling and acquisition
bounds -- is the root coordinate-system defect this module fixes.
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

    @property
    def width(self) -> float:
        """Physical width of the parameter range."""
        return self.hi - self.lo
