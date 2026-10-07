"""Focus-only Bayesian locator: SBED's belief + focusing, bisection instead of EIG for acquisition."""

from __future__ import annotations

from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.sim.locs.bayesian.sobol_bayesian_locator import van_der_corput


class BisectionFocusLocator(SequentialBayesianExperimentDesignLocator):
    """Keeps the SBED belief, focusing and stopping rules; acquisition is a plain binary search.

    Inside the current focus window ``[lo, hi]`` the probe positions follow bisection order: the
    midpoint, then the two quarter points, then the four eighth points, ... (the base-2 van der
    Corput sequence scaled onto the window). Whenever the focus window changes the sequence
    restarts on the new window. There is no EIG evaluation and no uniform exploration probe.

    The focus only narrows or expands when ``frequency`` is an inferred particle dimension (see
    ``_update_focus_window``); with the default fixed ``frequency`` the window stays the full probe
    axis and this degenerates to a plain dyadic sweep.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Focus the current bisection run belongs to, and how many points it has produced.
        self._bisect_focus = self._focus
        self._bisect_index: int = 0

    def _acquire(self) -> float:
        """Next measurement x (physical Hz): the next bisection point of the current focus window."""
        if self._focus != self._bisect_focus:
            self._bisect_focus = self._focus
            self._bisect_index = 0
        self._bisect_index += 1
        lo_phys, hi_phys = self._acquisition_bounds()
        return float(lo_phys + van_der_corput(self._bisect_index) * (hi_phys - lo_phys))
