"""Tests ensuring focus-window narrowing never falls back to the full domain.

These tests exercise the two critical code paths that compute the focus window
shown in the UI:

1.  ``StagedSobolSweepLocator.acquisition_window()`` – used by the staged sobol
    initial sweep inside the Bayesian locator.
2.  ``SequentialBayesianLocator.bayesian_focus_window()`` – used for the
    post-sweep focus band drawn on Bayesian plots.

(``GenericSweepLocator`` has no such path: its acquisition window comes
directly from its model fit in ``finalize()``, not from dip-shape inference —
see ``nvision/sim/locs/coarse/generic_sweep_locator.py``.)

All synthetic signals contain a clear dip so *any* of the above paths that
returns the full ``[0, 1]`` domain is considered a failure.
"""

from __future__ import annotations

import numpy as np

from nvision.models.observation import Observation
from nvision.sim.locs.refocus import infer_focus_window as _refocus_infer_focus_window
from nvision.sim.locs.refocus.window import infer_focus_window


def _observation(x: float, y: float) -> Observation:
    return Observation(x=x, signal_value=y)


class TestInferFocusWindowFallbacks:
    """``infer_focus_window`` and helpers must never return the full domain
    when there is a detectable minimum in the data.
    """

    def test_infer_focus_window_with_detectable_dip(self):
        """A dense triple-dip signal must produce a window < 50 % of domain."""
        x = np.linspace(0, 1, 300)
        y = np.ones_like(x)
        for centre in (0.30, 0.50, 0.70):
            y -= 0.9 * np.exp(-0.5 * ((x - centre) / 0.025) ** 2)

        from nvision.models.observation import ObservationHistory

        hist = ObservationHistory(500)
        for xi, yi in zip(x, y, strict=False):
            hist.append(_observation(float(xi), float(yi)))

        lo, hi = infer_focus_window(hist, 0.0, 1.0, expected_dips=3, noise_threshold=0.5)
        assert hi - lo < 0.9, f"infer_focus_window returned too-wide window ({lo}, {hi})"

    def test_refocus_infer_focus_window_no_false_full_domain(self):
        """``_refocus_infer_focus_window`` alias must also narrow."""
        x = np.linspace(0, 1, 200)
        y = np.ones_like(x)
        y -= 0.8 * np.exp(-0.5 * ((x - 0.5) / 0.03) ** 2)
        from nvision.models.observation import ObservationHistory

        hist = ObservationHistory(300)
        for xi, yi in zip(x, y, strict=False):
            hist.append(_observation(float(xi), float(yi)))

        lo, hi = _refocus_infer_focus_window(hist, 0.0, 1.0, noise_threshold=0.5)
        assert hi - lo < 0.5, f"_refocus_infer_focus_window returned too-wide window ({lo}, {hi})"
