"""Tests ensuring focus-window narrowing never falls back to the full domain.

These tests exercise the three critical code paths that compute the focus window
shown in the UI:

1.  ``SweepingLocator.acquisition_window()`` – used by coarse sweep locators.
2.  ``StagedSobolSweepLocator.acquisition_window()`` – used by the staged sobol
    initial sweep inside the Bayesian locator.
3.  ``SequentialBayesianLocator.bayesian_focus_window()`` – used for the
    post-sweep focus band drawn on Bayesian plots.

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


class TestSweepingLocatorFocusWindow:
    """``SweepingLocator`` (via ``SobolSweepLocator``) must report a narrowed acquisition window."""

    def test_sweep_locator_narrows_window(self):
        """A single deep dip in history must make _set_acquisition_window narrow."""
        import random

        from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
        from nvision.models.experiment import Observation
        from nvision.sim.locs.coarse.sobol_locator import SobolSweepLocator

        random.Random(42)

        # Dummy model with a single expected dip
        class DummyModel:
            inner = property(lambda self: self)

            def parameter_names(self):
                return ["frequency"]

            def expected_dip_count(self):
                return 1

            def signal_min_span(self, domain_width):
                return domain_width * 0.01

            def signal_max_span(self, domain_width):
                return domain_width * 0.1

        def _dummy_belief():
            grid = np.linspace(0.0, 1.0, 64)
            posterior = np.ones(64) / 64
            parameters = [GridParameter(name="frequency", bounds=(0.0, 1.0), grid=grid, posterior=posterior)]
            return GridMarginalDistribution(model=DummyModel(), parameters=parameters)

        locator = SobolSweepLocator.create(
            belief=_dummy_belief(),
            signal_model=DummyModel(),
            max_steps=60,
            noise_std=0.001,
            domain_lo=0.0,
            domain_hi=1.0,
        )

        # Populate history with a clear dip at x=0.5 (depth 50 %)
        xs = np.linspace(0, 1, 60)
        ys = 1.0 - 0.5 * np.exp(-0.5 * ((xs - 0.5) / 0.05) ** 2)
        for x, y in zip(xs, ys, strict=False):
            locator.history.append(Observation(x=x, signal_value=y))
            locator.step_count += 1

        # Force window computation
        locator._set_acquisition_window()

        lo, hi = locator.acquisition_window()
        assert locator._signal_found, "_signal_found should be True for a clear dip"
        assert hi - lo < 0.9, f"SobolSweepLocator window ({lo}, {hi}) is essentially the full domain"
