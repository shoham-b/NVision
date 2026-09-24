"""Regression coverage for the per-step focus-window-candidate plumbing.

Added so the UI timeline can animate SBED's dip candidates narrowing down to a
single settled focus window (see ``FocusWindowConfidence.all_candidates`` and
``StepSnapshot.focus_window_candidates``). These tests exercise only the new
surface -- the dip detector itself (``identify_dip_candidates``) already has
full coverage in ``test_dip_detection.py``.
"""

from __future__ import annotations

from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.sim.locs.bayesian.sbed_locator import FocusWindowConfidence, SequentialBayesianExperimentDesignLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


def _make_locator() -> SequentialBayesianExperimentDesignLocator:
    model = NVCenterLorentzianModel()
    phys_bounds = {
        "frequency": (2.86e9, 2.88e9),
        "linewidth": (5e6, 15e6),
        "split": (1e6, 5e6),
        "k_np": (0.5, 1.5),
        "c_total": (0.05, 0.2),
        "background": (0.0, 0.1),
    }
    x_bounds = phys_bounds["frequency"]
    wrapped_model = UnitCubeSignalModel(model, phys_bounds, x_bounds)
    param_bounds = {name: (0.0, 1.0) for name in phys_bounds}
    belief = UnitCubeSMCMarginalDistribution(
        model=wrapped_model,
        parameter_bounds=param_bounds,
        num_particles=50,
        physical_param_bounds=phys_bounds,
        physical_x_bounds=x_bounds,
    )
    return SequentialBayesianExperimentDesignLocator(belief=belief, max_steps=10, candidate_step_hz=200e3)


def _conf(*candidates: tuple[float, float], is_stable: bool = False) -> FocusWindowConfidence:
    left, right = candidates[0]
    return FocusWindowConfidence(
        left_bound=left,
        right_bound=right,
        left_unc=1e5,
        right_unc=1e5,
        detector_confidence=0.99,
        background=1.0,
        center=0.5 * (left + right),
        center_std=1e3,
        center_ci_lo=left,
        center_ci_hi=right,
        methods_agree=True,
        is_stable=is_stable,
        all_candidates=tuple(candidates),
    )


class TestFocusWindowCandidateMethods:
    def test_no_conf_yet_all_return_none(self):
        locator = _make_locator()
        assert locator._focus_window_conf is None
        assert locator.bayesian_focus_window() is None
        assert locator.per_dip_windows() is None
        assert locator.focus_window_candidates() is None

    def test_single_dominant_candidate_is_not_treated_as_multi_dip(self):
        locator = _make_locator()
        locator._focus_window_conf = _conf((2.869e9, 2.871e9))
        assert locator.bayesian_focus_window() == (2.869e9, 2.871e9)
        # Gated to len >= 2 -- matches sweep locators' per_dip_windows convention.
        assert locator.per_dip_windows() is None
        # Never gated -- always reflects whatever is currently known, even len == 1.
        assert locator.focus_window_candidates() == [(2.869e9, 2.871e9)]

    def test_multiple_competing_candidates_are_all_exposed(self):
        locator = _make_locator()
        dominant = (2.869e9, 2.871e9)
        secondary = (2.875e9, 2.877e9)
        locator._focus_window_conf = _conf(dominant, secondary)
        assert locator.bayesian_focus_window() == dominant
        assert locator.per_dip_windows() == [dominant, secondary]
        assert locator.focus_window_candidates() == [dominant, secondary]

    def test_candidates_collapse_to_one_as_run_settles(self):
        """Simulates the timeline story: many candidates narrowing to one."""
        locator = _make_locator()
        dominant = (2.869e9, 2.871e9)
        secondary = (2.875e9, 2.877e9)

        locator._focus_window_conf = _conf(dominant, secondary)
        assert len(locator.focus_window_candidates()) == 2

        locator._focus_window_conf = _conf(dominant, is_stable=True)
        assert len(locator.focus_window_candidates()) == 1
        assert locator.per_dip_windows() is None
        assert locator.bayesian_focus_window() == dominant
