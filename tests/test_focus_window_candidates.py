"""SBED reports the dips its observations show as focus windows.

The windows come straight from the belief's deterministic dip detection
(``SMCMarginalDistribution.dip_candidates``) and feed the UI timeline, which animates the
candidates narrowing down to a single settled focus window
(``StepSnapshot.focus_window_candidates``). The detector itself is covered by
``test_dip_detection.py``.
"""

from __future__ import annotations

from nvision.belief.dip_detection import DipCandidate
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel
from tests.noise import gaussian_noise


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
        noise_model=gaussian_noise(),
    )
    return SequentialBayesianExperimentDesignLocator(belief=belief, max_steps=10, candidate_step_hz=200e3)


def _dip(left: float, right: float) -> DipCandidate:
    return DipCandidate(
        centroid_hz=0.5 * (left + right),
        significance=5.0,
        n_points=5,
        f_min=left,
        f_max=right,
        confidence=0.99,
        background=1.0,
    )


class TestFocusWindowCandidateMethods:
    def test_no_dips_yet_all_return_none(self):
        locator = _make_locator()
        assert locator.belief.dip_candidates == []
        assert locator.bayesian_focus_window() is None
        assert locator.per_dip_windows() is None
        assert locator.focus_window_candidates() is None

    def test_single_dominant_candidate_is_not_treated_as_multi_dip(self):
        locator = _make_locator()
        locator.belief._dip_candidates = [_dip(2.869e9, 2.871e9)]
        assert locator.bayesian_focus_window() == (2.869e9, 2.871e9)
        # Gated to len >= 2 -- matches sweep locators' per_dip_windows convention.
        assert locator.per_dip_windows() is None
        # Never gated -- always reflects whatever is currently known, even len == 1.
        assert locator.focus_window_candidates() == [(2.869e9, 2.871e9)]

    def test_multiple_competing_candidates_are_all_exposed(self):
        locator = _make_locator()
        dominant = (2.869e9, 2.871e9)
        secondary = (2.875e9, 2.877e9)
        locator.belief._dip_candidates = [_dip(*dominant), _dip(*secondary)]
        assert locator.bayesian_focus_window() == dominant
        assert locator.per_dip_windows() == [dominant, secondary]
        assert locator.focus_window_candidates() == [dominant, secondary]

    def test_candidates_collapse_to_one_as_run_settles(self):
        """Simulates the timeline story: many candidates narrowing to one."""
        locator = _make_locator()
        dominant = (2.869e9, 2.871e9)
        secondary = (2.875e9, 2.877e9)

        locator.belief._dip_candidates = [_dip(*dominant), _dip(*secondary)]
        assert len(locator.focus_window_candidates()) == 2

        locator.belief._dip_candidates = [_dip(*dominant)]
        assert len(locator.focus_window_candidates()) == 1
        assert locator.per_dip_windows() is None
        assert locator.bayesian_focus_window() == dominant
