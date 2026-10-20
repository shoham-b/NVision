"""SBED reports the sections it actually restricts acquisition to as the UI focus window.

``focus_window_candidates()`` exposes the locator's ``_focus`` sections (what ``_eig_acquire`` filters
EIG candidates by; derived from the significant dips) and feeds the UI timeline
(``StepSnapshot.focus_window_candidates``); ``bayesian_focus_window()`` is their outer extent.
``per_dip_windows()`` is separate, display-only evidence from the belief's deterministic dip detection
(``SMCMarginalDistribution.dip_candidates``), covered by ``test_dip_detection.py``.
"""

from __future__ import annotations

from nvision.belief.dip_detection import DipCandidate
from nvision.belief.focus_window import FocusRegion
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel
from tests.noise import gaussian_noise


def _make_locator() -> SequentialBayesianExperimentDesignLocator:
    model = NVCenterLorentzianModel()
    phys_bounds = {
        "center_freq": (2.86e9, 2.88e9),
        "linewidth": (5e6, 15e6),
        "split": (1e6, 5e6),
        "k_np": (0.5, 1.5),
        "c_total": (0.05, 0.2),
        "background": (0.0, 0.1),
    }
    x_bounds = phys_bounds["center_freq"]
    wrapped_model = UnitCubeSignalModel(model, phys_bounds, x_bounds)
    param_bounds = {name: (0.0, 1.0) for name in phys_bounds}
    belief = SMCMarginalDistribution(
        model=wrapped_model,
        parameter_bounds=param_bounds,
        num_particles=50,
        physical_param_bounds=phys_bounds,
        drive_freq_bounds_phys=x_bounds,
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
        width_hz=(right - left),
    )


def _narrow(locator: SequentialBayesianExperimentDesignLocator, *sections: tuple[float, float]) -> None:
    full = locator._focus
    locator._focus = FocusRegion.from_intervals(list(sections), full_lo=full.full_lo, full_hi=full.full_hi)


class TestFocusWindowMethods:
    def test_full_axis_focus_reports_no_window(self):
        locator = _make_locator()
        assert locator.bayesian_focus_window() is None
        assert locator.focus_window_candidates() is None
        assert locator.per_dip_windows() is None

    def test_narrowed_focus_is_the_reported_window(self):
        locator = _make_locator()
        _narrow(locator, (2.865e9, 2.872e9))
        assert locator.bayesian_focus_window() == (2.865e9, 2.872e9)
        assert locator.focus_window_candidates() == [(2.865e9, 2.872e9)]

    def test_multiple_sections_are_each_reported(self):
        locator = _make_locator()
        _narrow(locator, (2.862e9, 2.864e9), (2.875e9, 2.877e9))
        # The timeline gets the exact sections; the end-of-run field gets their outer extent.
        assert locator.focus_window_candidates() == [(2.862e9, 2.864e9), (2.875e9, 2.877e9)]
        assert locator.bayesian_focus_window() == (2.862e9, 2.877e9)

    def test_significant_dips_drive_the_focus_window(self, monkeypatch):
        # This test axis is only 20 MHz wide: shrink the pad so the window can be narrower than the axis.
        monkeypatch.setattr("nvision.sim.locs.bayesian.sbed_locator.NVISION_SMC_FOCUSING_COVER_FACTOR", 0.1)
        locator = _make_locator()
        locator.belief._dip_candidates = [_dip(2.869e9, 2.871e9)]
        # Dips alone change nothing until the locator re-decides the focus (after a resample)...
        assert locator.bayesian_focus_window() is None
        locator.belief._step_count = 20
        locator._update_focus_window()
        # ...then the reported window is the (padded) dip extent and equals the acquisition filter.
        lo, hi = locator.bayesian_focus_window()
        assert lo < 2.869e9
        assert hi > 2.871e9
        assert locator.focus_window_candidates() == [(lo, hi)]
        assert locator._focus.sections == ((lo, hi),)

    def test_per_dip_windows_still_expose_competing_dips(self):
        locator = _make_locator()
        dominant = (2.869e9, 2.871e9)
        secondary = (2.875e9, 2.877e9)
        locator.belief._dip_candidates = [_dip(*dominant)]
        assert locator.per_dip_windows() is None
        locator.belief._dip_candidates = [_dip(*dominant), _dip(*secondary)]
        assert locator.per_dip_windows() == [dominant, secondary]
