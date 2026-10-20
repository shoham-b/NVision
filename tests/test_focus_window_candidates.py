"""SBED reports the window it actually restricts acquisition to as the UI focus window.

``bayesian_focus_window()`` / ``focus_window_candidates()`` expose the locator's ``_focus`` (the
interval ``_acquire`` filters EIG candidates by) and feed the UI timeline
(``StepSnapshot.focus_window_candidates``). ``per_dip_windows()`` is separate, display-only evidence
from the belief's deterministic dip detection (``SMCMarginalDistribution.dip_candidates``), covered
by ``test_dip_detection.py``.
"""

from __future__ import annotations

from nvision.belief.dip_detection import DipCandidate
from nvision.belief.focus_window import FocusWindow
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
        background=1.0,
    )


def _narrow(locator: SequentialBayesianExperimentDesignLocator, lo: float, hi: float) -> None:
    full = locator._focus
    locator._focus = FocusWindow(lo=lo, hi=hi, full_lo=full.full_lo, full_hi=full.full_hi)


class TestFocusWindowMethods:
    def test_full_axis_focus_reports_no_window(self):
        locator = _make_locator()
        assert locator.bayesian_focus_window() is None
        assert locator.focus_window_candidates() is None
        assert locator.per_dip_windows() is None

    def test_narrowed_focus_is_the_reported_window(self):
        locator = _make_locator()
        _narrow(locator, 2.865e9, 2.872e9)
        assert locator.bayesian_focus_window() == (2.865e9, 2.872e9)
        assert locator.focus_window_candidates() == [(2.865e9, 2.872e9)]
        # The acquisition filter and the reported window are the same interval.
        assert locator._acquisition_bounds() == locator.bayesian_focus_window()

    def test_detected_dips_do_not_change_the_focus_window(self):
        locator = _make_locator()
        locator.belief._dip_candidates = [_dip(2.869e9, 2.871e9)]
        assert locator.bayesian_focus_window() is None
        assert locator.focus_window_candidates() is None

    def test_per_dip_windows_still_expose_competing_dips(self):
        locator = _make_locator()
        dominant = (2.869e9, 2.871e9)
        secondary = (2.875e9, 2.877e9)
        locator.belief._dip_candidates = [_dip(*dominant)]
        assert locator.per_dip_windows() is None
        locator.belief._dip_candidates = [_dip(*dominant), _dip(*secondary)]
        assert locator.per_dip_windows() == [dominant, secondary]
