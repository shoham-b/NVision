"""Refine rates and the speedup (nvision/metrics/refine_rates.py) and the fixed-budget variants."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from nvision.metrics import refine_rates as rr
from nvision.sim.combinations import CombinationGrid
from nvision.sim.grid_enums import base_strategy_name, fixed_budget_name, is_fixed_budget
from nvision.sim.locs.bayesian.sequential_bayesian_locator import ConvergenceStopRules


def _synthetic_runs(q: float, n0: float, i_max: float, n_runs: int = 20, n_max: int = 2000):
    """Runs whose mean square error is exactly 1 / (q i_max (n - n0)) for n > n0 (every run the same)."""
    n = np.unique(np.round(np.geomspace(1, n_max, 80))).astype(float)
    err = np.where(n > n0, (q * i_max * np.maximum(n - n0, 1e-9)) ** -0.5, 1.0)
    return [n] * n_runs, [err] * n_runs, np.full(n_runs, n_max)


def test_known_q_and_n0_are_recovered():
    q, n0, i_max = 0.4, 150.0, 2e-12
    steps, errs, ends = _synthetic_runs(q, n0, i_max)
    inv = rr.inverse_mse_curve(steps, errs, ends, np.linspace(1, 2000, 200))
    window = rr.late_window(inv, [100] * 20)
    assert window == (1000.0, 2000.0)  # the second half of the usable range, after the dip
    fit = rr.fit_rate_line(inv, i_max, window)
    assert fit.q == pytest.approx(q, rel=2e-3)
    assert fit.n0 == pytest.approx(n0, abs=1.0)
    assert fit.r2 == pytest.approx(1.0, abs=1e-4)


def test_stopped_runs_keep_their_final_error_and_late_points_need_half_the_runs():
    n = np.array([1.0, 100.0, 1000.0])
    steps = [n, n, n, n]
    errs = [np.array([10.0, 1.0, 0.5])] * 4
    ends = np.array([1000, 1000, 400, 400])  # two runs stopped at 400 (series ends later: held)
    inv = rr.inverse_mse_curve(steps, errs, ends, np.array([100.0, 1000.0]))
    assert inv.alive_share.tolist() == [1.0, 0.5]
    assert np.isfinite(inv.inv_mse).all()  # exactly half still run at 1000: kept
    ends_low = np.array([1000, 400, 400, 400])
    inv = rr.inverse_mse_curve(steps, errs, ends_low, np.array([1000.0]))
    assert np.isnan(inv.inv_mse[0])  # fewer than half still running


def test_non_positive_slope_is_a_measurement_not_a_failure():
    n = np.linspace(100, 200, 10)
    inv = rr.InverseMse(n, 5.0 - 0.01 * n, np.ones(10))  # the information falls: no growth to report
    fit = rr.fit_rate_line(inv, 1e-12, (100.0, 200.0))
    assert fit.q < 0.0
    assert math.isnan(fit.n0)


def test_place_efficiency_counts_only_measurements_after_the_dip():
    eta = np.array([0.0, 0.0, 0.0, 1.0, 0.5])
    assert rr.place_efficiency(eta, 3) == pytest.approx(0.75)  # steps 4 and 5
    assert rr.place_efficiency(eta, 5) is None  # the run ends at the detection
    with pytest.raises(ValueError, match="1-indexed"):
        rr.place_efficiency(eta, 0)


def test_lorentzian_closed_forms_match_the_numerical_sweep_ratio():
    gamma, window = 1.0, 60.0  # HWHM 1, FWHM 2, W = 30 FWHM
    x = np.linspace(-window / 2, window / 2, 200_001)
    slope = np.gradient(1.0 / (1.0 + (x / gamma) ** 2), x)  # d signal / d centre, up to sign
    q_exact, r_exact = rr.exact_sweep_ratio(slope**2)
    assert q_exact == pytest.approx(rr.lorentzian_sweep_q(2 * gamma, window), rel=0.02)
    assert r_exact == pytest.approx(rr.lorentzian_speedup(2 * gamma, window), rel=0.02)
    assert rr.lorentzian_speedup(1.0, 1.0) == pytest.approx(27 / (8 * math.pi))
    # the steepest point of a Lorentzian is at gamma / sqrt(3)
    assert abs(x[np.argmax(slope**2)]) == pytest.approx(gamma * rr.LORENTZIAN_SLOPE_OFFSET_GAMMA, abs=1e-3)


def test_sweep_positions_repeat_the_grid_pass_after_pass():
    assert rr.sweep_n_points(10.0, 2.0, 2) == math.ceil(10 / 1.0) + 1 == 11
    x = rr.sweep_positions_phys(100.0, 110.0, 6, 14)
    assert x[:6].tolist() == [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    assert x[6:12].tolist() == x[:6].tolist()
    assert x[12] == 100.0


def test_cost_curve_censors_runs_that_never_get_there_and_the_speedup_uses_the_best_sweep():
    steps = np.array([1.0, 10.0, 100.0, 1000.0])
    fast = np.array([5.0, 1.0, 0.1, 0.01])
    slow = np.array([5.0, 4.0, 1.0, 0.1])
    stuck = np.array([5.0, 4.0, 3.0, 2.0])
    taus = np.array([0.5, 0.05])
    sbed = rr.cost_curve([steps, steps], [fast, fast], taus)
    sweep = rr.cost_curve([steps, steps, steps], [slow, slow, stuck], taus)
    assert sbed.t_median.tolist() == [100.0, 1000.0]
    assert sbed.share.tolist() == [1.0, 1.0]
    assert sweep.share.tolist() == pytest.approx([2 / 3, 0.0])
    assert np.isnan(sweep.t_median[1])
    ratio = rr.speedup_curve(sbed, [sweep, rr.cost_curve([steps], [fast], taus)])
    assert ratio.tolist() == [1.0, 1.0]  # the best sweep here is as good as SBED
    only_slow = rr.speedup_curve(sbed, [sweep])
    assert only_slow[0] == pytest.approx(1000.0 / 100.0)
    assert np.isnan(only_slow[1])


def test_prediction_matches_the_line():
    assert rr.predicted_cost(np.array([0.1]), 0.5, 100.0, 1.0)[0] == pytest.approx(100.0 + 100.0 / 0.5)
    with pytest.raises(ValueError, match="finite n0"):
        rr.predicted_cost(np.array([0.1]), 0.5, float("nan"), 1.0)


def test_offsets_are_signed_from_the_nearest_centre():
    off = rr.nearest_centre_offsets(np.array([9.0, 10.5, 19.0, 21.0]), np.array([10.0, 20.0]), 2.0)
    assert off.tolist() == [-0.5, 0.25, -0.5, 0.5]
    hist = rr.offset_histogram(off, np.array([-1.0, 0.0, 1.0]))
    assert hist.fraction.tolist() == [0.5, 0.5]
    assert hist.outside == 0.0
    assert rr.offset_histogram(np.array([]), np.array([-1.0, 1.0])) is None


def test_fixed_budget_names_and_resolution_leave_the_default_grid_alone():
    assert fixed_budget_name("Bayesian-SBED") == "Bayesian-SBED-fixed"
    assert is_fixed_budget("SimpleSweep-3perFWHM-fixed")
    assert not is_fixed_budget("SimpleSweep-3perFWHM")
    assert base_strategy_name("SimpleSweep-3perFWHM-fixed") == "SimpleSweep-3perFWHM"
    grid = CombinationGrid()
    gen = next(iter(grid.generators))
    assert not any(is_fixed_budget(s) for s, _ in grid.strategies_for(gen))  # never in the default grid
    plain = grid.resolve(gen, "Gauss(0.05)", "Bayesian-SBED")
    fixed = grid.resolve(gen, "Gauss(0.05)", "Bayesian-SBED-fixed")
    assert "stop_rule" not in plain.strategy["config"]
    assert fixed.strategy["config"]["stop_rule"] is False
    assert fixed.strategy["class"] is plain.strategy["class"]
    assert grid.resolve(gen, "Gauss(0.05)", "Smart-SBED-fixed") is None  # only the SBED and the sweeps have one


class _Stub(ConvergenceStopRules):
    def __init__(self):
        self.belief = SimpleNamespace(
            model=SimpleNamespace(parameter_names=lambda: ["zeeman_shift"]), uncertainty=lambda: {}
        )
        self.step_count = 0
        self._init_stop_rules(0.1, 1)

    def _target_params_converged(self, uncertainties):
        return True

    def _data_show_dip(self):
        return True


def test_the_stop_rule_can_be_switched_off_and_is_on_by_default():
    on = _Stub()
    on._update_stop_rules()
    assert on._is_converged
    off = _Stub()
    off.stop_rule_enabled = False
    for _ in range(5):
        off._update_stop_rules()
    assert not off._is_converged
