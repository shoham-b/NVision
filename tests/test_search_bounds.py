"""Search-phase bounds: formulas against hand values (nvision/metrics/search_bounds.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nvision.metrics import search_bounds as sb

W = 5.0e6
GAMMA = 1.0e6  # HWHM
DEPTH = 0.25
SIGMA = 0.05


def _lorentzian_line(span: float = 4000.0, n: int = 400_001):
    """Dip ``C gamma^2 / (gamma^2 + u^2)`` and its squared slope on a uniform grid of +-span*gamma."""
    u = np.linspace(-span * GAMMA, span * GAMMA, n)
    dip = DEPTH * GAMMA**2 / (GAMMA**2 + u**2)
    grad = 2.0 * DEPTH * GAMMA**2 * u / (GAMMA**2 + u**2) ** 2  # |d dip / d theta| = |d dip / du|
    return u, dip, grad**2


def test_finite_reach_limit_at_zero_is_prior_variance():
    assert sb.finite_reach_curve(np.array([0.0]), W, 0.4e6)[0] == pytest.approx(W**2 / 12.0)


def test_finite_reach_hand_value_and_zero_past_cover():
    dd = 1.0e6
    n = np.array([2.0, 5.0, 9.0])
    got = sb.finite_reach_curve(n, W, dd)
    assert got[0] == pytest.approx((W - 2 * dd) ** 3 / (12 * W))
    assert got[1:] == pytest.approx([0.0, 0.0])


def test_change_of_measure_at_zero_and_rate():
    at0 = sb.change_of_measure_curve(np.array([0.0]), W, 0.01)[0]
    assert at0 == pytest.approx(W**2 / (8 * math.e * (1 + math.log(2)) ** 2))
    assert at0 * 62.3 / W**2 == pytest.approx(1.0, abs=0.01)  # "about W^2 / 62.3"
    # decays at rate kappa once a >> 1: ratio of two points far out ~ exp(-kappa dn) times a polynomial factor
    n = np.array([1000.0, 1100.0])
    c = sb.change_of_measure_curve(n, W, 0.01)
    a = 1 + math.log(2) + n * 0.01
    assert c[1] / c[0] == pytest.approx(math.exp(-1.0) * (a[0] / a[1]) ** 2)


def test_kappa_bar_matches_lorentzian_closed_form():
    u, dip, g2 = _lorentzian_line()
    ints = sb.line_integrals(u, dip, g2)
    assert ints.dip_sq_integral == pytest.approx(math.pi * GAMMA * DEPTH**2 / 2.0, rel=1e-3)
    assert ints.max_grad_sq == pytest.approx(27.0 / 64.0 * DEPTH**2 / GAMMA**2, rel=1e-4)
    kappa = sb.kappa_bar(ints.dip_sq_integral, W, SIGMA)
    assert kappa == pytest.approx(sb.kappa_bar_lorentzian(W, GAMMA, DEPTH / SIGMA), rel=1e-3)
    # hand number: pi * 1e6 * 25 / (4 * 5e6) = 3.927
    assert kappa == pytest.approx(math.pi * 25.0 / 20.0, rel=1e-3)


def test_kappa_to_fisher_ratio_and_implied_shots():
    u, dip, g2 = _lorentzian_line()
    ints = sb.line_integrals(u, dip, g2)
    kappa = sb.kappa_bar(ints.dip_sq_integral, W, SIGMA)
    fisher = ints.max_grad_sq / SIGMA**2  # one shot
    assert kappa * W / (fisher * GAMMA**3) == pytest.approx(16.0 * math.pi / 27.0, rel=1e-3)
    assert sb.implied_shots(fisher, SIGMA, ints.max_grad_sq) == pytest.approx(1.0)


def test_line_integrals_rejects_bad_grids():
    x = np.array([0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="uniformly"):
        sb.line_integrals(x, np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="equal-length"):
        sb.line_integrals(x, np.ones(2), np.ones(3))


def test_mean_curve_hand_values():
    dd, imax = 1.0e6, 1.0e-12
    n = np.array([0.0, 1.0])
    got = sb.mean_curve(n, W, dd, imax)
    assert got[0] == pytest.approx(W**2 / 12.0)
    # n = 1: search (W-dd)^3/(12W) plus the cell found at step 1 (fraction dd/W, variance dd^2/12)
    assert got[1] == pytest.approx((W - dd) ** 3 / (12 * W) + (dd / W) * dd**2 / 12.0)


def test_mean_curve_tends_to_crlb_when_everything_found():
    dd, imax = 1.0e6, 1.0e-12
    n = np.array([1.0e6])
    # all cells found long ago: sum of weights = 1, variance ~ 1 / (n I)
    assert sb.mean_curve(n, W, dd, imax)[0] == pytest.approx(1.0 / (n[0] * imax), rel=0.02)


def test_survival_floor_hand_value():
    t = 0.5e6
    got = sb.survival_floor_curve(np.array([0.0, 10.0]), W, 0.1, t)
    assert got[0] == pytest.approx(math.exp(-math.log(2) / (1 - 2 * t / W)))
    assert got[1] == pytest.approx(math.exp(-(1.0 + math.log(2)) / (1 - 2 * t / W)))
    with pytest.raises(ValueError, match="0 < 2t < W"):
        sb.survival_floor_curve(np.array([0.0]), W, 0.1, W)


def test_bound_chain_is_consistent_with_survival_floor():
    """eq:search-bound is t^2 x floor at t = W / (2a); a fixed-t value stays within the quoted 1.42 factor."""
    n = np.array([0.0, 5.0, 20.0])
    kappa = 0.2
    com = sb.change_of_measure_curve(n, W, kappa)
    for t in (0.2e6, 0.5e6, 1.0e6):
        fixed = t**2 * sb.survival_floor_curve(n, W, kappa, t)
        assert np.all(fixed <= com * 1.5)  # within the 1.42 factor the article quotes for n kappa <= 50 (plus margin)


def test_probe_spacing_satisfies_reach_equation():
    fwhm = 2 * GAMMA
    reach = sb.probe_spacing(W, fwhm, snr=6.0)
    assert reach is not None
    assert reach.kind == "reach"
    z = math.sqrt(2 * math.log(W / reach.dd_phys)) + sb.Z_BETA
    assert reach.z == pytest.approx(z)
    assert reach.dd_phys == pytest.approx(fwhm * math.sqrt(6.0 / z - 1.0), rel=1e-6)


def test_probe_spacing_core_regime_and_no_reach():
    fwhm = 2 * GAMMA
    core = sb.probe_spacing(W, fwhm, snr=3.8)
    assert core is not None
    assert core.kind == "core"
    assert core.dd_phys == pytest.approx(fwhm / math.sqrt(3.0))
    assert sb.probe_spacing(W, fwhm, snr=1.0) is None  # sqrt(k) SNR <= z: nothing shows in one measurement
    covers = sb.probe_spacing(W, fwhm, snr=30.0)  # the reach exceeds the window
    assert covers is not None
    assert covers.kind == "window"
    assert covers.dd_phys == W


def test_predicted_sweep_ratio():
    assert sb.predicted_sweep_ratio(math.e**5 * 1.0, 1.0) == pytest.approx(1.0 / (2.4 * 5.0))
    with pytest.raises(ValueError, match="W > dd"):
        sb.predicted_sweep_ratio(1.0, 2.0)


def test_align_errors_carry_forward_and_running():
    steps = [np.array([1, 2, 5]), np.array([1, 10])]
    errs = [np.array([10.0, 8.0, 2.0]), np.array([9.0, 1.0])]
    n = np.array([1, 3, 5, 6, 10])
    e, running = sb.align_errors(steps, errs, n)
    assert e[0].tolist() == [10.0, 8.0, 2.0, 2.0, 2.0]  # stopped at 5 keeps the final error
    assert running[0].tolist() == [True, True, True, False, False]
    assert e[1].tolist() == [9.0, 9.0, 9.0, 9.0, 1.0]
    assert sb.usable_columns(running).tolist() == [True, True, True, True, True]  # 1 of 2 running = half is enough
    assert sb.usable_columns(running[:1]).tolist() == [True, True, True, False, False]
    with pytest.raises(ValueError, match="start at step 1"):
        sb.align_errors([np.array([2, 3])], [np.array([1.0, 1.0])], n)


def test_survival_and_kappa_eff_recover_known_rate():
    rng = np.random.default_rng(0)
    kappa_true = 0.02
    detect = rng.exponential(1.0 / kappa_true, 20_000)  # run is wrong until it detects
    n = np.arange(1, 400, dtype=float)
    errors = np.where(n[None, :] < detect[:, None], 5.0, 0.0)
    surv = sb.survival(errors, tolerance_phys=1.0)
    assert sb.fit_kappa_eff(n, surv) == pytest.approx(kappa_true, rel=0.03)
    assert sb.mean_squared_error(errors)[0] == pytest.approx(25.0 * float(np.mean(detect > 1.0)))


def test_fit_kappa_eff_none_when_no_decay():
    n = np.arange(1, 50, dtype=float)
    assert sb.fit_kappa_eff(n, np.ones_like(n)) is None


def test_detection_summary_counts_never_at_budget():
    got = sb.detection_summary([10, 20, None, 40], budget=100)
    assert got.median == pytest.approx(30.0)
    assert got.p75 == pytest.approx(np.percentile([10, 20, 100, 40], 75))
    assert got.capped_share == pytest.approx(0.25)
    assert got.n_runs == 4
    with pytest.raises(ValueError, match="at least one"):
        sb.detection_summary([], 10)


def test_api_n_grid_log_spaced_integers_up_to_budget():
    from nvision.cli.search_bounds_api import n_grid

    grid = n_grid(2500)
    assert grid[0] == 1.0
    assert grid[-1] == 2500.0
    assert np.all(np.diff(grid) > 0)
    assert np.all(grid == np.round(grid))


def test_api_run_rejects_other_primary_and_missing_columns():
    from nvision.cli.search_bounds_api import UseCaseError, _Run

    entries = [{"type": "scan", "series": {"s": [1], "e": [1.0]}, "true_params": {}, "metrics": {}}]
    with pytest.raises(UseCaseError, match="primary parameter"):
        _Run(entries, {"stage_primary_param": "zero_field_freq"})
    with pytest.raises(UseCaseError, match="zeeman_shift bounds"):
        _Run(entries, {"stage_primary_param": "zeeman_shift", "final_err_primary": 1.0})
    with pytest.raises(UseCaseError, match="no scan entry"):
        _Run([], {})


def test_api_run_rejects_series_that_is_not_the_primary_error():
    from nvision.cli.search_bounds_api import UseCaseError, _Run

    entries = [{"type": "scan", "series": {"s": [1], "e": [1.0]}, "true_params": {}, "metrics": {}}]
    with pytest.raises(UseCaseError, match="not the stored final"):
        _Run(entries, {"stage_primary_param": "zeeman_shift", "final_err_primary": 50.0})
