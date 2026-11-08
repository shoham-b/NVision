# ruff: noqa: PT018, PT011, E501
"""The article-claim metrics (nvision/metrics/claims.py) on synthetic curves whose exact answer is known.

Each test builds data from the article's own formula and checks the metric recovers the parameters, and (the point of the
metrics) that it reports a clipped / frozen / starved case honestly rather than confirming the claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nvision.metrics import claims

IMAX = 168.75  # Gamma = 1, SNR = 10, k = 1 (Fig. budget)
W, DD = 20.0, 1.0 / 3.0
N_D = 30.0
GRID = claims.step_grid(1000)


def _repeats(curve: np.ndarray, reps: int = 12) -> np.ndarray:
    return np.tile(curve, (reps, 1))


def test_step_grid_is_increasing_and_ends_at_max() -> None:
    g = claims.step_grid(2500)
    assert g[0] == 1 and g[-1] == 2500 and np.all(np.diff(g) > 0)
    assert list(claims.step_grid(5)) == [1, 2, 3, 4, 5]


def test_elimination_recovers_q_one_and_exact_offset() -> None:
    """Eq. after-detection gives 1/sigma^2 = 12/dd^2 + (n - n_d) Imax: q = 1, n0 = n_d - 12/(dd^2 Imax), not n_d."""
    err = _repeats(claims.elimination_curve(GRID, W, DD, N_D, IMAX))
    fit = claims.fit_strategy_numbers(GRID, err, IMAX, n_min=2 * N_D)
    assert fit.q == pytest.approx(1.0, rel=1e-6)
    assert fit.n0 == pytest.approx(N_D - 12.0 / (DD**2 * IMAX), rel=1e-6)
    assert fit.q_ci is not None and fit.q_ci[0] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("q_true", [0.5, 1.3])
def test_q_is_recovered_and_not_clipped(q_true: float) -> None:
    """A sweep-like q = 0.5 and an impossible q = 1.3 both come back as given (>1 must survive as a defect/overconfidence)."""
    n0 = 7.0
    err = _repeats((q_true * IMAX * (GRID - n0)).clip(min=1e-9) ** -0.5)
    fit = claims.fit_strategy_numbers(GRID, err, IMAX, n_min=50)
    assert fit.q == pytest.approx(q_true, rel=1e-6)
    assert fit.n0 == pytest.approx(n0, rel=1e-6)


def test_stopped_runs_are_not_held_and_end_the_window() -> None:
    """Runs that stop leave the average; a fit window only reaches where >= 90% still run, and a too-short one says why."""
    curve = (IMAX * (GRID - 5.0)).clip(min=1e-9) ** -0.5
    err = _repeats(curve, 20)
    stop = int(np.searchsorted(GRID, 200))
    err[:6, stop:] = np.nan  # 30% of the runs stop at step ~200
    fit = claims.fit_strategy_numbers(GRID, err, IMAX, n_min=50)
    assert fit.n_hi is not None and fit.n_hi <= 200
    assert fit.q == pytest.approx(1.0, rel=1e-6)  # survivors are the same curve here
    starved = claims.fit_strategy_numbers(GRID, err, IMAX, n_min=500)
    assert starved.q is None and "stop before" in (starved.why or "")


def test_flat_error_has_no_q() -> None:
    fit = claims.fit_strategy_numbers(GRID, _repeats(np.full(GRID.size, 3.0)), IMAX, n_min=10)
    assert fit.q is None and "does not grow" in (fit.why or "")


def test_regimes_of_the_elimination_curve() -> None:
    """Plateau near sigma_0, a detection cliff, and the -1/2 tail (claim ii)."""
    sigma0 = W / math.sqrt(12.0)
    err = _repeats(claims.elimination_curve(GRID, W, DD, N_D, IMAX))
    reg = claims.regimes(GRID, err, sigma0)
    assert reg.n_half is not None and abs(reg.n_half - N_D) <= 2
    assert reg.plateau_ratio is not None and 0.7 < reg.plateau_ratio <= 1.0
    assert reg.drop_factor is not None and reg.drop_factor > 3  # the single detection step
    # Local slope is -0.5 n/(n - n0) (n0 ~ n_d here), so just past 2 n_half it is steeper than -1/2.
    assert -0.7 < reg.tail_slope < -0.5


def test_regimes_report_missing_detection() -> None:
    reg = claims.regimes(GRID, _repeats(np.full(GRID.size, 5.0)), 5.0)
    assert reg.n_half is None and reg.why


def test_bound_ratio_is_one_on_the_bound_and_flags_a_violation() -> None:
    j0 = 12.0 / W**2
    on = claims.bound_ratio(GRID, _repeats(claims.van_trees_sigma(GRID, j0, IMAX)), j0, IMAX)
    assert np.allclose(on, 1.0)
    below = claims.bound_ratio(GRID, _repeats(0.5 * claims.van_trees_sigma(GRID, j0, IMAX)), j0, IMAX)
    assert np.allclose(below, 4.0)  # error below the bound: R > 1 is a defect, not a success


def test_prior_gap_and_truth_spread() -> None:
    w = 7.0
    assert claims.prior_gap(4 * math.pi**2 / w**2, 12 / w**2) == pytest.approx(math.pi / math.sqrt(3))
    rng = np.random.default_rng(1)
    uniform = rng.uniform(-w / 2, w / 2, 20000)
    assert claims.truth_prior_spread_ratio(uniform, -w / 2, w / 2) == pytest.approx(1.0, abs=0.02)
    padded = rng.uniform(-w / 2 + 1.5, w / 2 - 1.5, 20000)  # truth kept off the edges
    assert claims.truth_prior_spread_ratio(padded, -w / 2, w / 2) < 0.8


def test_elimination_mean_curve_starts_at_sigma0_and_stays_above_the_cramer_rao_tail() -> None:
    mean = claims.elimination_mean_curve(GRID, W, DD, IMAX)
    assert claims.elimination_mean_curve(np.array([0.0]), W, DD, IMAX)[0] == pytest.approx(W / math.sqrt(12.0))
    assert mean[-1] > claims.van_trees_sigma(GRID, 0.0 + 1e-12, IMAX)[-1]  # never below the Cramer-Rao tail
    with pytest.raises(ValueError):
        claims.elimination_curve(GRID, W, DD, 10_000.0, IMAX)


def test_information_rate_is_q_for_a_linear_gain_and_not_clipped() -> None:
    err = _repeats((1.3 * IMAX * GRID) ** -0.5)
    rate = claims.information_rate_curve(GRID, err, IMAX)
    assert np.allclose(rate[1:], 1.3)


def test_sensitivity_ratio_is_inverse_sqrt_q_without_offset() -> None:
    err = _repeats((0.25 * IMAX * GRID) ** -0.5)
    assert claims.sensitivity_ratio(GRID, err, IMAX, n_min=20) == pytest.approx(1 / math.sqrt(0.25), rel=1e-9)


def test_speedup_prediction_and_cost() -> None:
    assert claims.speedup_prediction(20.0, 1.0) == pytest.approx(27 * 20 / (8 * math.pi))
    assert claims.predicted_cost(10.0, 0.5, 100.0, 0.1) == pytest.approx(10 + 100 / 50)


def test_passage_steps() -> None:
    n = np.arange(1, 11)
    err = np.array([[5, 4, 1, 1, 3, 1, 1, 1, 1, 1.0], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5.0]])
    assert claims.passage_steps(n, err, 2.0)[0] == 6  # dips below at 3 but is back above at 5
    assert np.isnan(claims.passage_steps(n, err, 2.0)[1])


def test_coverage_curve_gate_and_gaussian_share() -> None:
    rng = np.random.default_rng(3)
    n = np.array([10.0, 20.0])
    u = np.ones((4000, 2))
    e = np.abs(rng.normal(size=(4000, 2)))
    gate = np.full(4000, 15.0)  # Gaussian from step 15: only the second column counts
    cov = claims.coverage_curve(e, u, gate, n)
    assert cov["runs"].to_list() == [0, 4000]
    assert np.isnan(cov["within_1"][0])
    assert cov["within_1"][1] == pytest.approx(claims.GAUSS_WITHIN[1], abs=0.03)
    assert cov["within_2"][1] == pytest.approx(claims.GAUSS_WITHIN[2], abs=0.02)
    over = claims.coverage_curve(e, 0.5 * u, gate, n)  # claims half the true spread: overconfident
    assert over["within_1"][1] < 0.4


def test_pit_coverage_detects_overconfidence() -> None:
    rng = np.random.default_rng(4)
    ok = claims.pit_coverage(rng.uniform(size=4000))
    assert ok["central_68"] == pytest.approx(0.68, abs=0.03) and ok["central_95"] == pytest.approx(0.95, abs=0.02)
    over = claims.pit_coverage(np.clip(rng.normal(0.5, 0.4, 4000), 0, 1))  # truth too often in the tails
    assert over["central_68"] < 0.65 and over["below"] + over["above"] > 0.1


def test_far_placement_sign() -> None:
    rng = np.random.default_rng(5)
    mass = rng.uniform(0.0, 0.1, 200)
    res = claims.far_placement(mass + 0.3, mass)
    assert res["diff"] == pytest.approx(0.3) and res["diff_ci"][0] > 0.29
    same = claims.far_placement(mass, mass)
    assert same["diff"] == 0.0


def test_series_matrix_interpolates_and_leaves_nan_after_stop() -> None:
    grid = np.array([1, 2, 4, 8, 16])
    m = claims.series_matrix([[1, 4, 8]], [[4.0, 1.0, None]], grid)
    assert m[0, 0] == pytest.approx(4.0) and m[0, 2] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(2.0)  # log-log midpoint of 4 -> 1 over 1 -> 4
    assert np.isnan(m[0, 3]) is np.True_ or np.isnan(m[0, 4])
    assert np.isnan(m[0, 4])


def test_analyse_use_case_recovers_known_parameters_and_reports_missing_columns() -> None:
    from nvision.metrics.claims_report import analyse_use_case

    steps = [int(v) for v in GRID]
    curve = claims.elimination_curve(GRID, W, DD, N_D, IMAX)
    rows = [
        {
            "compare": {
                "stage_prior_info": 12.0 / W**2,
                "stage_prior_info_var": 12.0 / W**2,
                "stage_fisher_star": IMAX,
                "stage_window_phys": W,
                "stage_fwhm_true_phys": 1.0,
                "stage_tau_phys": 0.05,
                "refine_start_step": 40,
            },
            "series": {"s": steps, "e": [float(v) for v in curve], "u": [float(v) for v in curve]},
            "true_params": None,
            "failure_reason": None,
        }
        for _ in range(12)
    ]
    out = analyse_use_case(rows)
    assert out["fit_true"]["q"] == pytest.approx(1.0, rel=1e-3)
    assert out["fit_claimed"]["q"] == pytest.approx(1.0, rel=1e-3)
    assert out["speedup_predicted"] == pytest.approx(27 * W / (8 * math.pi))
    assert out["pit"] is None and "PIT" in out["missing_columns"]  # recorded only by newer runs
    assert out["coverage"][100]["within_1"] == 1.0  # claimed == true error: always within one sigma
    assert analyse_use_case(rows[:3])["unavailable"]
