"""Synthetic known-answer tests of nvision.metrics.claims: each metric must pass a case built to satisfy its claim and
fail one built to violate it."""

import math

import numpy as np
import pytest

from nvision.metrics import claims
from nvision.metrics.claims_report import analyse_use_case

I_MAX, J0 = 4.0, 1.0


def _runs(n_runs, sigma_fn, grid, seed=0):
    """Runs whose error at n is sigma_fn(n) * |N(0, 1)| (Gaussian, so MSE = sigma_fn^2), stored on ``grid``."""
    rng = np.random.default_rng(seed)
    steps = [grid for _ in range(n_runs)]
    vals = [sigma_fn(grid) * np.abs(rng.standard_normal(grid.size)) + 1e-12 for _ in range(n_runs)]
    return steps, vals


def test_truth_spread_ratio():
    t = np.linspace(-0.5, 0.5, 20001)
    assert claims.truth_spread_ratio(t, -0.5, 0.5) == pytest.approx(1.0, rel=1e-3)
    assert claims.truth_spread_ratio(0.5 * t, -0.5, 0.5) == pytest.approx(0.5, rel=1e-3)


def test_regimes_four_boundaries():
    grid = claims.step_grid(2000)
    sigma0, q = 10.0, 1.0
    r = np.where(grid < 50, sigma0, (1.0 / sigma0**2 + q * I_MAX * np.maximum(grid - 49, 0)) ** -0.5)
    reg = claims.regimes(grid, np.tile(r, (12, 1)), sigma0, detection_step=50.0, gaussian_onset=80.0, q=q, imax=I_MAX)
    assert reg.plateau_ratio == pytest.approx(1.0)
    assert (reg.detection_step, reg.gaussian_onset) == (50.0, 80.0)
    # the data limit is reached once the offset n0 = 49 - 1/(q I sigma0^2) stops mattering (within 10 %)
    assert 50 < reg.converge_end < 1000
    assert reg.tail_slope == pytest.approx(-0.5, abs=0.12)  # the offset n0 steepens the local slope


def test_regimes_report_what_is_missing():
    grid = claims.step_grid(100)
    reg = claims.regimes(
        grid, np.full((10, grid.size), 9.0), 10.0, detection_step=None, gaussian_onset=None, q=None, imax=I_MAX
    )
    assert reg.detection_step is None
    assert reg.converge_end is None
    assert reg.plateau_ratio is None
    assert len(reg.why) >= 3
    # a fit exists but the error stays far above the data limit: no converge end, with a reason
    reg = claims.regimes(
        grid, np.full((10, grid.size), 9.0), 10.0, detection_step=5.0, gaussian_onset=6.0, q=1.0, imax=I_MAX
    )
    assert reg.converge_end is None
    assert any("data limit" in w for w in reg.why)


def test_gate_passes_on_a_straight_line_and_fails_on_a_kink():
    grid = claims.step_grid(2000)
    q, n0 = 0.4, 30.0
    win = (200.0, 2000.0)
    steps, vals = _runs(40, lambda n: (q * I_MAX * np.maximum(n - n0, 1.0)) ** -0.5, grid)
    ok = claims.fit_with_gate(steps, vals, grid, I_MAX, win, bootstrap=60)
    assert ok.linear is True
    assert ok.fit.rate == pytest.approx(q, rel=0.3)

    def kinked(n):  # the information rate triples at n = 700: a kink, not a line
        return np.where(
            n < 700, (0.2 * I_MAX * n) ** -0.5, (0.2 * I_MAX * 700 + 0.9 * I_MAX * np.maximum(n - 700, 0)) ** -0.5
        )

    steps, vals = _runs(40, kinked, grid, seed=1)
    bad = claims.fit_with_gate(steps, vals, grid, I_MAX, win, bootstrap=60)
    assert bad.linear is False
    assert bad.why


def test_rate_ratio_detects_overconfidence():
    grid = claims.step_grid(2000)
    win = (200.0, 2000.0)
    steps, true_v = _runs(40, lambda n: (0.3 * I_MAX * n) ** -0.5, grid)
    _, claimed_v = _runs(40, lambda n: (0.6 * I_MAX * n) ** -0.5, grid, seed=2)
    ft = claims.fit_with_gate(steps, true_v, grid, I_MAX, win, bootstrap=20)
    fc = claims.fit_with_gate(steps, claimed_v, grid, I_MAX, win, bootstrap=20)
    assert claims.rate_ratio(fc, ft) == pytest.approx(2.0, rel=0.3)


def test_sensitivity_ratio_is_one_over_sqrt_q():
    grid = claims.step_grid(2000)
    err = np.tile((0.25 * I_MAX * grid) ** -0.5, (10, 1))
    assert claims.sensitivity_ratio(grid, err, I_MAX, (100.0, 2000.0)) == pytest.approx(2.0)


def test_coverage_calibrated_vs_overconfident():
    rng = np.random.default_rng(3)
    grid = np.array([10.0, 100.0, 1000.0])
    unc = np.ones((4000, 3))
    calibrated = np.abs(rng.standard_normal((4000, 3)))
    gate = np.zeros(4000)
    c = claims.coverage_curve(calibrated, unc, gate, grid)[1]
    assert c["within_1"] == pytest.approx(0.683, abs=0.03)
    assert c["within_2"] == pytest.approx(0.954, abs=0.02)
    over = claims.coverage_curve(3 * calibrated, unc, gate, grid)[1]
    assert over["within_1"] < 0.3
    assert over["within_1_ci"][1] < 0.683  # the interval excludes the nominal value: the claim fails


def test_coverage_gate_excludes_runs_before_their_gate():
    err, unc = np.ones((20, 2)), np.ones((20, 2)) * 2.0
    gate = np.full(20, math.nan)
    gate[:12] = 5.0
    c = claims.coverage_curve(err, unc, gate, np.array([1.0, 10.0]))
    assert c[0]["runs_gated"] == 0
    assert c[0]["within_1"] is None
    assert c[1]["runs_gated"] == 12


def test_coverage_from_z():
    z = np.random.default_rng(4).standard_normal(5000)
    c = claims.coverage_from_z(z)
    assert c["within_1"] == pytest.approx(0.683, abs=0.03)
    assert c["std"] == pytest.approx(1.0, abs=0.05)
    assert claims.coverage_from_z(3 * z)["within_1"] < 0.4


def test_coverage_from_pit_uniform_is_calibrated():
    pit = np.random.default_rng(5).uniform(size=5000)
    c = claims.coverage_from_pit(pit)
    assert c["within_1"] == pytest.approx(0.68, abs=0.03)
    assert c["within_2"] == pytest.approx(0.95, abs=0.02)
    assert c["within_1_ci"][0] < 0.68 < c["within_1_ci"][1]
    assert c["std"] == pytest.approx(1 / math.sqrt(12), abs=0.01)


def test_coverage_from_pit_overconfident_is_u_shaped():
    pit = np.concatenate([np.full(50, 0.001), np.full(50, 0.999), np.full(20, 0.5)])
    c = claims.coverage_from_pit(pit)
    assert c["within_1"] == pytest.approx(20 / 120)
    assert c["within_2"] == pytest.approx(20 / 120)


def test_coverage_from_pit_boundaries_inclusive():
    pit = np.array([0.16, 0.84, 0.025, 0.975] * 5)
    c = claims.coverage_from_pit(pit)
    assert c["within_1"] == 0.5
    assert c["within_2"] == 1.0


def test_coverage_from_pit_rejects_bad_input():
    with pytest.raises(ValueError, match="PIT"):
        claims.coverage_from_pit(np.array([0.5, 1.5] * 10))
    with pytest.raises(ValueError, match="PIT"):
        claims.coverage_from_pit(np.array([0.5, np.nan] * 10))
    with pytest.raises(ValueError, match="PIT"):
        claims.coverage_from_pit(np.array([0.5]))


def test_wilson_interval_rejects_bad_counts():
    with pytest.raises(ValueError, match="hits"):
        claims.wilson_interval(5, 3)


def test_value_matrix_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        claims.value_matrix([np.array([1.0, 2.0])], [np.array([1.0, np.nan])], np.array([1.0]))


def test_report_end_to_end_on_stored_rows():
    grid = claims.step_grid(1000)
    rows = []
    for i in range(12):
        rng = np.random.default_rng(i)
        e = ((J0 + grid * I_MAX) ** -0.5 * (1.0 + 0.1 * rng.random(grid.size))).tolist()
        rows.append(
            {
                "series": {
                    "step": grid.tolist(),
                    "error": e,
                    "claimed_sigma": e,
                    "negentropy": np.where(grid < 60, 0.5, 0.01).tolist(),
                },
                "compare": {
                    "line_walk_step_phys": 0.0,
                    "i_max": I_MAX,
                    "window_phys": 10.0,
                    "fwhm_phys": 1.0,
                    "sigma_target_phys": 0.05,
                    "detection_step": 20,
                    "prior_kind": "uniform",
                    "sigma0_phys": 0.3,
                },
                "true_params": {"params": {"zeeman_shift": 0.1 * i}, "bounds": {"zeeman_shift": [0.0, 1.1]}},
            }
        )
    out = analyse_use_case(rows)
    assert out["sigma0_phys"] == pytest.approx(1.1 / math.sqrt(12))  # W / sqrt(12) of the drawn zeeman_shift range
    rms = out["rms_curve"]
    assert len(rms["n"]) == len(rms["error"]) == len(rms["claimed_sigma"])
    assert rms["error"] == pytest.approx(rms["claimed_sigma"])  # e == u in these rows
    assert rms["error"][-1] < rms["error"][0]
    assert out["coverage_pit_final"] is None  # these rows store no pit series
    assert out["fitted_rate"]["rate"] == pytest.approx(1.0, rel=0.15)
    assert out["claimed_over_fitted_rate"] == pytest.approx(1.0)
    assert out["gaussian_share"] == 1.0
    assert out["regimes"]["detection_step"] == 20.0
    assert 60 <= out["regimes"]["gaussian_onset_step"] <= 70
    assert out["regimes"]["converge_end_step"] is not None
    assert out["coverage_gated_final"]["runs_gated"] == 12
    assert "unavailable" in analyse_use_case(rows[:3])


def _tracking_rows(err_scale: float = 1.0, prior: object = (0.5, 0.2)) -> list[dict]:
    """12 tracking runs whose error sits at ``err_scale`` x van Trees from n = 0 with the honest prior std 0.2; the
    locator's own (broad-mixture) prior spread is stored as 0.9 to tell the two apart."""
    grid = claims.step_grid(1000)
    rows = []
    for i in range(12):
        rng = np.random.default_rng(i)
        e = (err_scale * (1.0 / 0.04 + grid * I_MAX) ** -0.5 * (1.0 + 0.02 * rng.standard_normal(grid.size))).tolist()
        rows.append(
            {
                "series": {"step": grid.tolist(), "error": e, "claimed_sigma": e, "negentropy": [0.01] * grid.size},
                "compare": {
                    "line_walk_step_phys": 0.0,
                    "i_max": I_MAX,
                    "detection_step": 1,
                    "prior_kind": "gaussian",
                    "sigma0_phys": 0.9,
                },
                "true_params": {
                    "params": {"zeeman_shift": 0.5},
                    "bounds": {"zeeman_shift": [0.0, 1.1], "_priors": {"zeeman_shift": prior}},
                },
            }
        )
    return rows


def test_report_tracking_uses_the_truths_gaussian_prior_not_the_locators_belief():
    rows = _tracking_rows()
    out = analyse_use_case(rows)
    assert out["prior_kind"] == "gaussian"
    assert out["sigma0_phys"] == pytest.approx(0.2)  # the honest prior's std, not W / sqrt(12) nor the belief's 0.9
    assert out["locator_prior_sigma_phys"] == pytest.approx(0.9)
    vt = out["van_trees_from_start"]
    assert vt["rho_at"]["1"] == pytest.approx(1.0, abs=0.05)  # errors at the bound: rho ~ 1 at every n
    assert vt["rho_at"]["1000"] == pytest.approx(1.0, abs=0.05)
    assert analyse_use_case(_tracking_rows(err_scale=1.5))["van_trees_from_start"]["violated"] is False
    below = analyse_use_case(_tracking_rows(err_scale=0.5))["van_trees_from_start"]
    assert below["violated"] is True
    assert below["rho_max"] == pytest.approx(4.0, rel=0.1)
    with pytest.raises(ValueError, match="no Gaussian zeeman_shift prior"):
        analyse_use_case(_tracking_rows(prior=("sin^2", 1.0)))
    with pytest.raises(ValueError, match="no Gaussian zeeman_shift prior"):
        analyse_use_case(_tracking_rows(prior=None))
    mixed = [dict(r, compare={**r["compare"], "prior_kind": "uniform"}) if i == 0 else r for i, r in enumerate(rows)]
    with pytest.raises(ValueError, match="one stored prior_kind"):
        analyse_use_case(mixed)


def test_report_excludes_runs_that_never_reach_the_gaussian_peak():
    grid = claims.step_grid(1000)
    rows = []
    for i in range(12):
        e = ((J0 + grid * I_MAX) ** -0.5).tolist()
        g = np.where(grid < 60, 0.5, 0.01) if i < 9 else np.full(grid.size, 0.5)  # 3 of 12 stay non-Gaussian
        rows.append(
            {
                "series": {"step": grid.tolist(), "error": e, "claimed_sigma": e, "negentropy": g.tolist()},
                "compare": {
                    "line_walk_step_phys": 0.0,
                    "i_max": I_MAX,
                    "detection_step": 20,
                    "prior_kind": "uniform",
                    "sigma0_phys": 0.3,
                },
                "true_params": {"params": {"zeeman_shift": 0.1 * i}, "bounds": {"zeeman_shift": [0.0, 1.1]}},
            }
        )
    out = analyse_use_case(rows)
    assert out["gaussian_share"] == pytest.approx(0.75)
    assert out["coverage_gated_final"]["runs_gated"] == 9
    bad = [dict(r, series={k: v for k, v in r["series"].items() if k != "negentropy"}) for r in rows]
    with pytest.raises(ValueError, match="negentropy"):
        analyse_use_case(bad)
