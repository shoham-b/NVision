import math

import numpy as np
import pytest

from nvision.cli.detection_aligned_api import aligned_run, analyse_rows
from nvision.metrics import detection_aligned as da

I_MAX = 1e-12  # Hz^-2 per measurement
BUDGET = 200


def _run(
    tau: int, j_tau: float, *, scale: float = 1.0, neg_onset: int | None = 30, claim_scale: float = 1.0, far=None
) -> da.AlignedRun:
    """A run whose squared error after detection is ``scale`` times its restarted bound, exactly; its claimed sigma is
    ``claim_scale`` times the error (1 = calibrated, < 1 = overconfident)."""
    steps = np.arange(1, BUDGET + 1, dtype=np.float64)
    lag = np.clip(steps - (tau - 1), 0, None)
    err = np.sqrt(scale / (j_tau + lag * I_MAX))
    neg = np.full(steps.shape, 0.3)
    if neg_onset is not None:
        neg[steps >= neg_onset] = 0.01
    return da.AlignedRun(steps, err, neg, err * claim_scale, tau, j_tau, far=far, far_mass_tau=0.02)


def test_lag_grid_dense_then_log():
    g = da.lag_grid(500)
    assert g[0] == 0
    assert g[-1] == 500
    assert list(g[: da.DENSE_LAGS]) == list(range(da.DENSE_LAGS))
    assert np.all(np.diff(g) > 0)
    assert list(da.lag_grid(3)) == [0, 1, 2, 3]


def test_restart_is_tau_minus_one_and_alignment():
    r = _run(tau=11, j_tau=1e-10)
    assert r.restart == 10
    lags = np.array([0.0, 5.0])
    e = da.aligned([r], lags, "err")[0]
    assert e[0] == pytest.approx(r.err[9])  # step 10
    assert e[1] == pytest.approx(r.err[14])  # step 15


def test_ratio_is_one_when_error_equals_bound_and_above_when_better():
    lags = da.lag_grid(100)
    for scale, expect in ((1.0, 1.0), (0.5, 2.0)):
        runs = [_run(tau=5 + i, j_tau=1e-10 * (1 + i), scale=scale) for i in range(12)]
        err = da.aligned(runs, lags, "err")
        bound = da.restarted_bound(np.array([r.j_tau for r in runs]), lags, I_MAX)
        assert np.allclose(da.localize_ratio(err, bound), expect)


def test_common_max_lag_and_past_budget():
    runs = [_run(tau=5, j_tau=1e-10), _run(tau=51, j_tau=1e-10)]
    assert da.common_max_lag(runs, BUDGET) == BUDGET - 50
    with pytest.raises(ValueError, match="past the budget"):
        da.common_max_lag([_run(tau=5, j_tau=1e-10)], 4)


def test_gaussian_onset():
    s = np.array([1, 2, 3, 4, 5], dtype=float)
    assert da.gaussian_onset(s, np.array([0.3, 0.3, 0.01, 0.02, 0.01])) == 3
    assert da.gaussian_onset(s, np.array([0.3, 0.01, 0.3, 0.02, 0.01])) == 4  # must stay below to the end
    assert da.gaussian_onset(s, np.array([0.01, 0.01, 0.01, 0.01, 0.3])) is None
    assert da.gaussian_onset(s, np.array([np.nan, np.nan, 0.01, 0.01, 0.01])) == 3
    assert da.gaussian_onset(s, np.full(5, np.nan)) is None


def test_bound_rejects_zero_information_at_lag_zero():
    with pytest.raises(ValueError, match="must be positive"):
        da.restarted_bound(np.array([0.0]), np.array([0.0, 1.0]), I_MAX)


def test_analyse_end_to_end_and_too_few_runs():
    runs = [_run(tau=5 + i, j_tau=1e-10, scale=1.0) for i in range(12)]
    out = da.analyse(runs, budget=BUDGET, i_max=I_MAX, dd_phys=1e6, n_undetected=1)
    assert out["n_runs"] == 12
    assert out["n_undetected"] == 1
    assert out["rho_j_tau_at"]["10"] == pytest.approx(1.0)
    assert out["bound_violated"] is False
    assert out["j_d"] == pytest.approx(12e-12)
    assert out["gaussian"]["share_reached"] == 1.0
    assert out["gaussian"]["onset_step_median"] == 30
    assert "unavailable" in da.analyse(runs[:3], budget=BUDGET, i_max=I_MAX, dd_phys=None, n_undetected=0)


def test_better_than_bound_is_flagged():
    runs = [_run(tau=5 + i, j_tau=1e-10 * (1 + i % 3), scale=0.5) for i in range(12)]
    out = da.analyse(runs, budget=BUDGET, i_max=I_MAX, dd_phys=None, n_undetected=0)
    assert out["bound_violated"] is True
    assert "rho_j_d" not in out


def _row(tau: int | None, *, with_g: bool = True) -> dict:
    s = list(range(1, BUDGET + 1))
    return {
        "series": {
            "step": s,
            "error": [None] + [1e5 / math.sqrt(k) for k in s[1:]],
            "negentropy": [None, None] + [0.01] * (BUDGET - 2) if with_g else None,
            "claimed_sigma": [None] + [0.5e5 / math.sqrt(k) for k in s[1:]],
            "far_placed": [None] + [0.0] * (BUDGET - 1),
        },
        "compare": {
            "detection_step": tau,
            "j_tau_phys": 1e-10,
            "i_max": I_MAX,
            "far_mass_tau": 0.1,
            "fwhm_phys": 2e6,
            "snr": 10.0,
        },
        "true_params": {"bounds": {"zeeman_shift": [1e6, 40e6]}},
    }


def test_api_rows():
    assert aligned_run(_row(None)) is None
    a = aligned_run(_row(5))
    assert a.steps[0] == 2
    assert np.isnan(a.negentropy[0])
    assert a.detected_step == 5
    with pytest.raises(ValueError, match="no stored s/e/g/u"):
        aligned_run(_row(5, with_g=False))
    out = analyse_rows([_row(5 + i) for i in range(11)] + [_row(None)])
    assert out["n_runs"] == 11
    assert out["n_undetected"] == 1
    assert out["dd_phys"] > 0


def test_claimed_rate_calibrated_and_overconfident():
    runs = [_run(tau=5 + i, j_tau=1e-10) for i in range(12)]
    cal = da.analyse(runs, budget=BUDGET, i_max=I_MAX, dd_phys=None, n_undetected=0)["claimed_rate"]
    d = cal["from_detection"]
    assert d["q_hat_over_q"] == pytest.approx(1.0)
    assert d["q_hat"] == pytest.approx(d["q_true"])
    assert d["q_hat_over_q_ci"][0] <= 1.0 <= d["q_hat_over_q_ci"][1]
    over = [_run(tau=5 + i, j_tau=1e-10, claim_scale=0.5) for i in range(12)]
    o = da.analyse(over, budget=BUDGET, i_max=I_MAX, dd_phys=None, n_undetected=0)["claimed_rate"]["from_detection"]
    assert o["q_hat_over_q"] == pytest.approx(4.0)  # sigma half as large: 1/sigma^2 four times as steep
    assert o["q_hat_over_q_ci"][0] > 1.0


def test_claimed_rate_from_gaussian_starts_at_onset():
    runs = [_run(tau=5 + i, j_tau=1e-10, neg_onset=30 + i) for i in range(12)]
    out = da.analyse(runs, budget=BUDGET, i_max=I_MAX, dd_phys=None, n_undetected=0)["claimed_rate"]
    g = out["from_gaussian"]
    assert g["window"] == [0, BUDGET - 41]  # the latest onset (step 41) bounds the common lags
    assert g["window"][1] < out["from_detection"]["window"][1]
    assert g["q_hat_over_q"] == pytest.approx(1.0)
    never = [_run(tau=5 + i, j_tau=1e-10, neg_onset=None) for i in range(12)]
    n = da.analyse(never, budget=BUDGET, i_max=I_MAX, dd_phys=None, n_undetected=0)["claimed_rate"]
    assert "unavailable" in n["from_gaussian"]


def test_far_share():
    steps = np.arange(1, BUDGET + 1)
    # tau = 11 (restart 10), Gaussian from step 30: far at steps 11..20 only
    far = np.where((steps >= 11) & (steps <= 20), 1.0, 0.0)
    far[0] = np.nan
    runs = [_run(tau=11, j_tau=1e-10, far=far) for _ in range(12)]
    f = da.analyse(runs, budget=BUDGET, i_max=I_MAX, dd_phys=None, n_undetected=0)["far_placed"]
    assert f["converge_share_median"] == pytest.approx(10 / 20)  # steps 11..30
    assert f["after_detection_share_median"] == pytest.approx(10 / (BUDGET - 10))
    assert f["far_mass_tau_median"] == pytest.approx(0.02)
    plain = [_run(tau=11, j_tau=1e-10) for _ in range(12)]
    out = da.analyse(plain, budget=BUDGET, i_max=I_MAX, dd_phys=None, n_undetected=0)
    assert out["far_placed"]["converge_share_median"] is None


def test_claimed_must_be_positive_and_aligned():
    r = _run(tau=5, j_tau=1e-10)
    with pytest.raises(ValueError, match="claimed"):
        da.AlignedRun(r.steps, r.err, r.negentropy, np.zeros_like(r.err), 5, 1e-10)
    with pytest.raises(ValueError, match="equal-length"):
        da.AlignedRun(r.steps, r.err, r.negentropy, r.err[:-1], 5, 1e-10)
