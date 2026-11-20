import math

import numpy as np
import pytest

from nvision.cli.detection_aligned_api import aligned_run, analyse_rows
from nvision.metrics import detection_aligned as da

I_MAX = 1e-12  # Hz^-2 per measurement
BUDGET = 200


def _run(tau: int, j_tau: float, *, scale: float = 1.0, neg_onset: int | None = 30) -> da.AlignedRun:
    """A run whose squared error after detection is ``scale`` times its restarted bound, exactly."""
    steps = np.arange(1, BUDGET + 1, dtype=np.float64)
    lag = np.clip(steps - (tau - 1), 0, None)
    err = np.sqrt(scale / (j_tau + lag * I_MAX))
    neg = np.full(steps.shape, 0.3)
    if neg_onset is not None:
        neg[steps >= neg_onset] = 0.01
    return da.AlignedRun(steps, err, neg, tau, j_tau)


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
            "s": s,
            "e": [None] + [1e5 / math.sqrt(k) for k in s[1:]],
            "g": [None, None] + [0.01] * (BUDGET - 2) if with_g else None,
        },
        "compare": {
            "dip_detected_step": tau,
            "posterior_J_tau_phys": 1e-10,
            "stage_fisher_star": I_MAX,
            "stage_fwhm_true_phys": 2e6,
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
    with pytest.raises(ValueError, match="no stored s/e/g"):
        aligned_run(_row(5, with_g=False))
    out = analyse_rows([_row(5 + i) for i in range(11)] + [_row(None)])
    assert out["n_runs"] == 11
    assert out["n_undetected"] == 1
    assert out["dd_phys"] > 0
