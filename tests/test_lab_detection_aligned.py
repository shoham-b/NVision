import math

import numpy as np
import pytest

from nvision.metrics import lab_detection_aligned as lda

REF = 1466.0
MODEL = {"baseline": 1.0, "contrast": 0.14, "fwhm_mhz": 1.82, "shape_final_only": False}


def _trace(i: int, *, n: int = 120, drop: int = 10, sigma: float = 0.025, window: float = 40.0) -> dict:
    rng = np.random.default_rng(i)
    std = np.where(np.arange(1, n + 1) < drop, 10.0, 0.5 / np.sqrt(np.arange(1, n + 1)))
    mean = REF + rng.normal(0.0, 1.0, n) * std
    d = rng.uniform(REF - window / 2, REF + window / 2, n)
    y = lda.lorentzian_signal(d, mean[-1], 1.0, 0.14, 1.82) + rng.normal(0.0, sigma, n)
    return {
        "file": f"run{i}.mat",
        "model": dict(MODEL),
        "scan_range_mhz": [REF - window / 2, REF + window / 2],
        "drive_freq_mhz": d.tolist(),
        "signal": y.tolist(),
        "mean_mhz": mean.tolist(),
        "std_mhz": std.tolist(),
        "stop_reason": "budget",
    }


def test_i_max_matches_steepest_slope():
    gamma = 0.91
    expect = (0.14 * 3 * math.sqrt(3) / 8 / gamma) ** 2 / 0.025**2
    assert lda.i_max_per_mhz2(1.0, 0.14, 1.82, 0.025) == pytest.approx(expect)
    # The slope maximum really is 3 sqrt3 / 8 at u = 1/sqrt3.
    u = np.linspace(0, 3, 30001)
    assert np.max(2 * u / (1 + u**2) ** 2) == pytest.approx(lda.STEEPEST_SLOPE, rel=1e-6)


def test_noise_sigma_recovers_the_noise():
    assert lda.noise_sigma(_trace(0, n=2000)) == pytest.approx(0.025, rel=0.1)


def test_noise_sigma_rejects_inferred_shape():
    t = _trace(0)
    t["model"]["shape_final_only"] = True
    with pytest.raises(ValueError, match="shape was inferred"):
        lda.noise_sigma(t)


def test_analyse_window_aligns_at_detection():
    out = lda.analyse_window([_trace(i) for i in range(12)], REF)
    assert out["n_detected"] == 12
    assert out["detection_step_median"] == 10
    assert out["max_lag"] == 120 - 10
    assert out["lags"][0] == 0
    assert out["snr"] == pytest.approx(0.14 / 0.025, rel=0.15)
    assert "rho_elimination_at" in out
    assert "unavailable" in out["gaussian"]
    # Errors drawn at the claimed sigma: the mean claimed variance tracks the MSE.
    assert np.allclose(out["mse"][5], out["mean_claimed_var"][5], rtol=1.0)


def test_analyse_window_rejects_mixed_windows_and_models():
    with pytest.raises(ValueError, match="scan ranges"):
        lda.analyse_window([_trace(0), _trace(1, window=100.0)], REF)
    t = _trace(1)
    t["model"]["fwhm_mhz"] = 3.5
    with pytest.raises(ValueError, match="fwhm_mhz"):
        lda.analyse_window([_trace(0), t], REF)


def test_too_few_detected_runs():
    out = lda.analyse_window([_trace(i) for i in range(3)], REF)
    assert "unavailable" in out


def test_claimed_rate_recovers_the_slope():
    lags = np.arange(0, 200, dtype=float)
    imax = 10.0
    out = lda.claimed_rate(lags, 1.0 / (2.0 + 0.4 * imax * lags), imax)
    assert out["claimed_rate"] == pytest.approx(0.4)


def test_runs_that_stop_at_detection_are_unavailable():
    traces = [_trace(i) for i in range(12)]
    for key in ("mean_mhz", "std_mhz", "signal", "drive_freq_mhz"):
        traces[0][key] = traces[0][key][:11]  # stops one step after its detection at step 10
    out = lda.analyse_window(traces, REF)
    assert "full-budget" in out["unavailable"]


def test_pinned_restart_starts_on_the_claimed_variance():
    out = lda.analyse_window([_trace(i) for i in range(12)], REF)
    assert out["bound_pinned"][0] == pytest.approx(out["mean_claimed_var"][0])
    assert out["rho_pinned_claimed_at"]["0"] == pytest.approx(1.0)


def test_relative_information_peaks_at_the_steepest_slope():
    gamma = 0.91
    d = REF + np.array([0.0, gamma / math.sqrt(3.0), 10.0])
    r = lda.relative_information(d, REF, 1.82)
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(1.0)
    assert r[2] < 1e-3


def test_placement_of_uniform_measurements_is_low():
    out = lda.analyse_window([_trace(i, window=100.0) for i in range(12)], REF)
    assert out["placement"]["placement_efficiency_median"] < 0.1
    assert out["placement"]["far_share_median"] > 0.8
