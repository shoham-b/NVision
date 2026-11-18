import numpy as np
import pytest

from nvision.metrics.lab_search_fit import (
    SEARCH_FIT_VERSION,
    detection_weights,
    fit_search_curve,
    is_current,
    search_model_sigma,
)


def test_detection_weights_sum_to_one_with_partial_last_cell():
    w = detection_weights(window=10.0, v=3.0)
    assert w.size == 4
    assert w[:3] == pytest.approx([0.3, 0.3, 0.3])
    assert w.sum() == pytest.approx(1.0)


def test_model_starts_near_prior_and_ends_at_cramer_rao():
    steps = np.arange(1, 2001)
    s = search_model_sigma(steps, window=30.0, v=1.0, j_d=12.0, i_max=100.0)
    assert s[0] == pytest.approx(30.0 / 12**0.5, rel=0.1)
    assert s[-1] == pytest.approx(1 / np.sqrt(2000 * 100.0), rel=0.05)
    assert np.all(np.diff(s) < 0)


def test_van_trees_starts_at_detection_with_j_d():
    # Window cleared at the first measurement: every run detects at n_d = 1 and follows 1/(J_d + (n-1) I).
    steps = np.arange(1, 50)
    s = search_model_sigma(steps, window=2.0, v=2.0, j_d=3.0, i_max=7.0)
    assert s == pytest.approx(1 / np.sqrt(3.0 + (steps - 1) * 7.0))


def test_slow_search_does_not_pin_the_dip_at_detection():
    # v and J_d are independent: a slow search (v small) with a loose detection (small J_d).
    steps = np.arange(1, 301)
    s = search_model_sigma(steps, window=50.0, v=0.5, j_d=0.05, i_max=0.3)
    after = s[100:]  # search over: what is left is the van Trees average
    assert after[0] > 0.1  # not the 12/v^2 cell (sigma ~ 0.14 MHz) the old model forced


def test_fit_recovers_generating_parameters():
    steps = np.arange(1, 301)
    sigma = search_model_sigma(steps, 30.0, 0.5, 0.2, 2.0)
    sigma = sigma * np.exp(np.random.default_rng(0).normal(0, 0.02, sigma.size))
    fit = fit_search_curve(sigma, window_mhz=30.0)
    assert fit["version"] == SEARCH_FIT_VERSION
    assert is_current(fit)
    assert fit["v_mhz_per_step"] == pytest.approx(0.5, rel=0.2)
    assert fit["i_max_per_mhz2"] == pytest.approx(2.0, rel=0.3)
    assert fit["search_end_step"] == pytest.approx(60.0, rel=0.2)
    assert fit["rms_log"] < 0.1


def test_old_two_parameter_fit_is_not_current():
    assert not is_current({"delta_d_mhz": 1.0, "i_max_per_mhz2": 5.0})
    assert not is_current(None)


def test_fit_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="positive"):
        fit_search_curve(np.array([1.0, 0.0, 0.5]))
