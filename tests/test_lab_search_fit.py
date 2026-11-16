import numpy as np
import pytest

from nvision.metrics.lab_search_fit import fit_search_curve, search_model_sigma


def test_model_starts_near_prior_and_ends_at_cramer_rao():
    steps = np.arange(1, 2001)
    s = search_model_sigma(steps, window=30.0, delta_d=1.0, i_max=100.0)
    assert s[0] == pytest.approx(30.0 / 12**0.5, rel=0.1)
    assert s[-1] == pytest.approx(1 / np.sqrt(2000 * 100.0), rel=0.05)
    assert np.all(np.diff(s) < 0)


def test_fit_recovers_generating_parameters():
    steps = np.arange(1, 301)
    sigma = search_model_sigma(steps, 30.0, 1.5, 50.0)
    sigma = sigma * np.exp(np.random.default_rng(0).normal(0, 0.02, sigma.size))
    fit = fit_search_curve(sigma)
    assert fit["delta_d_mhz"] == pytest.approx(1.5, rel=0.3)
    assert fit["i_max_per_mhz2"] == pytest.approx(50.0, rel=0.3)
    assert fit["rms_log"] < 0.1


def test_fit_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="positive"):
        fit_search_curve(np.array([1.0, 0.0, 0.5]))
