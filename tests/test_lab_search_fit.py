import numpy as np
import pytest

from nvision.metrics.lab_search_fit import (
    SEARCH_FIT_VERSION,
    detection_step,
    detection_weights,
    fit_search_curve,
    fit_van_trees_from_run_detection,
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


def _run(n_d: int, n_steps: int, w: float, j_d: float = 2.0, i_max: float = 2.0) -> np.ndarray:
    """Prior-width plateau, then van Trees from *n_d*: one steep drop onto sigma = 1/sqrt(j_d) at step n_d."""
    n = np.arange(1, n_steps + 1, dtype=float)
    return np.where(n < n_d, w / np.sqrt(12), 1 / np.sqrt(j_d + np.clip(n - n_d, 0, None) * i_max))


def test_detection_is_the_step_after_the_drop():
    assert detection_step(_run(21, 200, 30.0)) == 21


def test_detection_skips_a_drop_the_run_later_undoes():
    # Settles on a candidate at step 5 (sigma 8 -> 0.6, the run's steepest fall), rejects it (back to 6), then finds
    # the dip at step 30 (6 -> 0.5).
    sigma = np.full(100, 8.0)
    sigma[4:10] = 0.6
    sigma[10:29] = 6.0
    sigma[29:] = 0.5 / np.sqrt(1 + np.arange(71) * 0.5)
    assert detection_step(sigma) == 30


def test_detection_skips_drops_much_shallower_than_the_steepest():
    # A halving at step 10 is far below 0.7 of the run's steepest fall (x20 at step 40), so it is search, not detection.
    sigma = np.full(80, 10.0)
    sigma[9:39] = 5.0
    sigma[39:] = 0.25
    assert detection_step(sigma) == 40


def test_detection_is_none_when_sigma_never_falls():
    assert detection_step(np.full(50, 5.0)) is None


def test_detection_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="positive"):
        detection_step(np.array([1.0, 0.0, 0.5]))


def test_van_trees_fit_from_detection_recovers_parameters():
    det = fit_van_trees_from_run_detection([_run(21, 200, 30.0)])
    assert det["step"] == 21
    assert det["j_d_per_mhz2"] == pytest.approx(2.0, rel=0.05)
    assert det["i_max_per_mhz2"] == pytest.approx(2.0, rel=0.05)
    assert det["steps"][0] == 21
    assert len(det["model"]) == len(det["steps"]) == 180


def test_runs_aligned_at_their_own_detection_start_on_the_data():
    # Runs detect at steps 10..50: the median over runs at a fixed step is a smeared cliff, but aligned at each run's
    # detection every run is the same van Trees curve, so the fit starts on sigma(n_d) = 1/sqrt(J_d).
    runs = [_run(n_d, 300, 30.0) for n_d in range(10, 51, 5)]
    det = fit_van_trees_from_run_detection(runs)
    assert det["n_detected"] == det["n_runs"] == 9
    assert det["step"] == 30
    assert det["aligned_median"][0] == pytest.approx(1 / np.sqrt(2.0))
    assert det["model"][0] == pytest.approx(det["aligned_median"][0])  # J_d pinned: starts exactly on the data
    assert det["j_d_per_mhz2"] == pytest.approx(2.0)
    assert det["i_max_per_mhz2"] == pytest.approx(2.0, rel=0.01)


def test_pinned_start_holds_when_the_tail_is_not_van_trees():
    # A first aligned point above the van Trees line: a free J_d would move the start off it, the pinned one cannot.
    sigma = _run(10, 200, 30.0)
    sigma[9] = 0.8  # above the line's 0.71 MHz, still the run's steepest drop
    det = fit_van_trees_from_run_detection([sigma])
    assert det["step"] == 10
    assert det["model"][0] == pytest.approx(0.8)
    assert det["j_d_per_mhz2"] == pytest.approx(1 / 0.8**2)


def test_lags_kept_while_half_the_detected_runs_reach_them():
    det = fit_van_trees_from_run_detection([_run(5, 20, 30.0), _run(5, 100, 30.0), _run(5, 100, 30.0)])
    assert len(det["steps"]) == 96  # two of three runs reach lag 95


def test_no_detection_gives_none():
    assert fit_van_trees_from_run_detection([np.full(50, 5.0)]) is None


def test_search_fit_without_runs_has_no_detection():
    steps = np.arange(1, 301)
    sigma = search_model_sigma(steps, 30.0, 0.5, 0.2, 2.0)
    assert fit_search_curve(sigma, window_mhz=30.0)["detection"] is None
    assert fit_search_curve(sigma, window_mhz=30.0, run_sigmas=[sigma])["detection"] is not None


def test_old_two_parameter_fit_is_not_current():
    assert not is_current({"delta_d_mhz": 1.0, "i_max_per_mhz2": 5.0})
    assert not is_current(None)


def test_fit_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="positive"):
        fit_search_curve(np.array([1.0, 0.0, 0.5]))
