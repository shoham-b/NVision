"""Segmented log-log fit of claimed uncertainty against measurements (nvision/metrics/loglog_fit.py)."""

from __future__ import annotations

import numpy as np
import pytest

from nvision.metrics.loglog_fit import fit_segments


def _curve(k1: float, k2: float, s2: float, s3: float, n_max: int = 2500, sigma0: float = 4.8e5, noise: float = 0.03):
    """Flat until ``k1``, slope ``s2`` until ``k2``, slope ``s3`` after, with multiplicative noise (seeded)."""
    n = np.arange(1, n_max + 1, dtype=float)
    y = np.log(sigma0) + np.where(n > k1, s2 * np.log(np.clip(n, k1, k2) / k1), 0.0)
    y = y + np.where(n > k2, s3 * np.log(np.clip(n, k2, None) / k2), 0.0)
    rng = np.random.default_rng(7)
    return n, np.exp(y + rng.normal(0.0, noise, n.size))


def test_recovers_flat_cliff_tail_and_marks_refine() -> None:
    n, sigma = _curve(k1=15, k2=60, s2=-1.5, s3=-0.5)
    fit = fit_segments(n, sigma)
    assert fit is not None
    assert fit["n_segments"] == 3
    s1, s2, s3 = fit["slopes"]
    assert abs(s1) < 0.1
    assert s2 == pytest.approx(-1.5, abs=0.3)
    assert s3 == pytest.approx(-0.5, abs=0.08)
    assert fit["breaks_n"][0] == pytest.approx(15, rel=0.35)
    assert fit["breaks_n"][1] == pytest.approx(60, rel=0.35)
    assert fit["refine"] is True


def test_flat_tail_is_not_refine() -> None:
    """A sweep levels off after the cliff: slope ~ 0 in the tail means no refine stage, not a slow one."""
    n, sigma = _curve(k1=15, k2=60, s2=-1.5, s3=0.0)
    fit = fit_segments(n, sigma)
    assert fit is not None
    assert abs(fit["slopes"][-1]) < 0.1
    assert fit["refine"] is False


def test_tail_still_on_the_cliff_is_not_refine() -> None:
    """A run that stops while still falling steeply has no -1/2 tail yet (flat then cliff: two segments)."""
    n, sigma = _curve(k1=15, k2=60, s2=-1.5, s3=-1.5, n_max=400)
    fit = fit_segments(n, sigma)
    assert fit is not None
    assert fit["n_segments"] == 2
    assert fit["slopes"][-1] < -0.8
    assert fit["refine"] is False


def test_flat_for_the_whole_run_has_no_breakpoints() -> None:
    """The dip was never found: one flat segment, nothing to mark, no refine."""
    n, sigma = _curve(k1=1e9, k2=1e9, s2=0.0, s3=0.0)
    fit = fit_segments(n, sigma)
    assert fit is not None
    assert fit["n_segments"] == 1
    assert fit["breaks_n"] == []
    assert fit["refine"] is False


def test_single_power_law_is_not_a_refine_stage() -> None:
    """One straight -1/2 line from the first measurement has no search or focus: not three stages."""
    n = np.arange(1, 2501, dtype=float)
    rng = np.random.default_rng(3)
    sigma = 4.8e5 * n**-0.5 * np.exp(rng.normal(0.0, 0.03, n.size))
    fit = fit_segments(n, sigma)
    assert fit is not None
    assert fit["n_segments"] == 1
    assert fit["slopes"][0] == pytest.approx(-0.5, abs=0.05)
    assert fit["refine"] is False


def test_flat_then_tail_is_two_segments_with_refine() -> None:
    """Search then a straight -1/2 fall, no distinct focus cliff."""
    n = np.arange(1, 2501, dtype=float)
    rng = np.random.default_rng(5)
    sigma = 4.8e5 * np.maximum(n / 30.0, 1.0) ** -0.5 * np.exp(rng.normal(0.0, 0.03, n.size))
    fit = fit_segments(n, sigma)
    assert fit is not None
    assert fit["n_segments"] == 2
    assert fit["breaks_n"][0] == pytest.approx(30, rel=0.4)
    assert fit["slopes"][-1] == pytest.approx(-0.5, abs=0.08)
    assert fit["refine"] is True


def test_too_short_curve_returns_none() -> None:
    assert fit_segments([1, 2, 3, 4], [5.0, 4.0, 3.0, 2.0]) is None


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        fit_segments([1, 2, 3], [1.0, 2.0])


def test_drops_invalid_points() -> None:
    n, sigma = _curve(k1=15, k2=60, s2=-1.5, s3=-0.5)
    sigma[5] = np.nan
    sigma[9] = -1.0
    assert fit_segments(n, sigma) is not None
