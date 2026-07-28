"""_max_dip_cluster_span_hz prefers the belief's current estimate over the prior bound.

Regression coverage for the fix where background-point classification stayed
pinned at the Zeeman prior's worst-case bound (e.g. 200 MHz) for the entire
run instead of shrinking as the belief localized the true split.
"""

from __future__ import annotations

from nvision.sim.locs.bayesian.sbed_locator import _max_dip_cluster_span_hz


def test_falls_back_to_bound_when_no_estimate_given():
    bounds = {"zeeman_split": (0.0, 100e6), "split": (0.0, 4e6)}
    assert _max_dip_cluster_span_hz(bounds) == 2.0 * 100e6 + 2.0 * 4e6


def test_falls_back_to_bound_when_estimate_lacks_zeeman_split():
    bounds = {"zeeman_split": (0.0, 100e6)}
    assert _max_dip_cluster_span_hz(bounds, est={"frequency": 2.87e9}) == 2.0 * 100e6


def test_uses_current_estimate_instead_of_worst_case_bound():
    bounds = {"zeeman_split": (0.0, 100e6), "split": (0.0, 4e6)}
    est = {"zeeman_split": 12e6, "split": 2.16e6}
    result = _max_dip_cluster_span_hz(bounds, est)
    assert result == 2.0 * 12e6 + 2.0 * 2.16e6
    assert result < 2.0 * 100e6 + 2.0 * 4e6


def test_no_splitting_estimated_returns_none():
    bounds = {"zeeman_split": (0.0, 100e6)}
    est = {"zeeman_split": 0.0}
    assert _max_dip_cluster_span_hz(bounds, est) is None


def test_no_splitting_bounds_returns_none():
    assert _max_dip_cluster_span_hz({}) is None
