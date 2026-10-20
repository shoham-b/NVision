"""Grids sized from the resonance width: uniform spacing, nested refinement, prior FWHM."""

from __future__ import annotations

import numpy as np
import pytest

from nvision.belief.focus_window import FocusRegion
from nvision.sim.locs.search_grid import (
    SWEEP_POINTS_PER_FWHM,
    RefiningGrid,
    max_sweep_points,
    prior_fwhm_phys,
    uniform_grid_phys,
)


def test_uniform_grid_includes_endpoints_and_respects_max_spacing():
    grid = uniform_grid_phys(0.0, 10.0, 3.0)
    assert grid[0] == 0.0
    assert grid[-1] == 10.0
    assert np.max(np.diff(grid)) <= 3.0 + 1e-12
    assert grid.size == 5  # ceil(10 / 3) + 1


def test_uniform_grid_rejects_degenerate_input():
    with pytest.raises(ValueError, match="need lo < hi"):
        uniform_grid_phys(1.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="spacing must be positive"):
        uniform_grid_phys(0.0, 1.0, 0.0)


def test_sweep_spacing_puts_three_points_in_a_fwhm():
    fwhm = 6.0
    grid = uniform_grid_phys(0.0, 100.0, fwhm / SWEEP_POINTS_PER_FWHM)
    spacing = grid[1] - grid[0]
    assert spacing <= fwhm / 3.0
    # Any window of one FWHM holds at least 3 grid points.
    for start in np.linspace(0.0, 100.0 - fwhm, 40):
        assert np.count_nonzero((grid >= start) & (grid <= start + fwhm)) >= 3


def test_max_sweep_points_is_the_count_for_the_narrowest_dip():
    n = max_sweep_points(0.0, 150.0, min_hwhm_phys=0.2, points_per_fwhm=3)
    assert n == uniform_grid_phys(0.0, 150.0, 2 * 0.2 / 3).size


def test_refining_grid_sweeps_leftmost_first_then_refines_by_halving():
    full = FocusRegion.full(0.0, 8.0)
    grid = RefiningGrid(0.0, 8.0, 2.0)  # level 0: 0, 2, 4, 6, 8
    visited = []
    for _ in range(5):
        x = float(grid.next_unmeasured(full)[0])
        visited.append(x)
        grid.record(x)
    assert visited == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert grid.level == 0
    nxt = grid.next_unmeasured(full)
    assert grid.level == 1  # level 0 exhausted -> spacing halves
    assert grid.spacing_phys == pytest.approx(1.0)
    assert list(nxt) == [1.0, 3.0, 5.0, 7.0]  # only the new midpoints are uncovered


def test_refining_grid_stays_inside_the_region():
    grid = RefiningGrid(0.0, 10.0, 1.0)
    region = FocusRegion(sections=((3.0, 5.0),), full_lo=0.0, full_hi=10.0)
    candidates = grid.next_unmeasured(region)
    assert candidates.min() >= 3.0
    assert candidates.max() <= 5.0


def test_refining_grid_levels_are_nested():
    grid = RefiningGrid(0.0, 7.0, 2.0)
    level0 = set(np.round(grid._grid, 9))
    grid._level = 1
    level1 = set(np.round(grid._build_grid(), 9))
    assert level0 <= level1


class _Belief:
    def __init__(self, bounds, priors):
        self.physical_param_bounds = bounds
        self.priors = priors


def test_prior_fwhm_uses_prior_mean_not_midpoint():
    belief = _Belief({"linewidth": (0.2e6, 5.0e6)}, {"linewidth": (0.1, 0.05)})
    expected_hwhm = 0.2e6 + 0.1 * (5.0e6 - 0.2e6)
    assert prior_fwhm_phys(belief) == pytest.approx(2.0 * expected_hwhm)


def test_prior_fwhm_clips_a_prior_mean_outside_the_bounds():
    # The generator draws prior means that can fall below the lower bound; the belief truncates there.
    belief = _Belief({"linewidth": (0.2e6, 5.0e6)}, {"linewidth": (-0.5, 0.05)})
    assert prior_fwhm_phys(belief) == pytest.approx(2.0 * 0.2e6)


def test_prior_fwhm_without_prior_is_the_bounds_midpoint():
    belief = _Belief({"linewidth": (1.0e6, 3.0e6)}, None)
    assert prior_fwhm_phys(belief) == pytest.approx(2.0 * 2.0e6)
