"""Grids sized from the resonance width: uniform spacing, repeated passes, prior FWHM."""

from __future__ import annotations

import numpy as np
import pytest

from nvision.belief.focus_window import FocusRegion
from nvision.sim.locs.search_grid import (
    SWEEP_POINTS_PER_FWHM,
    FocusSweepGrid,
    RepeatingGrid,
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


def test_repeating_grid_sweeps_leftmost_first_then_sweeps_the_same_points_again():
    full = FocusRegion.full(0.0, 8.0)
    grid = RepeatingGrid(0.0, 8.0, 2.0)  # 0, 2, 4, 6, 8
    visited = []
    for _ in range(12):
        x = float(grid.next_unmeasured(full)[0])
        visited.append(x)
        grid.record(x)
    assert visited == [0.0, 2.0, 4.0, 6.0, 8.0] * 2 + [0.0, 2.0]  # the second pass repeats the first
    assert grid.passes == 2
    assert grid.spacing_phys == pytest.approx(2.0)  # never refined


def test_focus_sweep_grid_starts_at_each_section_edge_and_ignores_unrecorded_measurements():
    region = FocusRegion(sections=((3.1, 5.0), (7.0, 8.0)), full_lo=0.0, full_hi=10.0)
    grid = FocusSweepGrid(0.5)
    pts = grid.next_unmeasured(region)
    assert pts[0] == pytest.approx(3.1)  # anchored at the lower edge, not on a global 0.5 lattice
    assert pts.max() == pytest.approx(8.0)
    assert np.all(np.diff(pts[pts < 6.0]) <= 0.5 + 1e-9)


def test_focus_sweep_grid_sweeps_the_focus_again_when_it_is_covered():
    region = FocusRegion(sections=((0.0, 2.0),), full_lo=0.0, full_hi=10.0)
    grid = FocusSweepGrid(1.0)  # 0, 1, 2
    visited = []
    for _ in range(7):
        x = float(grid.next_unmeasured(region)[0])
        visited.append(x)
        grid.record(x)
    assert visited == [0.0, 1.0, 2.0] * 2 + [0.0]  # the same grid again, from the lower edge
    assert grid.passes == 2
    assert grid.spacing_phys == pytest.approx(1.0)  # never refined


def test_repeating_a_region_keeps_the_measurements_of_the_other_sections():
    left = FocusRegion(sections=((0.0, 2.0),), full_lo=0.0, full_hi=10.0)
    right = FocusRegion(sections=((6.0, 8.0),), full_lo=0.0, full_hi=10.0)
    grid = FocusSweepGrid(1.0)
    for _ in range(3):
        grid.record(float(grid.next_unmeasured(right)[0]))  # right section fully measured: 6, 7, 8
    for _ in range(4):  # the left section is swept, then repeated once
        grid.record(float(grid.next_unmeasured(left)[0]))
    assert grid.passes == 1
    assert list(grid.next_unmeasured(left)) == [1.0, 2.0]
    assert grid.passes == 1  # the right section's pass was not restarted by the left one


def test_a_region_without_grid_points_cannot_repeat():
    grid = RepeatingGrid(0.0, 10.0, 2.0)
    empty = FocusRegion(sections=((2.2, 3.8),), full_lo=0.0, full_hi=10.0)
    with pytest.raises(ValueError, match="holds no grid point"):
        grid.next_unmeasured(empty)


def test_repeating_grid_stays_inside_the_region():
    grid = RepeatingGrid(0.0, 10.0, 1.0)
    region = FocusRegion(sections=((3.0, 5.0),), full_lo=0.0, full_hi=10.0)
    candidates = grid.next_unmeasured(region)
    assert candidates.min() >= 3.0
    assert candidates.max() <= 5.0


class _Belief:
    def __init__(self, bounds, priors):
        self.physical_param_bounds = bounds
        self.priors = priors


def test_prior_fwhm_uses_prior_mean_not_midpoint():
    belief = _Belief({"hwhm": (0.2e6, 5.0e6)}, {"hwhm": (0.1, 0.05)})
    expected_hwhm = 0.2e6 + 0.1 * (5.0e6 - 0.2e6)
    assert prior_fwhm_phys(belief) == pytest.approx(2.0 * expected_hwhm)


def test_prior_fwhm_clips_a_prior_mean_outside_the_bounds():
    # The generator draws prior means that can fall below the lower bound; the belief truncates there.
    belief = _Belief({"hwhm": (0.2e6, 5.0e6)}, {"hwhm": (-0.5, 0.05)})
    assert prior_fwhm_phys(belief) == pytest.approx(2.0 * 0.2e6)


def test_prior_fwhm_without_prior_is_the_bounds_midpoint():
    belief = _Belief({"hwhm": (1.0e6, 3.0e6)}, None)
    assert prior_fwhm_phys(belief) == pytest.approx(2.0 * 2.0e6)
