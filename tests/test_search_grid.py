"""Grids sized from the resonance width: uniform spacing, prior FWHM."""

from __future__ import annotations

import numpy as np
import pytest

from nvision.sim.locs.search_grid import (
    SWEEP_POINTS_PER_FWHM,
    max_sweep_points,
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
