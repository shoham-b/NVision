"""Uniform drive-frequency grids sized from the resonance width, shared by the sweep and smart locators.

When the dip's width is already known (a previously classified resonance whose position moved), the
grid spacing is not a tuning knob but follows from how many points must land inside the dip:

* A dip is only *seen* when several measurements fall inside it. With ``SWEEP_POINTS_PER_FWHM = 3``
  points per full width at half maximum (FWHM), the grid cannot step over a dip, and the three
  points are what the binomial test in :func:`nvision.belief.dip_detection.find_dips` needs to call
  a cluster of low points a dip rather than noise. This is the spacing of every sweep.
* The adaptive locators (Smart-Sweep, Smart-SBED) search the very same grid, ``COARSE_SEARCH_POINTS_PER_FWHM`` =
  ``SWEEP_POINTS_PER_FWHM``: their search phase *is* the reference sweep until the focus, and takes its measurements
  from the sweep's persisted repeat (``nvision.runner.sweep_data``) instead of measuring again. Smart-Sweep visits
  the points in order, Smart-SBED lets expected information gain pick among the unmeasured ones; either way a
  point's value is the sweep's. (Before this they used 2 points per FWHM, chosen from the plain-sweep spacing study.)
* Smart-Sweep's focus sweep uses a finer grid, ``FOCUS_SWEEP_POINTS_PER_FWHM`` = 10 points per FWHM, laid fresh
  inside each focus section. At least 7 points per FWHM are needed for a grid point to lie within the refine-mode
  efficiency tolerance of the max-information point; 10 also gives the posterior enough measurements to reach the
  refine switch within one pass (see the constant).

Widths come from the locator's *prior* (``2 * effective_hwhm(belief.estimates())`` at construction),
never from the true signal.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
from numpy.typing import NDArray

from nvision.belief.focus_window import FocusRegion
from nvision.spectra.nv_center import effective_hwhm

# Points per FWHM so a dip cannot be stepped over and ``find_dips`` has the cluster it needs.
SWEEP_POINTS_PER_FWHM: int = 3
# Search grid of Smart-Sweep and Smart-SBED before the focus: the reference sweep's grid (Smart-SBED's EIG picks
# among its points).
COARSE_SEARCH_POINTS_PER_FWHM: int = SWEEP_POINTS_PER_FWHM

# Grid of Smart-Sweep's focus sweep. Lower bound: fine enough that wherever the max-information point x* lies, a grid
# point is within the refine-mode efficiency tolerance of it. A measurement offset eps from x* keeps
# eta ~ 1 - 4.5 (eps/HWHM)^2 of the optimal information (Lorentzian slope maximum x0 +- HWHM/sqrt(3)); eta >= 0.9
# needs eps <= 0.15 HWHM, and the worst case x* halfway between grid points gives spacing 2 eps = 0.30 HWHM =
# FWHM/6.7, i.e. 7 points per FWHM. That is not enough points, though: at 7 one sweep over the focus ended before the
# posterior was tight enough to switch to refine (1 of 8 Lorentzian/Voigt test runs). 10 is the empirical choice
# (0 of 8; 14 also 0 of 8, so it only adds measurements).
FOCUS_SWEEP_POINTS_PER_FWHM: int = 10

# Refusing to build grids denser than this: it means the spacing is far below any useful resolution.
_MAX_GRID_POINTS: int = 2_000_000


def prior_fwhm_phys(belief) -> float:
    """FWHM (Hz) the belief's *prior* implies: ``2 * effective_hwhm`` of the prior-mean parameters.

    A parameter's prior mean is its Gaussian prior's mean when it has one (the previously classified
    value), else the midpoint of its bounds. Deterministic: it reads the prior, not the sampled particles.
    """
    means: dict[str, float] = {}
    for name, (lo, hi) in belief.physical_param_bounds.items():
        prior = (belief.priors or {}).get(name)
        if prior is not None and not isinstance(prior[0], str):
            # Priors are stored on the unit cube; a drawn mean can lie outside the bounds (the belief truncates
            # there), so the usable mean is clipped to them.
            means[name] = lo + min(max(float(prior[0]), 0.0), 1.0) * (hi - lo)
        else:
            means[name] = 0.5 * (lo + hi)
    fwhm = 2.0 * float(effective_hwhm(means))
    if not (fwhm > 0.0) or not math.isfinite(fwhm):
        raise ValueError(f"prior_fwhm_phys: the belief implies a non-positive or non-finite FWHM ({fwhm!r} Hz).")
    return fwhm


def max_sweep_points(lo_phys: float, hi_phys: float, min_hwhm_phys: float, points_per_fwhm: int) -> int:
    """Most grid points a sweep can need: the count for the narrowest dip the prior allows.

    ``min_hwhm_phys`` is the lower bound of the hwhm prior; the prior-mean FWHM never falls below
    ``2 * min_hwhm_phys``, so a sweep sized from the prior never exceeds this count.
    """
    return uniform_grid_phys(lo_phys, hi_phys, 2.0 * min_hwhm_phys / points_per_fwhm).size


def uniform_grid_phys(lo_phys: float, hi_phys: float, max_spacing_phys: float) -> NDArray[np.float64]:
    """Evenly spaced points from ``lo_phys`` to ``hi_phys`` (both included) with spacing <= ``max_spacing_phys``.

    Returns:
        Ascending drive frequencies in Hz. shape: (n_points,), with n_points >= 2.
    """
    if not (hi_phys > lo_phys):
        raise ValueError(f"uniform_grid_phys: need lo < hi, got [{lo_phys}, {hi_phys}] Hz.")
    if not (max_spacing_phys > 0.0) or not math.isfinite(max_spacing_phys):
        raise ValueError(f"uniform_grid_phys: spacing must be positive and finite, got {max_spacing_phys!r} Hz.")
    n_points = math.ceil((hi_phys - lo_phys) / max_spacing_phys) + 1
    if n_points > _MAX_GRID_POINTS:
        raise ValueError(
            f"uniform_grid_phys: spacing {max_spacing_phys:g} Hz over [{lo_phys}, {hi_phys}] Hz needs "
            f"{n_points} points (> {_MAX_GRID_POINTS})."
        )
    return np.linspace(lo_phys, hi_phys, max(n_points, 2))


def _nearest_distance(sorted_points: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance (Hz) from each of ``values`` to the closest of ``sorted_points``.

    shape of ``sorted_points``: (n_points,), ascending and non-empty; shape of ``values``: (n_values,).
    Returns shape: (n_values,).
    """
    idx = np.searchsorted(sorted_points, values)
    left = np.abs(values - sorted_points[np.clip(idx - 1, 0, len(sorted_points) - 1)])
    right = np.abs(values - sorted_points[np.clip(idx, 0, len(sorted_points) - 1)])
    return np.minimum(left, right)


class _PassGrid:
    """Shared bookkeeping of the grids that sweep their points and, once all are measured, sweep them again.

    A grid point is covered by a recorded measurement within half a spacing of it. When every point of the
    region asked about is covered, the measurements covering those points are forgotten and a new pass over the
    same points begins (same spacing, same points), so the grid never runs out; the run's own stopping rules
    end it. Positions are Hz (physical); every array is ascending, shape (n_points,).
    """

    _spacing: float

    def __init__(self) -> None:
        self._measured: list[float] = []
        self._passes = 0

    @property
    def passes(self) -> int:
        """Number of complete passes over the grid so far (0 while the first pass is still running)."""
        return self._passes

    def record(self, drive_freq_phys: float) -> None:
        """Note that ``drive_freq_phys`` (Hz) was measured."""
        bisect.insort(self._measured, float(drive_freq_phys))

    def _half_spacing(self) -> float:
        return 0.5 * self._spacing * (1.0 - 1e-9)

    def _uncovered(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Mask over ``points`` (Hz): no measurement lies within half a spacing of the point."""
        if not self._measured:
            return np.ones(points.shape, dtype=bool)
        return _nearest_distance(np.asarray(self._measured, dtype=np.float64), points) > self._half_spacing()

    def _candidates(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """The uncovered ``points``, starting a new pass over them when none is left.

        Raises ``ValueError`` if ``points`` is empty: there is nothing to repeat.
        """
        if points.size == 0:
            raise ValueError(f"{type(self).__name__}: the region holds no grid point.")
        candidates = points[self._uncovered(points)]
        if candidates.size:
            return candidates
        measured = np.asarray(self._measured, dtype=np.float64)
        self._measured = measured[_nearest_distance(points, measured) > self._half_spacing()].tolist()
        self._passes += 1
        return points


class RepeatingGrid(_PassGrid):
    """A uniform grid over the drive-frequency axis, swept again from the start once every point is measured.

    :meth:`next_unmeasured` returns the grid points inside a :class:`FocusRegion` not yet covered by a
    measurement in the current pass; when the region is exhausted the pass repeats over the same points.
    """

    def __init__(self, lo_phys: float, hi_phys: float, spacing_phys: float) -> None:
        super().__init__()
        self._grid = uniform_grid_phys(float(lo_phys), float(hi_phys), float(spacing_phys))
        self._spacing = float(self._grid[1] - self._grid[0])

    @property
    def spacing_phys(self) -> float:
        """Actual spacing (Hz) of the grid."""
        return self._spacing

    def next_unmeasured(self, region: FocusRegion) -> NDArray[np.float64]:
        """Grid points inside ``region`` still uncovered, repeating the pass when none is.

        Returns:
            Ascending Hz positions. shape: (n_candidates,), never empty.
        """
        return self._candidates(self._grid[region.contains(self._grid)])


class FocusSweepGrid(_PassGrid):
    """A fresh uniform grid laid inside each focus section, independent of the search measurements.

    Every section ``[lo, hi]`` of a :class:`FocusRegion` gets its own grid starting at ``lo`` with spacing
    ``<= spacing_phys`` (both ends included). Only measurements recorded here (the focus sweep's own) cover a
    grid point -- those within half a spacing of it; the coarser search measurements that located the dip do not.
    The grid is rebuilt from the current sections on every call, so it follows the focus as it moves. Once the
    sweep has covered the whole focus it sweeps it again, from the lower edge, at the same spacing.
    """

    def __init__(self, spacing_phys: float) -> None:
        if not (spacing_phys > 0.0) or not math.isfinite(spacing_phys):
            raise ValueError(f"FocusSweepGrid: spacing must be positive and finite, got {spacing_phys!r} Hz.")
        super().__init__()
        self._spacing = float(spacing_phys)

    @property
    def spacing_phys(self) -> float:
        """Maximum spacing (Hz) of the grid laid in each section."""
        return self._spacing

    def next_unmeasured(self, region: FocusRegion) -> NDArray[np.float64]:
        """Grid points of ``region`` not yet covered by a focus-sweep measurement, repeating the pass when none is.

        Returns:
            Ascending Hz positions. shape: (n_candidates,), never empty.
        """
        return self._candidates(
            np.concatenate([uniform_grid_phys(lo, hi, self._spacing) for lo, hi in region.sections])
        )
