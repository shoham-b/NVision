"""Uniform drive-frequency grids sized from the resonance width, shared by the sweep and smart locators.

When the dip's width is already known (a previously classified resonance whose position moved), the
grid spacing is not a tuning knob but follows from how many points must land inside the dip:

* A dip is only *seen* when several measurements fall inside it. With ``SWEEP_POINTS_PER_FWHM = 3``
  points per full width at half maximum (FWHM), the grid cannot step over a dip, and the three
  points are what the binomial test in :func:`nvision.belief.dip_detection.find_dips` needs to call
  a cluster of low points a dip rather than noise. This is the spacing of every sweep.
* The adaptive locators (Smart-Sweep, Smart-SBED) start from the same FWHM/3 grid: Smart-Sweep measures it in
  order, Smart-SBED lets expected information gain pick among its unmeasured points. A coarser SBED grid (one
  point per FWHM) often puts only one or two points in the dip, too few for ``find_dips``, so the grid had to be
  refined before the focus appeared. ``SBED_COARSE_POINTS_PER_FWHM`` is kept separate so it can be changed
  (e.g. to 4) on its own.

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
# Candidate grid points per FWHM for SBED's pre-focus search (EIG picks among them).
SBED_COARSE_POINTS_PER_FWHM: int = 3

# Refusing to build grids denser than this: it means the refinement ran far past any useful resolution.
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

    ``min_hwhm_phys`` is the lower bound of the linewidth prior; the prior-mean FWHM never falls below
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


class RefiningGrid:
    """Nested uniform grids over the drive-frequency axis, refined by halving the spacing.

    Level ``j`` is a uniform grid with spacing ``<= base_spacing / 2**j`` that contains every point of level
    ``j - 1``. :meth:`next_unmeasured` returns the grid points inside a :class:`FocusRegion` that are not yet
    covered by a measurement at the current level, moving to the next level only once the current one is
    exhausted inside that region. Positions are Hz (physical); every array is ascending, shape (n_points,).
    """

    def __init__(self, lo_phys: float, hi_phys: float, base_spacing_phys: float) -> None:
        self._lo, self._hi = float(lo_phys), float(hi_phys)
        # Intervals at level 0; level j has m * 2**j intervals, so each level contains the one before it.
        self._base_intervals = len(uniform_grid_phys(self._lo, self._hi, float(base_spacing_phys))) - 1
        self._level = 0
        self._measured: list[float] = []
        self._grid = self._build_grid()

    @property
    def level(self) -> int:
        """Current refinement level (0 = the base spacing)."""
        return self._level

    @property
    def spacing_phys(self) -> float:
        """Actual spacing (Hz) of the current level's grid."""
        return float(self._grid[1] - self._grid[0])

    def _build_grid(self) -> NDArray[np.float64]:
        n_points = self._base_intervals * 2**self._level + 1
        if n_points > _MAX_GRID_POINTS:
            raise ValueError(f"RefiningGrid: level {self._level} needs {n_points} points (> {_MAX_GRID_POINTS}).")
        return np.linspace(self._lo, self._hi, n_points)

    def record(self, drive_freq_phys: float) -> None:
        """Note that ``drive_freq_phys`` (Hz) was measured."""
        bisect.insort(self._measured, float(drive_freq_phys))

    def _uncovered(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Mask over ``points`` (Hz): no measurement lies within half a spacing of the point."""
        if not self._measured:
            return np.ones(points.shape, dtype=bool)
        measured = np.asarray(self._measured, dtype=np.float64)
        idx = np.searchsorted(measured, points)
        left = np.abs(points - measured[np.clip(idx - 1, 0, len(measured) - 1)])
        right = np.abs(points - measured[np.clip(idx, 0, len(measured) - 1)])
        return np.minimum(left, right) > 0.5 * self.spacing_phys * (1.0 - 1e-9)

    def next_unmeasured(self, region: FocusRegion) -> NDArray[np.float64]:
        """Grid points inside ``region`` still uncovered, refining the grid until some exist.

        Returns:
            Ascending Hz positions. shape: (n_candidates,), never empty.
        """
        while True:
            in_region = self._grid[region.contains(self._grid)]
            candidates = in_region[self._uncovered(in_region)]
            if candidates.size:
                return candidates
            self._level += 1
            self._grid = self._build_grid()
