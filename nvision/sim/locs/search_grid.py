"""Uniform drive-frequency grids sized from the resonance width, of the sweep locators.

When the dip's width is already known (a previously classified resonance whose position moved), the
grid spacing is not a tuning knob but follows from how many points must land inside the dip:

* A dip is only *seen* when several measurements fall inside it. With ``SWEEP_POINTS_PER_FWHM = 3``
  points per full width at half maximum (FWHM) the grid cannot step over a dip, and the dip's information
  about its position is within a few percent of a dense grid wherever the dip sits relative to the grid
  (:mod:`nvision.sim.locs.sweep_optimal`: worst/best offset 0.96/1.04 of the dense limit; detection energy is
  flat from 2 points per FWHM). The boxcar GLRT scan of :func:`nvision.belief.dip_detection.find_dips` then
  scores windows of the grid points inside a dip; it has no minimum cluster size of its own. 3 is *kept as the
  reference sweep*, not because it is optimal for a
  plain sweep: the plain-sweep baseline of the study runs at ``k*``, the grid step that needs the fewest
  measurements for its task (``SimpleSweep-optimal``, :func:`nvision.sim.locs.sweep_optimal.optimal_sweep_spacing`).

Widths come from the locator's *prior* (``2 * effective_hwhm(belief.estimates())`` at construction),
never from the true signal.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from nvision.spectra.nv_center import effective_hwhm

# Points per FWHM of the reference sweep (SimpleSweep-3perFWHM): a dip cannot be stepped over and the position
# information is within a few percent of a dense grid at any dip offset (``find_dips`` is a boxcar GLRT scan, it
# does not need a minimum cluster). Kept as the reference sweep; the plain-sweep
# baseline uses the optimal step k* instead (``nvision.sim.locs.sweep_optimal``).
SWEEP_POINTS_PER_FWHM: float = 3  # an int value on purpose: it is part of the sweep config (cache key)
# Refusing to build grids denser than this: it means the spacing is far below any useful resolution.
_MAX_GRID_POINTS: int = 2_000_000


def prior_mean_params_phys(belief) -> dict[str, float]:
    """Physical prior-mean value of every parameter of the belief (Hz for frequencies, else the parameter's unit).

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
    return means


def prior_fwhm_phys(belief) -> float:
    """FWHM (Hz) the belief's *prior* implies: ``2 * effective_hwhm`` of the prior-mean parameters
    (:func:`prior_mean_params_phys`)."""
    fwhm = 2.0 * float(effective_hwhm(prior_mean_params_phys(belief)))
    if not (fwhm > 0.0) or not math.isfinite(fwhm):
        raise ValueError(f"prior_fwhm_phys: the belief implies a non-positive or non-finite FWHM ({fwhm!r} Hz).")
    return fwhm


def max_sweep_points(lo_phys: float, hi_phys: float, min_hwhm_phys: float, points_per_fwhm: float) -> int:
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
