"""Belief builders — callables that construct a BeliefDistribution.

A builder is any callable with the signature::

    (parameter_bounds?, **grid_config) -> AbstractBeliefDistribution

No base class required.  The acquisition locators accept one and call it
at creation time.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from nvision.signal.grid_belief import GridBeliefDistribution, GridParameter


def _uniform_prior(
    name: str,
    bounds: tuple[float, float],
    overrides: Mapping[str, tuple[float, float]] | None,
    n_grid: int,
) -> GridParameter:
    lo, hi = bounds
    if overrides:
        inj = overrides.get(name)
        if inj is not None and inj[1] > inj[0]:
            lo, hi = float(inj[0]), float(inj[1])
    grid = np.linspace(lo, hi, n_grid)
    return GridParameter(
        name=name,
        bounds=(lo, hi),
        grid=grid,
        posterior=np.ones(n_grid) / n_grid,
    )


def nv_center_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_grid_freq: int = 120,
    n_grid_linewidth: int = 60,
    n_grid_split: int = 60,
    n_grid_k_np: int = 40,
    n_grid_amplitude: int = 40,
    n_grid_background: int = 40,
    **_extra: object,
) -> GridBeliefDistribution:
    """Build an NV-center Lorentzian GridBeliefDistribution with uniform priors."""
    from nvision.signal.nv_center import A_PARAM, MAX_K_NP, MIN_K_NP, NVCenterLorentzianModel

    model = NVCenterLorentzianModel()
    specs: list[tuple[str, tuple[float, float], int]] = [
        ("frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("linewidth", (1e6, 50e6), n_grid_linewidth),
        ("split", (1e6, 200e6), n_grid_split),
        ("k_np", (MIN_K_NP, MAX_K_NP), n_grid_k_np),
        ("amplitude", (A_PARAM * 0.5, A_PARAM * 2.0), n_grid_amplitude),
        ("background", (0.95, 1.05), n_grid_background),
    ]
    return GridBeliefDistribution(
        model=model,
        parameters=[_uniform_prior(name, bounds, parameter_bounds, n) for name, bounds, n in specs],
    )
