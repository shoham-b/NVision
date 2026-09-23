"""The 7-dimensional constant space for Bayesian-SBED acquisition tuning.

Every optimizer in this package works internally on a ``[0, 1]^7`` unit cube and
maps to physical constant values through :func:`from_cube`. Two properties of
that mapping matter:

*   **Repair is part of the mapping, not a post-filter.** ``from_cube`` always
    returns a *feasible* point (probabilities ordered, ``min_obs`` integral,
    half-width no wider than the trigger). :func:`repair_cube` projects a cube
    point onto the cube point that actually produced the evaluated parameters,
    so the surrogate model is fitted on the coordinates that were really run.
    Fitting on the *requested* coordinates instead would make the model see a
    plateau of different inputs mapping to identical outputs and charge the
    difference to observation noise -- in a search whose entire difficulty is
    separating signal from noise, that is not a cosmetic issue.
*   **Scales are chosen per constant.** ``decay_tau`` and ``jitter_hz`` span
    more than an order of magnitude and are searched geometrically; the rest are
    linear. A uniform linear prior over ``jitter_hz in [0.5, 20] MHz`` would put
    ~75% of its mass above 5 MHz, which is a prior belief nobody stated.

Constraints and why they exist
------------------------------

``explore_p <= dip_p``
    The two probability constants gate consecutive branches of one
    ``rand_val`` draw: ``if rand_val < explore_p * decay: ... elif rand_val <
    dip_p:``. ``decay = exp(-step / tau) <= 1`` for every non-negative step, so
    ``explore_p <= dip_p`` is sufficient to keep the explore branch strictly
    inside the dip branch's interval at *every* step, not just at step 0.

``dual_halfwidth <= dual_trigger``
    The dual-window branch places two windows of half-width ``h * lw`` centred
    at ``center +/- split``. They overlap -- destroying the flank *balance* the
    branch exists to enforce -- as soon as ``h * lw >= split``. The branch only
    runs when ``split >= trigger * lw``, so ``h <= trigger`` guarantees
    ``h * lw <= split``. The rendered code additionally clamps the half-width to
    ``0.9 * split`` as a hard runtime backstop, but with this constraint in
    place that clamp is never the binding term at feasible points -- which is
    what keeps the rendered default byte-equivalent in behaviour to production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dimension:
    """One tunable constant: its bounds, its search scale and its baseline value."""

    name: str
    default: float
    lo: float
    hi: float
    log: bool = False
    integer: bool = False

    def to_value(self, u: float) -> float:
        """Map a unit-cube coordinate to a physical constant value."""
        u = min(max(float(u), 0.0), 1.0)
        value = self.lo * (self.hi / self.lo) ** u if self.log else self.lo + u * (self.hi - self.lo)
        if self.integer:
            value = float(round(value))
        return float(min(max(value, self.lo), self.hi))

    def to_unit(self, value: float) -> float:
        """Inverse of :meth:`to_value` (exact up to integer rounding)."""
        value = float(min(max(float(value), self.lo), self.hi))
        if self.log:
            u = math.log(value / self.lo) / math.log(self.hi / self.lo)
        else:
            span = self.hi - self.lo
            u = (value - self.lo) / span if span > 0 else 0.0
        return float(min(max(u, 0.0), 1.0))


# Bounds are deliberately generous rather than tight around the defaults: the
# point of the exercise is to find out whether the hand-picked values are near an
# optimum at all, and a box drawn snugly around them can only ever answer "yes".
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("decay_tau", 25.0, 5.0, 100.0, log=True),
    Dimension("explore_p", 0.1, 0.0, 0.5),
    Dimension("dip_p", 0.2, 0.02, 0.6),
    Dimension("min_obs", 5.0, 2.0, 30.0, integer=True),
    Dimension("jitter_hz", 5.0e6, 5.0e5, 2.0e7, log=True),
    Dimension("dual_trigger", 3.0, 0.5, 6.0),
    Dimension("dual_halfwidth", 3.0, 0.5, 6.0),
)

DIM_NAMES: tuple[str, ...] = tuple(d.name for d in DIMENSIONS)
N_DIMS = len(DIMENSIONS)
_BY_NAME = {d.name: d for d in DIMENSIONS}

BASELINE_PARAMS: dict[str, float] = {d.name: d.default for d in DIMENSIONS}


def repair_params(params: dict[str, float]) -> dict[str, float]:
    """Project a parameter dict onto the feasible set (see module docstring)."""
    out = dict(params)
    out["dip_p"] = max(out["dip_p"], out["explore_p"])
    out["dual_halfwidth"] = min(out["dual_halfwidth"], out["dual_trigger"])
    for dim in DIMENSIONS:
        value = float(min(max(out[dim.name], dim.lo), dim.hi))
        out[dim.name] = float(round(value)) if dim.integer else value
    return out


def from_cube(cube) -> dict[str, float]:
    """Unit-cube point -> feasible parameter dict."""
    cube = np.asarray(cube, dtype=float).ravel()
    if cube.size != N_DIMS:
        raise ValueError(f"expected {N_DIMS} coordinates, got {cube.size}")
    return repair_params({d.name: d.to_value(cube[i]) for i, d in enumerate(DIMENSIONS)})


def to_cube(params: dict[str, float]) -> np.ndarray:
    """Parameter dict -> unit-cube point."""
    return np.array([_BY_NAME[name].to_unit(params[name]) for name in DIM_NAMES], dtype=float)


def repair_cube(cube) -> np.ndarray:
    """The cube point that actually corresponds to what ``from_cube`` evaluates.

    Fit the surrogate on this, not on the raw proposal -- see module docstring.
    """
    return to_cube(from_cube(cube))


BASELINE_CUBE: np.ndarray = to_cube(BASELINE_PARAMS)


def format_params(params: dict[str, float]) -> str:
    """One-line human-readable rendering, with Hz shown in MHz."""
    parts = []
    for name in DIM_NAMES:
        value = params[name]
        if name == "jitter_hz":
            parts.append(f"jitter={value / 1e6:.2f}MHz")
        elif name == "min_obs":
            parts.append(f"min_obs={int(value)}")
        else:
            parts.append(f"{name}={value:.3g}")
    return " ".join(parts)
