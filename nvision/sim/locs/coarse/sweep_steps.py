"""Utility for computing the default number of sweep steps from signal model properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nvision.sim.defaults import (
    NVISION_SWEEP_COVERAGE_FACTOR,
    NVISION_SWEEP_MAX_STEPS,
    NVISION_SWEEP_MIN_STEPS,
)

if TYPE_CHECKING:
    from nvision.spectra.signal import SignalModel


def compute_sweep_max_steps(
    signal_model: SignalModel,
    domain_lo: float,
    domain_hi: float,
    *,
    coverage_factor: float = NVISION_SWEEP_COVERAGE_FACTOR,
    min_steps: int = NVISION_SWEEP_MIN_STEPS,
    max_steps: int = NVISION_SWEEP_MAX_STEPS,
) -> int:
    """Calculate a reasonable sweep step count from signal model dip properties.

    The formula estimates how many uniformly-spaced points are needed across the
    full domain so that the narrowest expected dip receives at least
    ``coverage_factor`` samples.

    Parameters
    ----------
    signal_model : SignalModel
        The signal model (may be wrapped in UnitCubeSignalModel).
    domain_lo : float
        Lower bound of the domain (physical units).
    domain_hi : float
        Upper bound of the domain (physical units).
    coverage_factor : float, default 3.0
        Minimum samples that should fall inside the narrowest dip.
    min_steps : int, default 50
        Hard floor for step count.
    max_steps : int, default 500
        Hard ceiling for step count.

    Returns
    -------
    int
        Computed sweep step count clamped to ``[min_steps, max_steps]``.
    """
    domain_width = float(domain_hi - domain_lo)
    if domain_width <= 0:
        return max_steps

    # Unwrap UnitCubeSignalModel if needed
    inner_model = getattr(signal_model, "inner", signal_model)

    min_span = inner_model.signal_min_span(domain_width)
    max_span = inner_model.signal_max_span(domain_width)

    # Pick the narrowest meaningful dip width.
    # ``signal_min_span`` can return an extremely small lower-bound from the
    # parameter space (e.g. 0.02 % of the domain).  We clamp to a sensible
    # floor so we don't request tens of thousands of points.
    effective_span = min_span
    if effective_span is None or effective_span <= 0:
        effective_span = max_span
    if effective_span is None or effective_span <= 0:
        effective_span = domain_width * 0.05  # 5 % of domain fallback

    # Clamp to at least 0.2 % of the domain to avoid parameter-bound artefacts.
    effective_span = max(effective_span, domain_width * 0.002)

    # Minimum steps to resolve a dip of this width with ``coverage_factor`` samples.
    expected = (domain_width / effective_span) * coverage_factor

    # Clamp to reasonable bounds.
    return max(min_steps, min(int(expected), max_steps))
