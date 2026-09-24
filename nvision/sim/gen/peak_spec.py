"""Peak specification and helper functions for signal generators."""

from __future__ import annotations

from dataclasses import dataclass

from nvision.spectra.signal import TrueSignal


@dataclass(frozen=True)
class PeakSpec:
    """Static constants that describe a single peak type.

    Attributes:
        width_key: Parameter name for the width field (e.g. ``"sigma"``
            for Gaussian, ``"linewidth"`` for Lorentzian).
        width_frac: ``(lo, hi)`` width range expressed as *fractions of the
            domain width* (so ``0.01`` means 1 % of ``x_max - x_min``).
        dip_depth: ``(lo, hi)`` allowed range for the dip depth parameter.
        background: ``(lo, hi)`` allowed range for the background parameter
            when used as a *standalone* (non-composite) peak.
        composite_background: ``(lo, hi)`` range used when the peak is one
            component of a composite model and the baseline is shared.
        background_default: The fixed background value generated peaks use
            (0.0 for bump-up models, 1.0 for dip-down models).
    """

    width_key: str
    width_frac: tuple[float, float]
    dip_depth: tuple[float, float]
    background: tuple[float, float]
    composite_background: tuple[float, float]
    background_default: float
    max_span_frac: float
    """Maximum total signal span as a fraction of the domain width.

    For a single peak this is approximately ``4 × width_frac[1]`` (the signal
    is meaningful out to ~2× the characteristic width on each side).  This
    value is injected into parameter bounds as ``"_signal_max_span"`` so the
    locator can size the initial Sobol sweep and mid-sweep refocus window
    without re-deriving the span from individual parameter bounds.
    """


# Named singletons for the three supported peak types
GAUSSIAN = PeakSpec(
    width_key="sigma",
    width_frac=(0.01, 0.2),
    dip_depth=(0.1, 1.4),
    background=(0.0, 0.5),
    composite_background=(0.0, 0.5),
    background_default=0.0,
    max_span_frac=4 * 0.2,  # ±2σ at maximum linewidth
)

LORENTZIAN = PeakSpec(
    width_key="linewidth",
    width_frac=(0.01, 0.2),
    dip_depth=(0.05, 1.5),
    background=(0.5, 1.2),
    composite_background=(0.0, 0.5),
    background_default=1.0,
    max_span_frac=4 * 0.2,  # ±2×linewidth at maximum linewidth
)


def _true_signal_from_typed(model, typed_params, bounds: dict[str, tuple[float, float]]) -> TrueSignal:
    """Create a backward-compatible TrueSignal from typed model params."""
    return TrueSignal.from_typed(model=model, params=typed_params, bounds=bounds)
