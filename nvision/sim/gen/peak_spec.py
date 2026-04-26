"""Peak specification and helper functions for signal generators."""

from __future__ import annotations

from dataclasses import dataclass

from nvision.spectra import (
    GaussianModel,
    LorentzianModel,
)
from nvision.spectra.gaussian import GaussianSpectrum
from nvision.spectra.lorentzian import LorentzianSpectrum
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


def _make_bounds(
    spec: PeakSpec,
    x_min: float,
    x_max: float,
    prefix: str = "",
    *,
    composite: bool = False,
) -> dict[str, tuple[float, float]]:
    """Return parameter bounds for *spec* over the domain ``[x_min, x_max]``.

    Args:
        spec: Peak type descriptor.
        x_min: Lower domain bound.
        x_max: Upper domain bound.
        prefix: Key prefix for composite models (e.g. ``"peak1_"``).
        composite: When ``True``, use the narrower background range
            appropriate for composite signal components.
    """
    w = x_max - x_min
    bg = spec.composite_background if composite else spec.background
    bounds: dict[str, tuple[float, float]] = {
        f"{prefix}frequency": (x_min, x_max),
        f"{prefix}{spec.width_key}": (w * spec.width_frac[0], w * spec.width_frac[1]),
        f"{prefix}dip_depth": spec.dip_depth,
        f"{prefix}background": bg,
    }
    # Inject max signal span for sweep density and refocus window sizing.
    # Stored without prefix so the locator always finds it under "_signal_max_span".
    bounds["_signal_max_span"] = (0.0, w * spec.max_span_frac)
    return bounds


def _make_model_and_spectrum(
    spec: PeakSpec,
    *,
    pos: float,
    width: float,
    dip_depth: float,
    background: float,
) -> tuple[object, object]:
    """Instantiate ``(model, spectrum)`` for the given *spec*.

    Returns a ``(SignalModel, typed_spectrum)`` pair.
    """
    if spec == GAUSSIAN:
        return GaussianModel(), GaussianSpectrum(frequency=pos, sigma=width, dip_depth=dip_depth, background=background)
    if spec == LORENTZIAN:
        return LorentzianModel(), LorentzianSpectrum(
            frequency=pos, linewidth=width, dip_depth=dip_depth, background=background
        )
    raise ValueError(f"Unknown PeakSpec: {spec!r}")


def _true_signal_from_typed(model, typed_params, bounds: dict[str, tuple[float, float]]) -> TrueSignal:
    """Create a backward-compatible TrueSignal from typed model params."""
    return TrueSignal.from_typed(model=model, params=typed_params, bounds=bounds)
