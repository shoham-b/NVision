"""Symmetric two-peak signal generator."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from nvision.spectra.composite import CompositeSpectrum

from .peak_spec import (
    GAUSSIAN,
    PeakSpec,
    _make_bounds,
    _make_model_and_spectrum,
    _true_signal_from_typed,
)


@dataclass
class SymmetricTwoPeakCoreGenerator:
    """Generates symmetric two-peak signals using core architecture.

    Creates two peaks symmetrically placed around a center point.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    center: float = 0.5
    sep_frac: float = 0.2  # Separation as fraction of domain
    peak_config: PeakSpec = field(default_factory=lambda: GAUSSIAN)

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate symmetric two-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with two symmetric peaks
        """
        from nvision.spectra import CompositePeakModel

        width = self.x_max - self.x_min
        delta = 0.5 * self.sep_frac * width

        left_pos = self.center - delta
        right_pos = self.center + delta

        peak_width = rng.uniform(0.02 * width, 0.06 * width)
        background = self.peak_config.background_default
        model1, peak1_typed = _make_model_and_spectrum(
            self.peak_config, pos=left_pos, width=peak_width, dip_depth=1.0, background=background / 2
        )
        model2, peak2_typed = _make_model_and_spectrum(
            self.peak_config, pos=right_pos, width=peak_width, dip_depth=1.0, background=background / 2
        )
        composite_model = CompositePeakModel([("peak1", model1), ("peak2", model2)])
        bounds: dict[str, tuple[float, float]] = {}
        bounds.update(_make_bounds(self.peak_config, self.x_min, self.x_max, "peak1_", composite=True))
        bounds.update(_make_bounds(self.peak_config, self.x_min, self.x_max, "peak2_", composite=True))

        typed_params = CompositeSpectrum(peaks=(peak1_typed, peak2_typed))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)
