"""Two-peak signal generator."""

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
class TwoPeakCoreGenerator:
    """Generates two-peak signals using core architecture.

    Creates a signal with two separate peaks by combining two single-peak signals.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    peak_config_left: PeakSpec = field(default_factory=lambda: GAUSSIAN)
    peak_config_right: PeakSpec = field(default_factory=lambda: GAUSSIAN)
    min_separation: float = 0.2  # Minimum separation as fraction of domain

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate two-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with two separated peaks using CompositePeakModel
        """
        from nvision.spectra import CompositePeakModel

        width = self.x_max - self.x_min
        min_sep = self.min_separation * width

        # Generate two well-separated peak positions
        peak1_pos = rng.uniform(self.x_min + 0.1 * width, self.x_max - 0.5 * width)
        peak2_pos = rng.uniform(peak1_pos + min_sep, self.x_max - 0.1 * width)

        # Random peak parameters
        peak1_width = rng.uniform(0.02 * width, 0.08 * width)
        peak2_width = rng.uniform(0.02 * width, 0.08 * width)
        background = max(
            self.peak_config_left.background_default,
            self.peak_config_right.background_default,
        )
        model1, peak1_typed = _make_model_and_spectrum(
            self.peak_config_left,
            pos=peak1_pos, width=peak1_width, dip_depth=1.0, background=background / 2,
        )
        model2, peak2_typed = _make_model_and_spectrum(
            self.peak_config_right,
            pos=peak2_pos, width=peak2_width, dip_depth=1.0, background=background / 2,
        )
        composite_model = CompositePeakModel([("peak1", model1), ("peak2", model2)])

        bounds: dict[str, tuple[float, float]] = {}
        bounds.update(_make_bounds(self.peak_config_left, self.x_min, self.x_max, "peak1_", composite=True))
        bounds.update(_make_bounds(self.peak_config_right, self.x_min, self.x_max, "peak2_", composite=True))

        typed_params = CompositeSpectrum(peaks=(peak1_typed, peak2_typed))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)
