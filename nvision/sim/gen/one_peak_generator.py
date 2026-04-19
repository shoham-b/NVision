"""Single peak signal generator."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from nvision.spectra import CompositePeakModel
from nvision.spectra.composite import CompositeSpectrum

from .peak_spec import (
    GAUSSIAN,
    PeakSpec,
    _make_bounds,
    _make_model_and_spectrum,
    _true_signal_from_typed,
)


@dataclass
class OnePeakCoreGenerator:
    """Generates single-peak signals using core architecture."""

    x_min: float = 0.0
    x_max: float = 1.0
    peak_config: PeakSpec = field(default_factory=lambda: GAUSSIAN)

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate single-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with single peak
        """
        width = self.x_max - self.x_min
        pos = rng.uniform(self.x_min + 0.1 * width, self.x_max - 0.1 * width)
        peak_width = rng.uniform(0.02 * width, 0.08 * width)
        background = self.peak_config.background_default
        model, peak_typed = _make_model_and_spectrum(
            self.peak_config, pos=pos, width=peak_width, dip_depth=1.0, background=background
        )
        composite_model = CompositePeakModel([("peak1", model)])

        bounds = _make_bounds(self.peak_config, self.x_min, self.x_max, "peak1_", composite=True)

        typed_params = CompositeSpectrum(peaks=(peak_typed,))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)
