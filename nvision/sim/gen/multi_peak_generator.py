"""Multi-peak signal generator."""

from __future__ import annotations

import random
from dataclasses import dataclass

from nvision.spectra.composite import CompositeSpectrum

from .peak_spec import (
    GAUSSIAN,
    PeakSpec,
    _make_bounds,
    _make_model_and_spectrum,
    _true_signal_from_typed,
)


@dataclass
class MultiPeakCoreGenerator:
    """Generates multi-peak signals using core architecture.

    Creates a signal with N separate peaks.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    count: int = 3
    peak_configs: list[PeakSpec] | None = None  # per-peak specs; defaults to all Gaussian
    min_separation: float = 0.1  # Minimum separation as fraction of domain

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate multi-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with N peaks using CompositePeakModel
        """
        from nvision.spectra import CompositePeakModel

        width = self.x_max - self.x_min
        min_sep = self.min_separation * width

        # Resolve per-peak configs, padding with Gaussian when too few supplied.
        if self.peak_configs is None:
            configs: list[PeakSpec] = [GAUSSIAN] * self.count
        else:
            configs = list(self.peak_configs[: self.count])
            while len(configs) < self.count:
                configs.append(GAUSSIAN)

        # Generate well-separated peak positions
        positions: list[float] = []
        max_attempts = 1000
        for _ in range(self.count):
            for _ in range(max_attempts):
                pos = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
                if not positions or all(abs(pos - p) >= min_sep for p in positions):
                    positions.append(pos)
                    break
        positions.sort()

        # Create signal and typed parameters
        models = []
        typed_peak_params: list[object] = []
        bounds: dict[str, tuple[float, float]] = {}
        background = max(cfg.background_default for cfg in configs)

        for i, (pos, cfg) in enumerate(zip(positions, configs, strict=False)):
            prefix = f"peak{i + 1}"
            peak_width = rng.uniform(0.02 * width, 0.08 * width)
            model, spectrum = _make_model_and_spectrum(
                cfg, pos=pos, width=peak_width, dip_depth=1.0, background=background / self.count
            )
            typed_peak_params.append(spectrum)
            bounds.update(_make_bounds(cfg, self.x_min, self.x_max, f"{prefix}_", composite=True))
            models.append((prefix, model))

        composite_model = CompositePeakModel(models)
        typed_params = CompositeSpectrum(peaks=tuple(typed_peak_params))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)
