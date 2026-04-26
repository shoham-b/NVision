"""Composite noise containers used by CoreExperiment and ScalarMeasure."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nvision.sim.batch import DataBatch, OverFrequencyNoise, OverProbeNoise


class CompositeOverFrequencyNoise(OverFrequencyNoise):
    """Applies multiple over-frequency noise models in sequence."""

    def __init__(self, parts: Sequence[OverFrequencyNoise] | None = None):
        self._parts: list[OverFrequencyNoise] = list(parts or [])

    def add(self, model: OverFrequencyNoise) -> None:
        self._parts.append(model)

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        out = data
        for part in self._parts:
            out = part.apply(out, rng)
        return out

    def noise_std(self) -> float:
        """Return the combined RMS noise standard deviation for all over-frequency components."""
        rss = sum(p.noise_std() ** 2 for p in self._parts)
        std = rss**0.5
        return std

    def max_noise_deviation(self, n_samples: int = 20) -> float:
        """Expected maximum downward deviation across n_samples for all components.

        Continuous (Gaussian/Poisson) components combine in quadrature and their
        combined EVT maximum is computed.  Impulsive components (those whose own
        max_noise_deviation exceeds what the EVT of their std would predict) add
        their excess linearly on top, because spikes can coincide with continuous
        background fluctuations.
        """
        factor = math.sqrt(2.0 * math.log(max(n_samples, 2)))
        combined_std = math.sqrt(sum(p.noise_std() ** 2 for p in self._parts))
        continuous_max = combined_std * factor
        impulsive_excess = 0.0
        for p in self._parts:
            component_evt = p.noise_std() * factor
            component_max = p.max_noise_deviation(n_samples)
            if component_max > component_evt:
                impulsive_excess += component_max - component_evt
        return continuous_max + impulsive_excess

    def likelihood_spec(self) -> tuple[dict[str, Any], ...]:
        """Return a structured description of frequency-noise components.

        This metadata is attached to observations so Bayesian updaters can use
        component-specific likelihoods where supported.
        """
        specs: list[dict[str, Any]] = []
        for part in self._parts:
            name = part.__class__.__name__
            if name == "OverFrequencyGaussianNoise":
                specs.append({"type": "gaussian", "sigma": float(getattr(part, "sigma", 0.0))})
            elif name == "OverFrequencyPoissonNoise":
                specs.append({"type": "poisson", "scale": float(getattr(part, "scale", 0.0))})
            elif name == "OverFrequencyOutlierSpikes":
                specs.append(
                    {
                        "type": "outlier_spikes",
                        "probability": float(getattr(part, "probability", 0.0)),
                        "magnitude": float(getattr(part, "magnitude", 0.0)),
                    }
                )
            else:
                specs.append({"type": "unknown", "name": name})
        return tuple(specs)


class CompositeOverProbeNoise(OverProbeNoise):
    """Applies multiple over-probe noise models in sequence."""

    def __init__(self, parts: Sequence[OverProbeNoise] | None = None):
        self._parts: list[OverProbeNoise] = list(parts or [])

    def add(self, model: OverProbeNoise) -> None:
        self._parts.append(model)

    def apply(self, signal_value: float, rng: random.Random, locator: object = None) -> float:
        out = signal_value
        for part in self._parts:
            out = part.apply(out, rng, locator)
        return out


@dataclass(frozen=True, slots=True)
class CompositeNoise:
    """Container for both over-frequency and over-probe noise; applies both in sequence."""

    over_frequency_noise: CompositeOverFrequencyNoise | None = None
    over_probe_noise: CompositeOverProbeNoise | None = None

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        """Apply over-frequency noise (over-probe noise is not applicable to batch data)."""
        if self.over_frequency_noise is not None:
            return self.over_frequency_noise.apply(data, rng)
        return data

    def estimated_noise_std(self) -> float:
        """Return the combined RMS noise standard deviation for all over-frequency components.

        Returns
        -------
        float
            Square-root of summed squared stds from each noise component.
            Falls back to 0.05 if no noise components or all return 0.
        """
        if self.over_frequency_noise is None:
            return 0.05
        parts = getattr(self.over_frequency_noise, "_parts", [])
        rss = sum(p.noise_std() ** 2 for p in parts)
        std = rss**0.5
        return std if std > 1e-12 else 0.05

    def estimated_max_noise_deviation(self, n_samples: int = 20) -> float:
        """Expected maximum downward deviation across n_samples, from all noise components.

        Returns the value from the composite over-frequency noise model, or falls
        back to ``estimated_noise_std() × √(2·log(n_samples))`` when no model is set.
        """
        if self.over_frequency_noise is None:
            import math

            std = self.estimated_noise_std()
            return std * math.sqrt(2.0 * math.log(max(n_samples, 2)))
        return self.over_frequency_noise.max_noise_deviation(n_samples)
