from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nvision.sim.locs import Locator


@dataclass
class OverProbeDriftNoise:
    """Adds a slow linear drift across sequential probes."""

    drift_per_unit: float = 0.05
    stateful: bool = True
    _current_drift: float | None = None

    def reset(self) -> None:
        """Reset accumulated drift for stateful usage."""
        self._current_drift = None

    def apply(self, signal_value: float, rng: random.Random, locator: Locator) -> float:
        estimates = getattr(locator, "current_estimates", {})
        amplitude = estimates.get("amplitude", 1.0)
        drift = self.drift_per_unit * amplitude

        if self.stateful:
            if self._current_drift is None:
                self._current_drift = -0.5 * drift
            else:
                self._current_drift += drift
            return signal_value + self._current_drift

        span = drift
        if span == 0:
            return signal_value
        offset = rng.uniform(-0.5 * span, 0.5 * span)
        return signal_value + offset
