from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class OverTimeDriftNoise:
    """Adds a slow linear drift across sequential measurements."""

    drift_per_unit: float = 0.05
    stateful: bool = True
    _current_drift: float | None = None

    def reset(self) -> None:
        """Reset accumulated drift for stateful usage."""
        self._current_drift = None

    def apply(self, signal_value: float, rng: random.Random) -> float:
        if self.stateful:
            if self._current_drift is None:
                self._current_drift = -0.5 * self.drift_per_unit
            else:
                self._current_drift += self.drift_per_unit
            return signal_value + self._current_drift

        span = self.drift_per_unit
        if span == 0:
            return signal_value
        offset = rng.uniform(-0.5 * span, 0.5 * span)
        return signal_value + offset
