from __future__ import annotations

import random
from dataclasses import dataclass

from nvision.sim import OverProbeNoise


@dataclass
class OverProbeRandomWalkNoise(OverProbeNoise):
    """Adds a cumulative random-walk offset along sequential probes.

    Models slow system changes that accumulate as more measurements are taken.
    The walk is independent of sampling interval; each probe advances one step.
    """

    step_sigma: float = 0.02
    initial_offset: float = 0.0
    stateful: bool = True
    _offset: float | None = None

    def reset(self, offset: float | None = None) -> None:
        """Reset the internal state. If provided, set a new starting offset."""
        self._offset = offset if offset is not None else self.initial_offset

    def apply(self, signal_value: float, rng: random.Random, locator: object = None) -> float:
        # Interpret "drive drift" as a multiplicative change to the dip depth
        # (i.e., scale the deviation from the baseline), not as an additive
        # vertical offset. This avoids awkward constant shifts of the plot.
        baseline = 1.0

        if self.stateful:
            if self._offset is None:
                self._offset = self.initial_offset
            self._offset += rng.gauss(0.0, self.step_sigma)
            return signal_value + self._offset * (signal_value - baseline)

        offset = self.initial_offset + rng.gauss(0.0, self.step_sigma)
        return signal_value + offset * (signal_value - baseline)
