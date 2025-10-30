from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from nvision.sim.locs.models.obs import Obs
from nvision.sim.locs.models.protocols import LocatorStrategy
from nvision.sim.scalar_measure import ScalarMeasure
from nvision.sim.scan_batch import ScanBatch


@dataclass
class MeasurementProcess:
    """Orchestrates a simulated measurement process to locate a feature in a signal.

    This class acts as the main driver for a simulation run. It combines a signal
    source (`ScanBatch`), a measurement model (`ScalarMeasure`), and a search
    algorithm (`LocatorStrategy`) to iteratively sample a signal and estimate
    the location of a target feature (e.g., a peak or a dip).

    The process runs for a specified number of steps (`max_steps`) or until the
    strategy decides to stop. At each step, the strategy proposes a point to measure,
    this class simulates the measurement (including noise), and the result is fed
    back to the strategy.

    Attributes:
        scan (ScanBatch): The underlying signal and its domain.
        meas (ScalarMeasure): The model for performing a single noisy measurement.
        strategy (LocatorStrategy): The algorithm for proposing measurement points and
            finalizing the estimate.
        max_steps (int): The maximum number of measurement steps to perform.
    """

    scan: ScanBatch
    meas: ScalarMeasure
    strategy: LocatorStrategy
    max_steps: int = 200

    def run(self, rng: random.Random) -> tuple[pl.DataFrame, dict[str, float]]:
        self.strategy.set_scan(self.scan)
        lo, hi = self.scan.x_min, self.scan.x_max
        history: list[Obs] = []
        steps = 0
        while steps < self.max_steps and not self.strategy.should_stop(history):
            x = self.strategy.propose_next(history, (lo, hi))
            x = min(max(x, lo), hi)
            y_clean = float(self.scan.signal(x))
            y_noisy = self.meas.measure(x, y_clean, rng)
            uncertainty = self._calculate_uncertainty(x, y_noisy, history, lo, hi)
            history.append(Obs(x=x, intensity=y_noisy, uncertainty=uncertainty))
            steps += 1
        df = pl.DataFrame(
            {
                "x": [o.x for o in history],
                "signal_values": [o.intensity for o in history],
            },
        )
        est = self.strategy.finalize(history)
        return df, est

    def _calculate_uncertainty(
        self,
        x: float,
        y: float,
        history: list[Obs],
        lo: float,
        hi: float,
    ) -> float:
        base_uncertainty = 0.1
        if not history:
            return base_uncertainty * 2.0
        # Simple density-based heuristic
        local_width = (hi - lo) * 0.1
        local_lo = max(lo, x - local_width / 2)
        local_hi = min(hi, x + local_width / 2)
        local_count = sum(1 for obs in history if local_lo <= obs.x <= local_hi)
        density_factor = max(0.1, 1.0 / (1.0 + local_count))
        return base_uncertainty * density_factor
