from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from ...core import DataBatch


@dataclass
class OverTimeRandomWalkNoise:
    """Adds a cumulative random-walk offset along the measurement sequence.

    Models slow system changes that accumulate as more measurements are taken.
    The walk is independent of sampling interval; each point advances one step.
    """

    step_sigma: float = 0.02
    initial_offset: float = 0.0
    stateful: bool = True
    _offset: float | None = None

    def reset(self, offset: float | None = None) -> None:
        """Reset the internal state. If provided, set new starting offset."""
        self._offset = offset if offset is not None else self.initial_offset

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        n = data.df.height
        if n == 0:
            return data
        # Determine starting offset depending on stateful mode
        if self.stateful:
            if self._offset is None:
                self._offset = self.initial_offset
            acc = self._offset
        else:
            acc = self.initial_offset
        # Generate random-walk increments and cumulative sum
        series: list[float] = []
        for _ in range(n):
            acc += rng.gauss(0.0, self.step_sigma)
            series.append(acc)
        if self.stateful:
            self._offset = acc
        df = data.df.with_columns(
            signal_values=pl.col("signal_values") + pl.Series(series, dtype=pl.Float64)
        )
        return DataBatch.from_frame(df, meta=dict(data.meta))
