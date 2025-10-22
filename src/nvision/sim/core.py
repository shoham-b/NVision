from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import polars as pl


@dataclass
class DataBatch:
    """Container for a single synthetic dataset.

    Attributes:
        time_points: time axis (seconds or arbitrary units)
        signal_values: signal values (e.g., photon counts or normalized intensity)
        meta: metadata such as true parameters for evaluation
        df: Polars DataFrame with columns ["t", "intensity"] for vectorized operations
    """

    time_points: list[float]
    signal_values: list[float]
    meta: dict[str, float]
    df: pl.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Build a Polars DataFrame for downstream vectorized work while keeping
        # the original list-backed attributes for backward compatibility.
        self.df = pl.DataFrame(
            {"time_points": self.time_points, "signal_values": self.signal_values},
        )

    def to_polars(self) -> pl.DataFrame:
        """Return the underlying Polars DataFrame (columns: time_points, signal_values)."""
        return self.df

    def with_y(self, new_y: list[float]) -> DataBatch:
        """
        Return a new DataBatch with the same time_points/meta and replaced
        signal_values, updating df accordingly.
        """
        nb = DataBatch(
            time_points=list(self.time_points),
            signal_values=list(new_y),
            meta=dict(self.meta),
        )
        return nb


class NoiseModel(Protocol):
    """Applies noise to a dataset and returns a new dataset (functional style)."""

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch: ...


class DataGenerator(Protocol):
    """Produces ideal (noise-free) signals and embeds ground-truth parameters in meta."""

    def generate(self, rng: random.Random) -> DataBatch: ...


class MeasurementStrategy(Protocol):
    """Consumes (noisy) data and returns estimated parameters."""

    def estimate(self, data: DataBatch) -> dict[str, float]: ...


class CompositeNoise:
    """Applies multiple noise models in sequence."""

    def __init__(self, parts: Sequence[NoiseModel] | None = None):
        self._parts: list[NoiseModel] = list(parts or [])

    def add(self, model: NoiseModel) -> None:
        self._parts.append(model)

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        out = data
        for part in self._parts:
            out = part.apply(out, rng)
        return out
