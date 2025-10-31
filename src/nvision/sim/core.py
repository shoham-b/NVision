from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import polars as pl

if TYPE_CHECKING:
    from .locators import ScanBatch


@dataclass(slots=True)
class DataBatch:
    """Container for a single synthetic dataset backed by a Polars DataFrame."""

    df: pl.DataFrame
    meta: dict[str, float]

    def __post_init__(self) -> None:
        expected_cols = {"time_points", "signal_values"}
        missing = expected_cols.difference(self.df.columns)
        if missing:
            raise ValueError(f"DataBatch.df is missing required columns: {sorted(missing)}")
        # Normalise column order and ensure we hold only the expected columns.
        self.df = self.df.select(["time_points", "signal_values"]).with_columns(
            pl.col("time_points").cast(pl.Float64),
            pl.col("signal_values").cast(pl.Float64),
        )
        self.meta = dict(self.meta)

    @classmethod
    def from_arrays(
        cls,
        time_points: Sequence[float],
        signal_values: Sequence[float],
        meta: dict[str, float] | None = None,
    ) -> DataBatch:
        df = pl.DataFrame(
            {
                "time_points": list(time_points),
                "signal_values": list(signal_values),
            }
        )
        return cls(df=df, meta=dict(meta or {}))

    @classmethod
    def from_frame(cls, df: pl.DataFrame, meta: dict[str, float]) -> DataBatch:
        return cls(df=df, meta=meta)

    @property
    def time_points(self) -> list[float]:
        return self.df.get_column("time_points").to_list()

    @property
    def signal_values(self) -> list[float]:
        return self.df.get_column("signal_values").to_list()

    def to_polars(self) -> pl.DataFrame:
        """Return the underlying Polars DataFrame (columns: time_points, signal_values)."""
        return self.df

    def with_y(self, new_y: Sequence[float]) -> DataBatch:
        df = self.df.with_columns(signal_values=pl.Series(list(new_y)))
        return DataBatch(df=df, meta=self.meta)


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


class ScanGenerator(Protocol):
    """Produces 1-D scan domains and ideal signal callable (for locators)."""

    def generate(self, rng: random.Random) -> ScanBatch: ...
