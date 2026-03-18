"""Core data structures and noise base classes for the simulation layer."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl


@dataclass(slots=True, init=False)
class DataBatch:
    """Container for a single synthetic dataset backed by a Polars DataFrame."""

    df: pl.DataFrame
    meta: dict[str, float]

    def __init__(
        self,
        *,
        df: pl.DataFrame | None = None,
        x: Sequence[float] | None = None,
        signal_values: Sequence[float] | None = None,
        meta: Mapping[str, float] | None = None,
    ) -> None:
        if df is None:
            if x is None or signal_values is None:
                msg = "DataBatch requires either a DataFrame or x and signal_values."
                raise TypeError(msg)
            if len(x) != len(signal_values):
                msg = "x and signal_values must be the same length."
                raise ValueError(msg)
            df = pl.DataFrame({"x": list(x), "signal_values": list(signal_values)})
        else:
            if x is not None or signal_values is not None:
                msg = "Cannot pass both a DataFrame and x/signal_values to DataBatch."
                raise TypeError(msg)

        object.__setattr__(self, "df", df)
        object.__setattr__(self, "meta", dict(meta or {}))
        self.__post_init__()

    def __post_init__(self) -> None:
        expected_cols = {"x", "signal_values"}
        missing = expected_cols.difference(self.df.columns)
        if missing:
            raise ValueError(f"DataBatch.df is missing required columns: {sorted(missing)}")
        self.df = self.df.select(["x", "signal_values"]).with_columns(
            pl.col("x").cast(pl.Float64),
            pl.col("signal_values").cast(pl.Float64),
        )
        self.meta = dict(self.meta)

    @classmethod
    def from_arrays(
        cls,
        x: Sequence[float],
        signal_values: Sequence[float],
        meta: dict[str, float] | None = None,
    ) -> DataBatch:
        return cls(x=x, signal_values=signal_values, meta=meta)

    @classmethod
    def from_frame(cls, df: pl.DataFrame, meta: dict[str, float]) -> DataBatch:
        return cls(df=df, meta=meta)

    @property
    def x(self) -> list[float]:
        return self.df.get_column("x").to_list()

    @property
    def signal_values(self) -> list[float]:
        return self.df.get_column("signal_values").to_list()

    def to_polars(self) -> pl.DataFrame:
        """Return the underlying Polars DataFrame (columns: x, signal_values)."""
        return self.df

    def with_y(self, new_y: Sequence[float]) -> DataBatch:
        df = self.df.with_columns(signal_values=pl.Series(list(new_y)))
        return DataBatch(df=df, meta=self.meta)


class OverFrequencyNoise(ABC):
    """Base class for noise applied across all frequencies in a batch."""

    @abstractmethod
    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch: ...


class OverProbeNoise(ABC):
    """Base class for noise applied per-probe to a single signal value."""

    @abstractmethod
    def apply(self, signal_value: float, rng: random.Random, locator: object = None) -> float: ...
