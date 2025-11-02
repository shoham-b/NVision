from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import polars as pl

if TYPE_CHECKING:
    from .locators import ScanBatch


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
        # Normalise column order and ensure we hold only the expected columns.
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


class OverVoltageNoise(Protocol):
    """Applies noise to a dataset and returns a new dataset (functional style)."""

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch: ...


class OverTimeNoise(Protocol):
    """Applies noise to a single signal value, representing noise over time."""

    def apply(self, signal_value: float, rng: random.Random) -> float: ...


class DataGenerator(Protocol):
    """Produces ideal (noise-free) signals and embeds ground-truth parameters in meta."""

    def generate(self, rng: random.Random) -> DataBatch: ...


class MeasurementStrategy(Protocol):
    """Consumes (noisy) data and returns estimated parameters."""

    def estimate(self, data: DataBatch) -> dict[str, float]: ...


class CompositeOverVoltageNoise:
    """Applies multiple over-voltage noise models in sequence."""

    def __init__(self, parts: Sequence[OverVoltageNoise] | None = None):
        self._parts: list[OverVoltageNoise] = list(parts or [])

    def add(self, model: OverVoltageNoise) -> None:
        self._parts.append(model)

    def apply(self, data: DataBatch, rng: random.Random) -> DataBatch:
        out = data
        for part in self._parts:
            out = part.apply(out, rng)
        return out


class CompositeOverTimeNoise:
    """Applies multiple over-time noise models in sequence."""

    def __init__(self, parts: Sequence[OverTimeNoise] | None = None):
        self._parts: list[OverTimeNoise] = list(parts or [])

    def add(self, model: OverTimeNoise) -> None:
        self._parts.append(model)

    def apply(self, signal_value: float, rng: random.Random) -> float:
        out = signal_value
        for part in self._parts:
            out = part.apply(out, rng)
        return out


@dataclass(frozen=True, slots=True)
class CompositeNoise:
    """A container for both over-voltage and over-time noise models."""

    over_voltage_noise: CompositeOverVoltageNoise | None = None
    over_time_noise: CompositeOverTimeNoise | None = None


class ScanGenerator(Protocol):
    """Produces 1-D scan domains and ideal signal callable (for locators)."""

    def generate(self, rng: random.Random) -> ScanBatch: ...
