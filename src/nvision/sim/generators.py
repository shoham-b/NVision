from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import polars as pl

from .core import DataBatch
from .locators import ScanBatch


@dataclass
class RabiGenerator:
    """Generates a simple sinusoidal Rabi oscillation signal via a manufacturer.

    signal_values(time_points) = base + A * sin(2π f time_points + phi)
    """

    n_points: int = 200
    duration: float = 5.0
    base: float = 0.0
    trend: float = 0.0  # linear slope for baseline: y = base + trend * t
    centers: list[float] | None = None  # where to place spikes (in time units)
    manufacturers: list[SeriesManufacturer] | None = None
    manufacturer: SeriesManufacturer | None = None

    def __post_init__(self) -> None:
        if self.manufacturer is None:
            self.manufacturer = RabiManufacturer()

    def generate(self, rng: random.Random) -> DataBatch:
        dt = self.duration / max(self.n_points - 1, 1)
        t_series = (pl.arange(0, self.n_points, eager=True) * dt).alias("time_points")
        t_list: list[float] = pl.Series(t_series).to_list()
        # Flat baseline with optional trend
        y: list[float] = [self.base + self.trend * t for t in t_list]
        # Determine spike centers
        centers = self.centers
        if not centers:
            centers = [0.5 * self.duration]
        metas: list[dict[str, float]] = []
        # Overlay additions
        for i, c in enumerate(centers):
            manuf = (
                self.manufacturers[i]
                if self.manufacturers is not None and i < len(self.manufacturers)
                else self.manufacturer
            )
            assert manuf is not None
            add, meta = manuf.build_addition(t_list, c, self.base, rng)
            y = [yi + ai for yi, ai in zip(y, add, strict=False)]
            metas.append(meta)
        df = pl.DataFrame({"time_points": t_list, "signal_values": y})
        meta_out = {"base": self.base, "trend": self.trend, "n_spikes": len(centers)}
        for idx, m in enumerate(metas):
            meta_out.update({f"spike{idx}_{k}": float(v) for k, v in m.items()})
        return DataBatch.from_frame(df, meta_out)


@dataclass
class T1Generator:
    """Generates a simple exponential decay (T1-like) signal via a manufacturer.

    signal_values(time_points) = base + A * exp(-time_points / tau)
    """

    n_points: int = 200
    duration: float = 5.0
    base: float = 0.0
    manufacturer: SeriesManufacturer | None = None

    def __post_init__(self) -> None:
        if self.manufacturer is None:
            self.manufacturer = T1DecayManufacturer()

    def generate(self, rng: random.Random) -> DataBatch:
        dt = self.duration / max(self.n_points - 1, 1)
        t_series = (pl.arange(0, self.n_points, eager=True) * dt).alias("time_points")
        t_list = pl.Series(t_series).to_list()
        y, meta = self.manufacturer.build_series(t_list, self.base, rng)
        df = pl.DataFrame({"time_points": t_list, "signal_values": y})
        return DataBatch.from_frame(df, meta)


# -----------------------------
# 1-D scan generators (for locators)
# -----------------------------


class PeakManufacturer(Protocol):
    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]: ...


class SeriesManufacturer(Protocol):
    def build_addition(
        self,
        time_points: list[float],
        center: float,
        base: float,
        rng: random.Random,
    ) -> tuple[list[float], dict[str, float]]: ...


@dataclass
class OnePeakGenerator:
    """Generates a single-peaked 1D signal around a hidden location x0.

    Modes:
      - gaussian (default): base + A * exp(-0.5 * ((x - x0)/sigma)^2)
      - rabi: base + A * exp(-0.5 * ((x - x0)/sigma)^2) * 0.5 * (1 + sin(2π f (x - x0) + phi))
      - t1_decay: base + A * exp(-|x - x0| / tau)

    The hidden position x0 is randomized within [x_min, x_max]. This lets the
    locator strategies search for the location while experiencing either a
    Rabi-like oscillatory envelope or a T1-like decay envelope.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    base: float = 0.0
    manufacturer: PeakManufacturer | None = None

    def __post_init__(self) -> None:
        # manufacturer must be provided explicitly (no fallback)
        if self.manufacturer is None:
            raise ValueError("OnePeakGenerator requires a manufacturer (e.g., gaussian_peak)")

    def generate(self, rng: random.Random) -> ScanBatch:
        width = self.x_max - self.x_min
        x0 = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
        f, extra_meta = self.manufacturer.build_peak(x0, self.base, self.x_min, self.x_max, rng)
        meta = {"base": self.base, **extra_meta}

        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=[x0],
            signal=f,
            meta=meta,
        )

    # no legacy mode/factory retained; strictly manufacturer-based


@dataclass
class MultiPeakGenerator:
    x_min: float = 0.0
    x_max: float = 1.0
    count: int = 1
    base: float = 0.0
    min_sep_frac: float = 0.1
    # Manufacturers to use per peak (length == count). If one manufacturer is provided,
    # it will be reused for all peaks.
    manufacturers: list[PeakManufacturer] | None = None
    manufacturer: PeakManufacturer | None = None

    def generate(self, rng: random.Random) -> ScanBatch:
        # Require manufacturers
        if self.manufacturers is None and self.manufacturer is None:
            raise ValueError("MultiPeakGenerator requires manufacturer(s)")
        width = self.x_max - self.x_min
        xs: list[float] = []
        # Sample centers ensuring min separation
        while len(xs) < max(1, self.count):
            x = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
            if not xs or all(abs(x - xi) >= self.min_sep_frac * width for xi in xs):
                xs.append(x)
        # Build per-peak functions
        fns: list[Callable[[float], float]] = []
        metas: list[dict[str, float]] = []
        for i, xc in enumerate(xs):
            manuf = (
                self.manufacturers[i]
                if self.manufacturers is not None and i < len(self.manufacturers)
                else self.manufacturer
            )
            if manuf is None:
                raise ValueError("Missing manufacturer for peak index {i}")
            f_i, meta_i = manuf.build_peak(xc, self.base, self.x_min, self.x_max, rng)
            fns.append(f_i)
            metas.append(meta_i)

        def f(x: float) -> float:
            return sum(fi(x) for fi in fns) - (len(fns) - 1) * self.base

        xs_sorted = sorted(xs)
        meta_agg: dict[str, float] = {"base": self.base}
        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=xs_sorted,
            signal=f,
            meta=meta_agg,
        )


@dataclass
class TwoPeakGenerator:
    """Generates a two-peak 1D signal as a sum of Gaussians at two random locations."""

    x_min: float = 0.0
    x_max: float = 1.0
    base: float = 0.0
    min_sep_frac: float = 0.1  # enforce minimum separation between peaks
    # Manufacturers for the two peaks
    manufacturer_left: PeakManufacturer | None = None
    manufacturer_right: PeakManufacturer | None = None

    def generate(self, rng: random.Random) -> ScanBatch:
        if self.manufacturer_left is None or self.manufacturer_right is None:
            raise ValueError("TwoPeakGenerator requires manufacturer_left and manufacturer_right")
        mg = MultiPeakGenerator(
            x_min=self.x_min,
            x_max=self.x_max,
            count=2,
            base=self.base,
            min_sep_frac=self.min_sep_frac,
            manufacturers=[self.manufacturer_left, self.manufacturer_right],
        )
        return mg.generate(rng)


@dataclass
class SymmetricTwoPeakGenerator:
    """Generates two symmetric peaks around a specified center value.

    The two peak centers are at `center - delta` and `center + delta`, where
    `delta = 0.5 * sep_frac * (x_max - x_min)`. The peaks use the provided
    `manufacturer` (e.g., `gaussian_peak`, `rabi_peak`).
    """

    x_min: float = 0.0
    x_max: float = 1.0
    center: float = 0.5
    sep_frac: float = 0.1
    base: float = 0.0
    manufacturers: tuple[PeakManufacturer, PeakManufacturer] | None = None

    def __post_init__(self) -> None:
        if self.manufacturers is None:
            raise ValueError("SymmetricTwoPeakGenerator requires manufacturers=(left, right)")
        if not (self.x_min <= self.center <= self.x_max):
            raise ValueError("center must lie within [x_min, x_max]")
        if self.sep_frac <= 0.0:
            raise ValueError("sep_frac must be positive")

    def generate(self, rng: random.Random) -> ScanBatch:
        width = self.x_max - self.x_min
        delta = 0.5 * self.sep_frac * width
        max_delta = min(self.center - self.x_min, self.x_max - self.center)
        if delta > max_delta:
            raise ValueError("sep_frac too large for given center and domain")
        x1 = self.center - delta
        x2 = self.center + delta

        # Build each peak using explicit manufacturers passed via manufacturers
        manuf_left, manuf_right = self.manufacturers
        f1, _m1 = manuf_left.build_peak(x1, self.base, self.x_min, self.x_max, rng)
        f2, _m2 = manuf_right.build_peak(x2, self.base, self.x_min, self.x_max, rng)

        def f(x: float) -> float:
            return f1(x) + f2(x) - self.base

        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=[x1, x2],
            signal=f,
            meta={"center": self.center, "sep_frac": self.sep_frac, "base": self.base},
        )


class GaussianManufacturer:
    """Gaussian-shaped OnePeak manufacturer."""

    def __init__(self, amplitude: float = 1.0, sigma: float = 0.08, **kwargs) -> None:
        # Back-compat alias
        if "A" in kwargs:
            amplitude = kwargs["A"]
        self.amplitude = amplitude
        self.sigma = sigma

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        def f(x: float) -> float:
            z = (x - center) / max(self.sigma, 1e-12)
            return base + self.amplitude * math.exp(-0.5 * z * z)
        return f, {"sigma": self.sigma, "amplitude": self.amplitude, "mode": "gaussian"}

    def build_addition(
        self, time_points: list[float], center: float, base: float, rng: random.Random,
    ) -> tuple[list[float], dict[str, float]]:
        if not time_points:
            return [], {"amplitude": self.amplitude, "sigma": self.sigma, "mode": "gaussian"}
        y = [
            self.amplitude
            * math.exp(-0.5 * ((t - center) / max(self.sigma, 1e-12)) ** 2)
            for t in time_points
        ]
        return y, {"amplitude": self.amplitude, "sigma": self.sigma, "mode": "gaussian"}


class RabiManufacturer:
    """Rabi-shaped OnePeak manufacturer (randomized phase if unset)."""

    def __init__(
        self,
        amplitude: float = 1.0,
        sigma: float = 0.08,
        rabi_freq: float = 5.0,
        rabi_phase: float | None = None,
        **kwargs,
    ) -> None:
        if "A" in kwargs:
            amplitude = kwargs["A"]
        self.amplitude = amplitude
        self.sigma = sigma
        self.rabi_freq = rabi_freq
        self.rabi_phase = rabi_phase

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        phi = self.rabi_phase if self.rabi_phase is not None else rng.uniform(0.0, 2 * math.pi)

        def f(x: float) -> float:
            z = (x - center) / max(self.sigma, 1e-12)
            env = math.exp(-0.5 * z * z)
            osc = 0.5 * (1.0 + math.sin(2 * math.pi * self.rabi_freq * (x - center) + phi))
            return base + self.amplitude * env * osc

        return f, {
            "sigma": self.sigma,
            "amplitude": self.amplitude,
            "mode": "rabi",
            "rabi_freq": self.rabi_freq,
            "rabi_phase": float(phi),
        }

    def build_addition(
        self, time_points: list[float], center: float, base: float, rng: random.Random,
    ) -> tuple[list[float], dict[str, float]]:
        phi = self.rabi_phase if self.rabi_phase is not None else rng.uniform(0.0, 2 * math.pi)
        y = [
            self.amplitude
            * math.exp(-0.5 * ((t - center) / max(self.sigma, 1e-12)) ** 2)
            * 0.5
            * (1.0 + math.sin(2 * math.pi * self.rabi_freq * (t - center) + phi))
            for t in time_points
        ]
        return y, {
            "amplitude": self.amplitude,
            "sigma": self.sigma,
            "rabi_freq": self.rabi_freq,
            "rabi_phase": float(phi),
        }


class T1DecayManufacturer:
    """T1-decay-shaped OnePeak manufacturer."""

    def __init__(self, amplitude: float = 1.0, t1_tau: float | None = None, **kwargs) -> None:
        if "A" in kwargs:
            amplitude = kwargs["A"]
        self.amplitude = amplitude
        self.t1_tau = t1_tau

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        tau = self.t1_tau if self.t1_tau is not None else max(0.05 * (x_max - x_min), 1e-6)

        def f(x: float) -> float:
            return base + self.amplitude * math.exp(-abs(x - center) / max(tau, 1e-12))

        return f, {"tau": float(tau), "amplitude": self.amplitude, "mode": "t1_decay"}

    def build_addition(
        self, time_points: list[float], center: float, base: float, rng: random.Random,
    ) -> tuple[list[float], dict[str, float]]:
        if not time_points:
            return [], {
                "amplitude": self.amplitude,
                "mode": "t1_decay",
                "tau": float(self.t1_tau or 0.0),
            }
        tau = self.t1_tau
        if tau is None:
            # Estimate a reasonable tau based on timescale
            span = max(time_points[-1] - time_points[0], 1e-6)
            tau = max(0.05 * span, 1e-6)
        y = [self.amplitude * math.exp(-abs(t - center) / max(tau, 1e-12)) for t in time_points]
        return y, {"amplitude": self.amplitude, "tau": float(tau), "mode": "t1_decay"}


# Generator factories (return generator instances wired with a peak manufacturer)
def gaussian(
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
    base: float = 0.0,
    amplitude: float = 1.0,
    sigma: float = 0.08,
) -> OnePeakGenerator:
    return OnePeakGenerator(
        x_min=x_min,
        x_max=x_max,
        base=base,
        manufacturer=GaussianManufacturer(amplitude=amplitude, sigma=sigma),
    )

def rabi(
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
    base: float = 0.0,
    amplitude: float = 1.0,
    sigma: float = 0.08,
    rabi_freq: float = 5.0,
    rabi_phase: float | None = None,
) -> OnePeakGenerator:
    return OnePeakGenerator(
        x_min=x_min,
        x_max=x_max,
        base=base,
        manufacturer=RabiManufacturer(
            amplitude=amplitude, sigma=sigma, rabi_freq=rabi_freq, rabi_phase=rabi_phase,
        ),
    )

def t1_decay(
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
    base: float = 0.0,
    amplitude: float = 1.0,
    t1_tau: float | None = None,
) -> OnePeakGenerator:
    return OnePeakGenerator(
        x_min=x_min,
        x_max=x_max,
        base=base,
        manufacturer=T1DecayManufacturer(amplitude=amplitude, t1_tau=t1_tau),
    )


# no string-spec manufacturer; use explicit factories



