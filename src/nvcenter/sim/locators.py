from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple, Callable

import polars as pl

from .core import CompositeNoise, DataBatch


# -----------------------------
# Data container for 1-D scans
# -----------------------------
@dataclass
class ScanBatch:
    """Describes a 1-D domain where we will query intensities.

    Attributes:
        x_min, x_max: domain bounds (inclusive of endpoints)
        truth_positions: ground-truth positions (one or two T1/Rabi locations)
        signal: callable f(x) -> float providing the ideal intensity (no noise)
        meta: optional metadata (e.g., peak widths, amplitudes)
    """

    x_min: float
    x_max: float
    truth_positions: List[float]
    signal: Callable[[float], float]
    meta: Dict[str, float] | None = None

    def domain_width(self) -> float:
        return self.x_max - self.x_min


# -----------------------------
# Noise wiring for scalar measurements
# -----------------------------
@dataclass
class ScalarMeasure:
    """Applies existing CompositeNoise to a single scalar measurement.

    Wraps a scalar signal_values into a length-1 DataBatch so we can reuse NoiseModel implementations.
    """

    noise: CompositeNoise | None = None

    def measure(self, x: float, y_clean: float, rng: random.Random) -> float:
        if self.noise is None:
            return y_clean
        db = DataBatch(time_points=[x], signal_values=[y_clean], meta={})
        noisy = self.noise.apply(db, rng)
        return float(noisy.signal_values[0])


# -----------------------------
# Strategy protocol and history
# -----------------------------
@dataclass
class Obs:
    x: float
    intensity: float


class LocatorStrategy(Protocol):
    def propose_next(self, history: Sequence[Obs], domain: Tuple[float, float]) -> float: ...

    def should_stop(self, history: Sequence[Obs]) -> bool: ...

    def finalize(self, history: Sequence[Obs]) -> Dict[str, float]: ...


# -----------------------------
# Baseline strategies
# -----------------------------
@dataclass
class GridScan:
    n_points: int = 21

    def _grid(self, lo: float, hi: float) -> List[float]:
        if self.n_points <= 1:
            return [0.5 * (lo + hi)]
        step = (hi - lo) / (self.n_points - 1)
        return [lo + i * step for i in range(self.n_points)]

    def propose_next(self, history: Sequence[Obs], domain: Tuple[float, float]) -> float:
        lo, hi = domain
        grid = self._grid(lo, hi)
        taken = {round(o.x, 12) for o in history}
        for g in grid:
            if round(g, 12) not in taken:
                return g
        return grid[-1]

    def should_stop(self, history: Sequence[Obs]) -> bool:
        return len({round(o.x, 12) for o in history}) >= self.n_points

    def finalize(self, history: Sequence[Obs]) -> Dict[str, float]:
        if not history:
            return {
                "n_peaks": 1.0,
                "x1_hat": math.nan,
                "x2_hat": math.nan,
                "uncert": math.inf,
                "uncert_pos": math.inf,
                "uncert_sep": math.nan,
            }
        best = max(history, key=lambda o: o.intensity)
        xs = sorted({o.x for o in history})
        if len(xs) < 2:
            dx = 0.5
        else:
            idx = xs.index(min(xs, key=lambda xv: abs(xv - best.x)))
            left = xs[idx - 1] if idx > 0 else xs[idx]
            right = xs[idx + 1] if idx + 1 < len(xs) else xs[idx]
            dx = 0.5 * (right - left) if right != left else (xs[1] - xs[0]) / 2 if len(xs) > 1 else 0.5
        return {
            "n_peaks": 1.0,
            "x1_hat": float(best.x),
            "x2_hat": float("nan"),
            "uncert": float(abs(dx)),
            "uncert_pos": float(abs(dx)),
            "uncert_sep": float("nan"),
        }


@dataclass
class GoldenSectionSearch:
    max_evals: int = 20

    def propose_next(self, history: Sequence[Obs], domain: Tuple[float, float]) -> float:
        lo, hi = domain
        if not history:
            return 0.5 * (lo + hi)
        best = max(history, key=lambda o: o.intensity)
        if best.x < 0.5 * (lo + hi):
            return 0.5 * (best.x + hi)
        return 0.5 * (lo + best.x)

    def should_stop(self, history: Sequence[Obs]) -> bool:
        return len(history) >= self.max_evals

    def finalize(self, history: Sequence[Obs]) -> Dict[str, float]:
        if not history:
            return {
                "n_peaks": 1.0,
                "x1_hat": math.nan,
                "x2_hat": math.nan,
                "uncert": math.inf,
                "uncert_pos": math.inf,
                "uncert_sep": math.nan,
            }
        best = max(history, key=lambda o: o.intensity)
        k = max(2, min(5, len(history)))
        top = sorted(history, key=lambda o: o.intensity, reverse=True)[:k]
        mean = sum(o.x for o in top) / len(top)
        var = sum((o.x - mean) ** 2 for o in top) / len(top)
        s = math.sqrt(var)
        return {
            "n_peaks": 1.0,
            "x1_hat": float(best.x),
            "x2_hat": float("nan"),
            "uncert": float(s),
            "uncert_pos": float(s),
            "uncert_sep": float("nan"),
        }


@dataclass
class TwoPeakGreedy:
    """Find two peaks by selecting the best two grid points with separation.

    First runs a coarse grid, then refines around the two highest separated candidates.
    """

    coarse_points: int = 25
    refine_points: int = 5
    min_separation_frac: float = 0.05  # as fraction of domain width

    def propose_next(self, history: Sequence[Obs], domain: Tuple[float, float]) -> float:
        lo, hi = domain
        taken = {round(o.x, 12) for o in history}
        if len(taken) < self.coarse_points:
            if self.coarse_points <= 1:
                return 0.5 * (lo + hi)
            step = (hi - lo) / (self.coarse_points - 1)
            for i in range(self.coarse_points):
                g = lo + i * step
                if round(g, 12) not in taken:
                    return g
        if not history:
            return 0.5 * (lo + hi)
        xs = [o.x for o in history]
        ys = [o.intensity for o in history]
        idx_best = int(max(range(len(ys)), key=lambda i: ys[i]))
        x1 = xs[idx_best]
        w = (hi - lo) * self.min_separation_frac
        candidates = [i for i in range(len(xs)) if abs(xs[i] - x1) >= w]
        if candidates:
            idx_second = max(candidates, key=lambda i: ys[i])
            x2 = xs[idx_second]
        else:
            x2 = x1
        n1 = sum(1 for o in history if abs(o.x - x1) <= w)
        n2 = sum(1 for o in history if abs(o.x - x2) <= w)
        target = x2 if n2 < n1 else x1
        if target <= x1 and target <= x2:
            return max(lo, target - 0.5 * w)
        if target >= x1 and target >= x2:
            return min(hi, target + 0.5 * w)
        return target

    def should_stop(self, history: Sequence[Obs]) -> bool:
        return len(history) >= (self.coarse_points + 2 * self.refine_points)

    def finalize(self, history: Sequence[Obs]) -> Dict[str, float]:
        if not history:
            return {
                "n_peaks": 2.0,
                "x1_hat": math.nan,
                "x2_hat": math.nan,
                "uncert": math.inf,
                "uncert_pos": math.inf,
                "uncert_sep": math.inf,
            }
        hs = sorted(history, key=lambda o: o.intensity, reverse=True)
        picks: List[float] = []
        for o in hs:
            if not picks or all(abs(o.x - p) > 1e-9 for p in picks):
                picks.append(o.x)
            if len(picks) == 2:
                break
        picks.sort()
        if len(picks) == 1:
            return {
                "n_peaks": 2.0,
                "x1_hat": float(picks[0]),
                "x2_hat": float(picks[0]),
                "uncert": 0.0,
                "uncert_pos": 0.0,
                "uncert_sep": 0.0,
            }
        dist = abs(picks[1] - picks[0])
        return {
            "n_peaks": 2.0,
            "x1_hat": float(picks[0]),
            "x2_hat": float(picks[1]),
            "uncert": float(0.5 * dist),
            "uncert_pos": float(0.5 * dist),
            "uncert_sep": float(0.5 * dist),
        }


# -----------------------------
# Measurement process
# -----------------------------
@dataclass
class MeasurementProcess:
    scan: ScanBatch
    meas: ScalarMeasure
    strategy: LocatorStrategy
    max_steps: int = 200

    def run(self, rng: random.Random) -> Tuple[pl.DataFrame, Dict[str, float]]:
        lo, hi = self.scan.x_min, self.scan.x_max
        history: List[Obs] = []
        steps = 0
        while steps < self.max_steps and not self.strategy.should_stop(history):
            x = self.strategy.propose_next(history, (lo, hi))
            x = min(max(x, lo), hi)
            y_clean = float(self.scan.signal(x))
            y_noisy = self.meas.measure(x, y_clean, rng)
            history.append(Obs(x=x, intensity=y_noisy))
            steps += 1
        df = pl.DataFrame({"x": [o.x for o in history], "signal_values": [o.intensity for o in history]})
        est = self.strategy.finalize(history)
        return df, est
