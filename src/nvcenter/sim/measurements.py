from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import polars as pl

from .core import DataBatch
from nvcenter.mathutils import safe_log


@dataclass
class FluorescenceCount:
    """Simple measurement: report mean and total signal."""

    def estimate(self, data: DataBatch) -> Dict[str, float]:
        # Use Polars to compute mean; keep total via count
        m = float(data.df.select(pl.col("signal_values").mean()).item()) if len(data.signal_values) > 0 else 0.0
        return {"mean": m, "total": m * len(data.signal_values)}


@dataclass
class RabiEstimate:
    """Naive Rabi parameter estimator.

    - amplitude: estimated as half the peak-to-peak.
    - frequency: estimated from zero-crossing period (basic approach).
    - offset: estimated as average.
    """

    def estimate(self, data: DataBatch) -> Dict[str, float]:
        y = data.signal_values
        t = data.time_points
        if len(y) < 2:
            return {"amplitude": 0.0, "frequency": 0.0, "offset": y[0] if y else 0.0}
        y_min = min(y)
        y_max = max(y)
        amp = 0.5 * (y_max - y_min)
        off = 0.5 * (y_max + y_min)
        # zero-crossings of (signal_values - offset)
        crossings: List[float] = []
        prev = y[0] - off
        for i in range(1, len(y)):
            cur = y[i] - off
            if prev == 0.0:
                crossings.append(t[i - 1])
            elif (prev < 0 and cur > 0) or (prev > 0 and cur < 0):
                # linear interpolation to estimate crossing time
                dt = t[i] - t[i - 1]
                if dt <= 0:
                    continue
                alpha = abs(prev) / (abs(prev) + abs(cur))
                tcross = t[i - 1] + alpha * dt
                crossings.append(tcross)
            prev = cur
        freq = 0.0
        if len(crossings) >= 3:
            # consecutive half-periods between alternating zero crossings
            periods = [2.0 * (crossings[i + 1] - crossings[i]) for i in range(len(crossings) - 1)]
            if periods:
                avg_period = sum(p for p in periods if p > 0) / max(len(periods), 1)
                if avg_period > 0:
                    freq = 1.0 / avg_period
        return {"amplitude": amp, "frequency": freq, "offset": off}


@dataclass
class T1Estimate:
    """Estimate tau and offset for an exponential decay signal_values = off + A*exp(-time_points/tau).

    Approach:
    - Estimate offset as the tail average (last 10%).
    - Subtract offset, keep positive values.
    - Fit log(signal_values) ~ a + b time_points (least squares); tau = -1/b; A = exp(a).
    """

    tail_frac: float = 0.1

    def estimate(self, data: DataBatch) -> Dict[str, float]:
        n = len(data.signal_values)
        if n == 0:
            return {"tau": 0.0, "A": 0.0, "offset": 0.0}
        k = max(1, int(self.tail_frac * n))
        off = sum(data.signal_values[-k:]) / k
        # prepare x, z = time_points, log(signal_values-off)
        x: List[float] = []
        z: List[float] = []
        for ti, yi in zip(data.time_points, data.signal_values):
            yi2 = yi - off
            if yi2 > 0:
                x.append(ti)
                z.append(safe_log(yi2))
        if len(x) < 2:
            return {"tau": 0.0, "A": 0.0, "offset": off}
        # least squares for z = a + b x
        sx = sum(x)
        sy = sum(z)
        sxx = sum(xi * xi for xi in x)
        sxy = sum(xi * zi for xi, zi in zip(x, z))
        npts = float(len(x))
        denom = npts * sxx - sx * sx
        if denom == 0:
            b = 0.0
            a = sy / npts
        else:
            b = (npts * sxy - sx * sy) / denom
            a = (sy - b * sx) / npts
        tau = -1.0 / b if b < 0 else 0.0
        A = math.exp(a) if tau > 0 else 0.0
        return {"tau": tau, "A": A, "offset": off}
