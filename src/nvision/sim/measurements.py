from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from nvision.mathutils import safe_log

from .core import DataBatch


@dataclass
class FluorescenceCount:
    """Simple measurement: report mean and total signal."""

    def estimate(self, data: DataBatch) -> dict[str, float]:
        # Use Polars to compute mean; keep total via count
        m = (
            float(data.df.select(pl.col("signal_values").mean()).item())
            if len(data.signal_values) > 0
            else 0.0
        )
        return {"mean": m, "total": m * len(data.signal_values)}


@dataclass
class RabiEstimate:
    """Naive Rabi parameter estimator.

    - amplitude: estimated as half the peak-to-peak.
    - frequency: estimated from zero-crossing period (basic approach).
    - offset: estimated as average.
    """

    def estimate(self, data: DataBatch) -> dict[str, float]:
        y = data.signal_values
        t = data.time_points
        if len(y) < 2:
            return {"amplitude": 0.0, "frequency": 0.0, "offset": y[0] if y else 0.0}
        y_min = min(y)
        y_max = max(y)
        amp = 0.5 * (y_max - y_min)
        off = 0.5 * (y_max + y_min)
        # zero-crossings of (signal_values - offset)
        crossings: list[float] = []
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

    def estimate(self, data: DataBatch) -> dict[str, float]:
        n = len(data.signal_values)
        if n == 0:
            return {"tau": 0.0, "A": 0.0, "offset": 0.0}

        # For T1 decay, the offset is the asymptotic value (minimum value)
        # Use the minimum value as the offset since exp(-t/tau) -> 0 as t -> inf
        off = min(data.signal_values)

        # prepare x, z = time_points, log(signal_values-off)
        x: list[float] = []
        z: list[float] = []
        for ti, yi in zip(data.time_points, data.signal_values):
            yi2 = yi - off
            if yi2 > 1e-10:  # Use a small threshold to avoid log(0)
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

        if abs(denom) < 1e-10:
            b = 0.0
            a = sy / npts
        else:
            b = (npts * sxy - sx * sy) / denom
            a = (sy - b * sx) / npts

        # tau = -1/b, A = exp(a)
        # Use a more robust tau estimation
        if b < -1e-10:
            tau = -1.0 / b
            A = math.exp(a)
        else:
            # Fallback: use a simple exponential fit on a subset of points
            # Take the first 30% of points for better tau estimation
            n_subset = max(2, int(0.3 * len(x)))
            x_subset = x[:n_subset]
            z_subset = z[:n_subset]

            if len(x_subset) >= 2:
                sx_sub = sum(x_subset)
                sy_sub = sum(z_subset)
                sxx_sub = sum(xi * xi for xi in x_subset)
                sxy_sub = sum(xi * zi for xi, zi in zip(x_subset, z_subset))
                npts_sub = float(len(x_subset))
                denom_sub = npts_sub * sxx_sub - sx_sub * sx_sub

                if abs(denom_sub) > 1e-10:
                    b_sub = (npts_sub * sxy_sub - sx_sub * sy_sub) / denom_sub
                    a_sub = (sy_sub - b_sub * sx_sub) / npts_sub
                    tau = -1.0 / b_sub if b_sub < -1e-10 else 0.0
                    A = math.exp(a_sub) if tau > 0 else 0.0
                else:
                    tau = 0.0
                    A = 0.0
            else:
                tau = 0.0
                A = 0.0

        return {"tau": tau, "A": A, "offset": off}
