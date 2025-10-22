from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

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
    truth_positions: list[float]
    signal: Callable[[float], float]
    meta: dict[str, float] | None = None

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
    uncertainty: float = 0.0


class LocatorStrategy(Protocol):
    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float: ...

    def should_stop(self, history: Sequence[Obs]) -> bool: ...

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]: ...


# -----------------------------
# Baseline strategies
# -----------------------------
@dataclass
class GridScan:
    n_points: int = 21

    def _grid(self, lo: float, hi: float) -> list[float]:
        if self.n_points <= 1:
            return [0.5 * (lo + hi)]
        step = (hi - lo) / (self.n_points - 1)
        return [lo + i * step for i in range(self.n_points)]

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        lo, hi = domain
        grid = self._grid(lo, hi)
        taken = {round(o.x, 12) for o in history}
        for g in grid:
            if round(g, 12) not in taken:
                return g
        return grid[-1]

    def should_stop(self, history: Sequence[Obs]) -> bool:
        return len({round(o.x, 12) for o in history}) >= self.n_points

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
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

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        lo, hi = domain
        if not history:
            return 0.5 * (lo + hi)
        best = max(history, key=lambda o: o.intensity)
        if best.x < 0.5 * (lo + hi):
            return 0.5 * (best.x + hi)
        return 0.5 * (lo + best.x)

    def should_stop(self, history: Sequence[Obs]) -> bool:
        return len(history) >= self.max_evals

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
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

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
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

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
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
        picks: list[float] = []
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
# ODMR Locator Strategy
# -----------------------------
@dataclass
class ODMRLocator:
    """Optically Detected Magnetic Resonance locator using frequency sweep.
    
    This locator performs a coarse frequency sweep to identify resonance peaks,
    then refines around the most promising regions.
    """
    
    coarse_points: int = 20
    refine_points: int = 10
    min_separation_frac: float = 0.05  # minimum separation as fraction of domain width
    uncertainty_threshold: float = 0.1  # stop when uncertainty is below this threshold
    
    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        lo, hi = domain
        taken = {round(o.x, 12) for o in history}
        
        # Phase 1: Coarse sweep
        if len(history) < self.coarse_points:
            step = (hi - lo) / (self.coarse_points - 1)
            candidates = [lo + i * step for i in range(self.coarse_points)]
            for x in candidates:
                if round(x, 12) not in taken:
                    return x
        
        # Phase 2: Refine around promising regions
        if len(history) >= self.coarse_points:
            # Find regions with high intensity and low uncertainty
            promising_regions = self._find_promising_regions(history, domain)
            for region_center in promising_regions:
                # Propose points around this region
                region_width = (hi - lo) * 0.1  # 10% of domain width
                for offset in [-region_width/2, 0, region_width/2]:
                    x = region_center + offset
                    if lo <= x <= hi and round(x, 12) not in taken:
                        return x
        
        # Fallback: random point in domain
        import random
        return lo + random.random() * (hi - lo)
    
    def _find_promising_regions(self, history: Sequence[Obs], domain: tuple[float, float]) -> list[float]:
        """Find regions with high intensity and low uncertainty."""
        if not history:
            return []
        
        # Sort by intensity (descending) and uncertainty (ascending)
        sorted_obs = sorted(history, key=lambda o: (o.intensity, -o.uncertainty), reverse=True)
        
        # Take top 3 most promising points
        promising = sorted_obs[:3]
        return [obs.x for obs in promising]
    
    def should_stop(self, history: Sequence[Obs]) -> bool:
        if len(history) < self.coarse_points:
            return False
        
        # Stop if we have enough points and uncertainty is low
        if len(history) >= (self.coarse_points + self.refine_points):
            # Check if uncertainty is below threshold
            recent_obs = history[-5:] if len(history) >= 5 else history
            avg_uncertainty = sum(o.uncertainty for o in recent_obs) / len(recent_obs)
            return avg_uncertainty < self.uncertainty_threshold
        
        return len(history) >= (self.coarse_points + 2 * self.refine_points)
    
    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
        if not history:
            return {"n_peaks": 0.0, "x1": 0.0, "uncert": float("inf")}
        
        # Find peaks using intensity and uncertainty
        peaks = self._find_peaks(history)
        
        if len(peaks) == 0:
            return {"n_peaks": 0.0, "x1": 0.0, "uncert": float("inf")}
        elif len(peaks) == 1:
            peak = peaks[0]
            return {"n_peaks": 1.0, "x1": peak.x, "uncert": peak.uncertainty}
        else:
            # Multiple peaks - return the most confident one
            best_peak = max(peaks, key=lambda p: p.intensity / max(p.uncertainty, 1e-6))
            return {"n_peaks": 1.0, "x1": best_peak.x, "uncert": best_peak.uncertainty}
    
    def _find_peaks(self, history: Sequence[Obs]) -> list[Obs]:
        """Find peaks in the observation history."""
        if len(history) < 3:
            return []
        
        # Sort by x coordinate
        sorted_obs = sorted(history, key=lambda o: o.x)
        peaks = []
        
        for i in range(1, len(sorted_obs) - 1):
            prev_obs = sorted_obs[i-1]
            curr_obs = sorted_obs[i]
            next_obs = sorted_obs[i+1]
            
            # Check if this is a local maximum
            if (curr_obs.intensity > prev_obs.intensity and 
                curr_obs.intensity > next_obs.intensity):
                peaks.append(curr_obs)
        
        return peaks


# -----------------------------
# Bayesian Locator Strategy
# -----------------------------
@dataclass
class BayesianLocator:
    """Bayesian optimization locator using Gaussian Process.
    
    This locator uses Bayesian optimization to efficiently explore the parameter space
    and find optimal measurement points based on acquisition function.
    """
    
    max_evals: int = 30
    exploration_weight: float = 0.1  # balance between exploration and exploitation
    uncertainty_threshold: float = 0.05
    
    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        lo, hi = domain
        taken = {round(o.x, 12) for o in history}
        
        if len(history) < 2:
            # Initial random points
            import random
            return lo + random.random() * (hi - lo)
        
        # Use acquisition function to propose next point
        return self._acquisition_function(history, domain, taken)
    
    def _acquisition_function(self, history: Sequence[Obs], domain: tuple[float, float], taken: set[float]) -> float:
        """Propose next point using acquisition function."""
        lo, hi = domain
        
        # Simple acquisition function: balance between exploitation and exploration
        # For now, use a simplified version that looks for high uncertainty regions
        
        # Find regions with high uncertainty
        if len(history) >= 2:
            # Sort by x coordinate
            sorted_obs = sorted(history, key=lambda o: o.x)
            
            # Find gaps with high uncertainty
            max_uncertainty = 0.0
            best_x = lo + (hi - lo) / 2
            
            for i in range(len(sorted_obs) - 1):
                x1, x2 = sorted_obs[i].x, sorted_obs[i+1].x
                gap_center = (x1 + x2) / 2
                gap_width = x2 - x1
                
                # Prefer larger gaps
                if gap_width > (hi - lo) * 0.1:  # At least 10% of domain
                    # Calculate uncertainty in this gap
                    uncertainty = (sorted_obs[i].uncertainty + sorted_obs[i+1].uncertainty) / 2
                    if uncertainty > max_uncertainty:
                        max_uncertainty = uncertainty
                        best_x = gap_center
            
            # Check if this point is available
            if round(best_x, 12) not in taken:
                return best_x
        
        # Fallback: random point
        import random
        return lo + random.random() * (hi - lo)
    
    def should_stop(self, history: Sequence[Obs]) -> bool:
        if len(history) < 3:
            return False
        
        # Stop if we've reached max evaluations
        if len(history) >= self.max_evals:
            return True
        
        # Stop if uncertainty is below threshold
        recent_obs = history[-3:] if len(history) >= 3 else history
        avg_uncertainty = sum(o.uncertainty for o in recent_obs) / len(recent_obs)
        return avg_uncertainty < self.uncertainty_threshold
    
    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
        if not history:
            return {"n_peaks": 0.0, "x1": 0.0, "uncert": float("inf")}
        
        # Use Bayesian inference to find the most likely peak
        # For simplicity, use the observation with highest intensity/uncertainty ratio
        best_obs = max(history, key=lambda o: o.intensity / max(o.uncertainty, 1e-6))
        
        return {
            "n_peaks": 1.0,
            "x1": best_obs.x,
            "uncert": best_obs.uncertainty
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

    def run(self, rng: random.Random) -> tuple[pl.DataFrame, dict[str, float]]:
        lo, hi = self.scan.x_min, self.scan.x_max
        history: list[Obs] = []
        steps = 0
        while steps < self.max_steps and not self.strategy.should_stop(history):
            x = self.strategy.propose_next(history, (lo, hi))
            x = min(max(x, lo), hi)
            y_clean = float(self.scan.signal(x))
            y_noisy = self.meas.measure(x, y_clean, rng)
            
            # Calculate uncertainty based on noise model and measurement history
            uncertainty = self._calculate_uncertainty(x, y_noisy, history, lo, hi)
            history.append(Obs(x=x, intensity=y_noisy, uncertainty=uncertainty))
            steps += 1
        df = pl.DataFrame({"x": [o.x for o in history], "signal_values": [o.intensity for o in history]})
        est = self.strategy.finalize(history)
        return df, est
    
    def _calculate_uncertainty(self, x: float, y: float, history: list[Obs], lo: float, hi: float) -> float:
        """Calculate uncertainty for a measurement based on noise model and history."""
        # Base uncertainty from noise model
        base_uncertainty = 0.1  # Default uncertainty
        
        # If we have a noise model, estimate uncertainty from it
        if self.meas.noise is not None:
            # For now, use a simple estimate based on signal strength
            # In practice, this would depend on the specific noise model
            base_uncertainty = abs(y) * 0.05  # 5% of signal strength
        
        # Increase uncertainty in unexplored regions
        if len(history) < 3:
            # Early measurements have higher uncertainty
            return base_uncertainty * 2.0
        
        # Calculate local density of measurements
        local_density = self._calculate_local_density(x, history, lo, hi)
        
        # Uncertainty decreases with local density
        density_factor = max(0.1, 1.0 / (1.0 + local_density))
        
        return base_uncertainty * density_factor
    
    def _calculate_local_density(self, x: float, history: list[Obs], lo: float, hi: float) -> float:
        """Calculate local density of measurements around point x."""
        if not history:
            return 0.0
        
        # Define local region (10% of domain width)
        local_width = (hi - lo) * 0.1
        local_lo = max(lo, x - local_width/2)
        local_hi = min(hi, x + local_width/2)
        
        # Count measurements in local region
        local_count = sum(1 for obs in history if local_lo <= obs.x <= local_hi)
        
        return local_count / max(1, len(history))
