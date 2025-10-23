from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .obs import Obs


@dataclass
class TwoPeakGreedy:
    """Find two peaks by selecting the best two grid points with separation."""

    coarse_points: int = 25
    refine_points: int = 5
    min_separation_frac: float = 0.05

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
        import math

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
