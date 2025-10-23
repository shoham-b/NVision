from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .obs import Obs


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
        import math

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
            if right != left:
                dx = 0.5 * (right - left)
            elif len(xs) > 1:
                dx = (xs[1] - xs[0]) / 2
            else:
                dx = 0.5
        return {
            "n_peaks": 1.0,
            "x1_hat": float(best.x),
            "x2_hat": float("nan"),
            "uncert": float(abs(dx)),
            "uncert_pos": float(abs(dx)),
            "uncert_sep": float("nan"),
        }
