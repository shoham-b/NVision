from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .obs import Obs


@dataclass
class ODMRLocator:
    coarse_points: int = 20
    refine_points: int = 10
    min_separation_frac: float = 0.05
    uncertainty_threshold: float = 0.1

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        lo, hi = domain
        taken = {round(o.x, 12) for o in history}
        if len(history) < self.coarse_points:
            step = (hi - lo) / (self.coarse_points - 1)
            for i in range(self.coarse_points):
                x = lo + i * step
                if round(x, 12) not in taken:
                    return x
        if not history:
            return 0.5 * (lo + hi)
        # Promising region heuristic
        best = max(history, key=lambda o: o.intensity)
        width = (hi - lo) * 0.1
        for offset in (-width / 2, 0.0, width / 2):
            x = best.x + offset
            if lo <= x <= hi and round(x, 12) not in taken:
                return x
        return 0.5 * (lo + hi)

    def should_stop(self, history: Sequence[Obs]) -> bool:
        if len(history) >= (self.coarse_points + self.refine_points):
            recent = history[-5:] if len(history) >= 5 else history
            avg_u = sum(o.uncertainty for o in recent) / max(1, len(recent))
            return avg_u < self.uncertainty_threshold
        return False

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
        if not history:
            return {"n_peaks": 0.0, "x1": 0.0, "uncert": float("inf")}
        best = max(history, key=lambda o: o.intensity)
        return {"n_peaks": 1.0, "x1": best.x, "uncert": best.uncertainty}
