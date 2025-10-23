from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .obs import Obs


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
        k = max(2, min(5, len(history)))
        top = sorted(history, key=lambda o: o.intensity, reverse=True)[:k]
        mean = sum(o.x for o in top) / len(top)
        var = sum((o.x - mean) ** 2 for o in top) / len(top)
        s = var**0.5
        return {
            "n_peaks": 1.0,
            "x1_hat": float(best.x),
            "x2_hat": float("nan"),
            "uncert": float(s),
            "uncert_pos": float(s),
            "uncert_sep": float("nan"),
        }
