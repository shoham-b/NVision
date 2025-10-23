from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .obs import Obs


@dataclass
class BayesianLocator:
    max_evals: int = 30
    exploration_weight: float = 0.1
    uncertainty_threshold: float = 0.05

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        lo, hi = domain
        if len(history) < 2:
            import random
            return lo + random.random() * (hi - lo)
        # Very simple acquisition: pick midpoint of largest gap
        xs = sorted({o.x for o in history})
        gaps = [(xs[i + 1] - xs[i], 0.5 * (xs[i + 1] + xs[i])) for i in range(len(xs) - 1)]
        if not gaps:
            return 0.5 * (lo + hi)
        _, x = max(gaps)
        return x

    def should_stop(self, history: Sequence[Obs]) -> bool:
        return len(history) >= self.max_evals

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
        if not history:
            return {"n_peaks": 0.0, "x1": 0.0, "uncert": float("inf")}
        best = max(history, key=lambda o: o.intensity / max(o.uncertainty, 1e-6))
        return {"n_peaks": 1.0, "x1": best.x, "uncert": best.uncertainty}
