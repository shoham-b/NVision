"""NVision metrics package: per-run metrics computed identically for every locator.

Per-step series (:mod:`nvision.metrics.series`), convergence milestones (:mod:`nvision.metrics.milestones`),
the search/focus/refine stage decomposition (:mod:`nvision.metrics.stages`) and the stage-by-stage
locator comparison table (:mod:`nvision.metrics.compare`).
"""

from nvision.metrics.series import (
    effective_convergence_threshold,
    extract_step_series,
)

__all__ = [
    "effective_convergence_threshold",
    "extract_step_series",
]
