"""NVision metrics package: per-run metrics computed identically for every locator.

Per-step series (:mod:`nvision.metrics.series`), the stop rule's convergence test (:mod:`nvision.metrics.milestones`),
dip detection, I_max and the axes (:mod:`nvision.metrics.run_axes`), and the article-claim analyses
(:mod:`nvision.metrics.claims`, :mod:`nvision.metrics.search_bounds`, :mod:`nvision.metrics.detection_aligned`,
:mod:`nvision.metrics.refine_rates`).
"""

from nvision.metrics.series import (
    effective_convergence_threshold,
    extract_step_series,
)

__all__ = [
    "effective_convergence_threshold",
    "extract_step_series",
]
