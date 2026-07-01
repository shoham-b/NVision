"""Shape-aware window narrowing for coarse sweep locators.

Used exclusively by the sweep stack (``SweepingLocator`` → ``GenericSweepLocator``,
``StagedSobolSweepLocator``) to discard irrelevant frequency space mid-sweep.

The approach is **geometric and model-free**: it looks for double-monotonic
regions in the raw (x, y) observations (down then up = a dip), infers each
dip's width from the signal shape, and computes a single tight bounding box
around all observed dips.  No particle filter, no statistics.

This is distinct from ``nvision.sim.locs.bayesian.dip_detection``, which is
used by SBED to assess posterior convergence.  The two detectors serve
different purposes and are not interchangeable:

- ``refocus.detect_dips``: geometric, returns raw ``(lo, hi)`` edge pairs.
  Used to answer "where should I keep scanning?"
- ``dip_detection.identify_dip_candidates``: statistical, uses per-particle
  noise sigmas and a binomial confidence test.  Returns ``DipCandidate``
  objects with confidence scores and background estimates.
  Used to answer "has the posterior converged to the right dip?"

Public API
----------
- :func:`detect_dips`: Find double-monotonic dips in (x, y) observations.
- :func:`infer_dip_widths`: Per-dip width inference using Strategies A/B.
- :func:`infer_focus_window`: Aggregate window from per-dip widths (Strategy C).
"""

from __future__ import annotations

from nvision.sim.locs.refocus.strategies import detect_dips, infer_dip_widths
from nvision.sim.locs.refocus.window import infer_focus_window, infer_focus_window_physical

__all__ = [
    "detect_dips",
    "infer_dip_widths",
    "infer_focus_window",
    "infer_focus_window_physical",
]
