"""Shape-aware window narrowing for coarse sweep locators.

Used by the sweep stack (``GenericSweepLocator``, ``StagedSobolSweepLocator``)
to discard irrelevant frequency space mid-sweep.

The approach is **geometric and model-free**: it looks for double-monotonic
regions in the raw (x, y) observations (down then up = a dip), infers each
dip's width from the signal shape, and computes a single tight bounding box
around all observed dips.  No particle filter, no statistics.

This is distinct from ``nvision.belief.dip_detection``, which SBED uses to find
the dips its observations show.  The two detectors serve different purposes and
are not interchangeable:

- ``refocus.detect_dips``: geometric, returns raw ``(lo, hi)`` edge pairs.
  Used to answer "where should I keep scanning?"
- ``dip_detection.find_dips``: statistical -- a noise-sigma threshold below the
  baseline plus a binomial confidence test, from the observations alone.
  Returns ``DipCandidate`` objects with confidence scores.
  Used to answer "where do the measurements show dips?"

Public API
----------
- :func:`detect_dips`: Find double-monotonic dips in (x, y) observations.
- :func:`infer_dip_widths`: Per-dip width inference using Strategies A/B.
- :func:`infer_focus_window`: Aggregate window from per-dip widths (Strategy C).
- :func:`infer_focus_window_physical`: ``infer_focus_window``, unit/physical
  coordinate agnostic.
- :func:`dip_noise_threshold`: Decide whether a sweep found a dip at all.
- :func:`infer_acquisition_window`: ``dip_noise_threshold`` +
  ``infer_focus_window_physical`` in one call; returns None if no dip found.
"""

from __future__ import annotations

from nvision.sim.locs.refocus.strategies import detect_dips, infer_dip_widths
from nvision.sim.locs.refocus.window import (
    dip_noise_threshold,
    infer_acquisition_window,
    infer_focus_window,
    infer_focus_window_physical,
)

__all__ = [
    "detect_dips",
    "dip_noise_threshold",
    "infer_acquisition_window",
    "infer_dip_widths",
    "infer_focus_window",
    "infer_focus_window_physical",
]
