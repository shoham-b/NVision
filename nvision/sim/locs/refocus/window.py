"""Aggregate window inference (Strategy C) from per-dip widths.

Given the maximum dip width inferred by Strategies A/B, the known number of
dips, and the observed dip positions, compute a single tight overall window.
"""

from __future__ import annotations

import numpy as np

from nvision.models.observation import ObservationHistory
from nvision.sim.locs.refocus.strategies import detect_dips, infer_dip_widths


def infer_focus_window(
    history: ObservationHistory,
    domain_lo: float,
    domain_hi: float,
    *,
    expected_dips: int = 1,
    signal_model: object | None = None,
    noise_threshold: float,
) -> tuple[float, float]:
    """Infer a tight overall window that contains all observed dips.

    Steps
    -----
    1. Detect dips in the full history using :func:`detect_dips`.
    2. Infer per-dip widths with :func:`infer_dip_widths`.
    3. Take the **smallest** detected width as the tightest upper bound
       ``max_dip_width``.
    4. Compute an aggregate window covering ``expected_dips * max_dip_width``
       plus observed inter-dip gaps, centred on the observed dip span.

    If detection fails, falls back to the full ``[domain_lo, domain_hi]``.

    Parameters
    ----------
    history :
        Sweep observation history.
    domain_lo, domain_hi :
        Physical bounds of the scan parameter.
    expected_dips :
        Number of dips expected from the signal model (e.g. 3 for NV centre).
    signal_model :
        Optional signal model used to read ``signal_max_span`` as a fallback.
    noise_threshold :
        Dip detection threshold.  Must be provided by the caller.

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)`` in physical units.
    """
    xs = history.xs
    ys = history.ys

    if len(xs) < 3:
        return domain_lo, domain_hi

    dips = detect_dips(xs, ys, noise_threshold=noise_threshold)
    if not dips:
        return domain_lo, domain_hi

    widths = infer_dip_widths(xs, ys, dips, noise_threshold=noise_threshold)
    if not widths:
        return domain_lo, domain_hi

    # Tightest upper bound: the smallest detected width (all true dips
    # share the same width in the signal model)
    max_dip_width = min(widths)

    # Strategy C: aggregate window from max_dip_width × expected_dips + gaps
    dip_centers = [(lo + hi) / 2.0 for lo, hi in dips]
    dip_centers.sort()

    # Observed span from first to last detected dip
    observed_span = dip_centers[-1] - dip_centers[0] if len(dip_centers) > 1 else max_dip_width

    # Inter-dip gaps (observed)
    gaps = [dip_centers[i + 1] - dip_centers[i] for i in range(len(dip_centers) - 1)]
    total_gap = sum(gaps)

    # Aggregate width: num_dips × max_width + gaps, with small padding
    window_width = expected_dips * max_dip_width + total_gap
    # Add a small fixed padding (5 % of window or 1 % of domain, whichever is larger)
    domain_width = domain_hi - domain_lo
    padding = max(0.05 * window_width, 0.01 * domain_width)
    window_width += 2.0 * padding

    # Centre on the observed dip span
    mid = (dip_centers[0] + dip_centers[-1]) / 2.0 if len(dip_centers) > 1 else dip_centers[0]
    lo = mid - window_width / 2.0
    hi = mid + window_width / 2.0

    # Clamp to domain
    lo = max(domain_lo, lo)
    hi = min(domain_hi, hi)

    # Sanity: don't expand if we already have a tighter window from the dips themselves
    dip_lo = min(lo for lo, _ in dips)
    dip_hi = max(hi for _, hi in dips)
    lo = min(lo, dip_lo - padding)
    hi = max(hi, dip_hi + padding)

    lo = max(domain_lo, lo)
    hi = min(domain_hi, hi)

    if lo >= hi:
        return domain_lo, domain_hi

    return lo, hi


def infer_max_dip_width(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    noise_threshold: float,
) -> float | None:
    """Return the smallest inferred dip width (tightest upper bound).

    Convenience wrapper used by StagedSobolSweepLocator Stage 2/3.
    """
    dips = detect_dips(xs, ys, noise_threshold=noise_threshold)
    if not dips:
        return None
    widths = infer_dip_widths(xs, ys, dips, noise_threshold=noise_threshold)
    return min(widths) if widths else None


def aggregate_window(
    dip_centers: list[float],
    max_dip_width: float,
    expected_dips: int,
    domain_lo: float,
    domain_hi: float,
) -> tuple[float, float]:
    """Build an aggregate window from dip centres and a known max width.

    Parameters
    ----------
    dip_centers :
        Sorted list of dip centre positions.
    max_dip_width :
        Tightest upper bound on individual dip width.
    expected_dips :
        Total number of dips expected.
    domain_lo, domain_hi :
        Hard physical bounds.

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)`` in physical units.
    """
    if not dip_centers:
        return domain_lo, domain_hi

    gaps = [dip_centers[i + 1] - dip_centers[i] for i in range(len(dip_centers) - 1)]
    total_gap = sum(gaps)

    domain_width = domain_hi - domain_lo
    window_width = expected_dips * max_dip_width + total_gap
    padding = max(0.05 * window_width, 0.01 * domain_width)
    window_width += 2.0 * padding

    mid = (dip_centers[0] + dip_centers[-1]) / 2.0 if len(dip_centers) > 1 else dip_centers[0]
    lo = mid - window_width / 2.0
    hi = mid + window_width / 2.0

    lo = max(domain_lo, lo)
    hi = min(domain_hi, hi)

    if lo >= hi:
        return domain_lo, domain_hi
    return lo, hi
