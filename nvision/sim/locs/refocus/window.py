"""Aggregate window inference (Strategy C) from per-dip widths.

Given the maximum dip width inferred by Strategies A/B, the known number of
dips, and the observed dip positions, compute a single tight overall window.
"""

from __future__ import annotations

import numpy as np

from nvision.models.observation import Observation, ObservationHistory
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

    If detection fails, raises ValueError so the caller knows the threshold
    or data is wrong rather than silently guessing a window.

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

    Raises
    ------
    ValueError
        If ``history`` has fewer than 3 points or no dips are detected at the
        given ``noise_threshold``.
    """
    xs = history.xs
    ys = history.ys

    if len(xs) < 3:
        raise ValueError(
            f"infer_focus_window needs at least 3 observations, got {len(xs)}"
        )

    dips = detect_dips(xs, ys, noise_threshold=noise_threshold)
    if not dips:
        raise ValueError(
            f"No dips detected with noise_threshold={noise_threshold} "
            f"on {len(xs)} observations (y range {float(np.min(ys)):.4f}..{float(np.max(ys)):.4f})"
        )

    widths = infer_dip_widths(xs, ys, dips, noise_threshold=noise_threshold)
    if not widths:
        raise ValueError(
            f"Dips were detected but infer_dip_widths returned no widths "
            f"on {len(xs)} observations"
        )

    # Tightest upper bound: the smallest detected width (all true dips
    # share the same width in the signal model)
    max_dip_width = min(widths)

    # Strategy C: aggregate window from max_dip_width × expected_dips + gaps
    dip_centers = [(lo + hi) / 2.0 for lo, hi in dips]
    dip_centers.sort()

    # Observed span from first to last detected dip
    dip_centers[-1] - dip_centers[0] if len(dip_centers) > 1 else max_dip_width
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

    # Ensure the window covers all detected dips plus padding (expand if needed)
    dip_lo = min(lo for lo, _ in dips)
    dip_hi = max(hi for _, hi in dips)
    lo = min(lo, dip_lo - padding)
    hi = max(hi, dip_hi + padding)

    # Also include any isolated deep points that fell below the threshold but
    # did not form a contiguous dip segment (e.g. a narrow dip hit by only 1-2
    # samples).  This prevents the focus window from missing dips that were
    # actually measured.
    deep_mask = ys < noise_threshold
    if np.any(deep_mask):
        deep_lo = float(np.min(xs[deep_mask]))
        deep_hi = float(np.max(xs[deep_mask]))
        lo = min(lo, deep_lo - padding)
        hi = max(hi, deep_hi + padding)

    lo = max(domain_lo, lo)
    hi = min(domain_hi, hi)

    if lo >= hi:
        raise ValueError(
            f"Inferred focus window collapsed (lo={lo:.6f} >= hi={hi:.6f}). "
            f"Dip span {dip_lo:.6f}..{dip_hi:.6f} with padding {padding:.6f} is too narrow."
        )

    return lo, hi


def infer_focus_window_physical(
    history: ObservationHistory,
    domain_lo: float,
    domain_hi: float,
    *,
    expected_dips: int = 1,
    signal_model: object | None = None,
    noise_threshold: float,
) -> tuple[float, float]:
    """Infer focus window in physical units.

    Works whether ``history.xs`` is already normalized ``[0,1]`` or physical.
    Internally normalises coordinates, runs :func:`infer_focus_window`, then
    denormalises the result.
    """
    xs = history.xs
    ys = history.ys
    domain_width = domain_hi - domain_lo

    if domain_width <= 0 or len(xs) < 3:
        return domain_lo, domain_hi

    # Detect whether history.xs is already normalized [0,1] (as returned by
    # experiment.measure) or physical (as returned by locator.next).
    # Real physical domains in this codebase are >> 1 Hz, so any xs span ≤ 1.5
    # with values inside [-0.5, 1.5] is treated as already normalized.
    xs_range = float(np.max(xs) - np.min(xs))
    xs_look_normalized = (
        xs_range <= 1.5
        and float(np.min(xs)) >= -0.5
        and float(np.max(xs)) <= 1.5
    )

    if xs_look_normalized:
        lo_norm, hi_norm = infer_focus_window(
            history,
            0.0,
            1.0,
            expected_dips=expected_dips,
            signal_model=signal_model,
            noise_threshold=noise_threshold,
        )
    else:
        # history.xs is physical; normalise to [0,1] before inference
        xs_norm = (xs - domain_lo) / domain_width
        temp_history = ObservationHistory(history.max_steps)
        for x_norm, y in zip(xs_norm, ys, strict=False):
            temp_history.append(Observation(x=float(x_norm), signal_value=float(y)))
        lo_norm, hi_norm = infer_focus_window(
            temp_history,
            0.0,
            1.0,
            expected_dips=expected_dips,
            signal_model=signal_model,
            noise_threshold=noise_threshold,
        )
    return (domain_lo + lo_norm * domain_width, domain_lo + hi_norm * domain_width)


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
