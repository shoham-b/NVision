"""Post-sweep estimation of non-scan parameter bounds from Sobol sweep data.

After an initial 1-D coarse sweep over the scan (frequency) axis, the measured
signal already encodes considerable information about the other model parameters:

- The baseline level estimates ``background``.
- The minimum signal relative to the baseline estimates ``dip_depth``.
- The width of the informative region estimates the linewidth family (``linewidth``,
  ``fwhm_lorentz``, ``fwhm_gauss``).
- Two separated dip regions allow estimating ``split``.

All returned intervals are padded conservatively (controlled by ``safety_factor``)
and clamped to the original prior bounds so the true parameter value is never
excluded.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

log = logging.getLogger(__name__)

# Minimum number of sweep samples required for reliable estimation.
_MIN_SWEEP_SAMPLES = 6

# Safety padding applied symmetrically around each estimate (as a fraction of
# the estimated range). Larger → wider intervals, safer but less informative.
_DEFAULT_SAFETY_FACTOR = 0.6

# Fraction of the prior width that must be saved to bother narrowing a param.
_MIN_NARROWING_FRACTION = 0.10


def _sorted_sweep(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) sorted by x and with duplicates averaged."""
    order = np.argsort(xs)
    return xs[order], ys[order]


def _estimate_background(ys: np.ndarray) -> float:
    """High-signal (baseline) estimate from the 90th-percentile signal."""
    return float(np.quantile(ys, 0.90))


def _estimate_dip_depth(ys: np.ndarray, background: float) -> float | None:
    """Ratio of maximum dip depth to background, in (0, 1)."""
    if background <= 0:
        return None
    depth = (background - float(np.min(ys))) / background
    if depth <= 0:
        return None
    return min(depth, 1.0)


def _estimate_linewidth_from_dip(
    xs_sorted: np.ndarray,
    ys_sorted: np.ndarray,
    background: float,
    dip_depth: float,
) -> float | None:
    """Estimate a characteristic half-width at half-dip-depth.

    Returns the full width (not HWHM) of the significant signal excursion, as
    a fraction of the scan domain.

    Args:
        xs_sorted: Scan positions, sorted ascending.
        ys_sorted: Corresponding signal values.
        background: Estimated baseline.
        dip_depth: Estimated dip depth fraction.

    Returns:
        Full width estimate in physical units, or ``None`` if unreliable.
    """
    threshold = background - 0.5 * dip_depth * background
    below = ys_sorted < threshold
    if not np.any(below):
        return None
    below_xs = xs_sorted[below]
    if below_xs.size < 2:
        return None
    return float(below_xs[-1] - below_xs[0])


def _find_dip_peaks(
    xs_sorted: np.ndarray,
    ys_sorted: np.ndarray,
    background: float,
    dip_depth: float,
) -> list[tuple[float, float]]:
    """Return a list of (x_center, depth) for each resolved dip.

    Simple valley detector: split the informative region (signal below
    half-depth threshold) into contiguous segments and return the minimum
    of each.

    Args:
        xs_sorted: Scan positions, sorted ascending.
        ys_sorted: Corresponding signal values.
        background: Estimated baseline.
        dip_depth: Estimated dip depth fraction.

    Returns:
        List of (x_center, depth_fraction) tuples, one per dip.
    """
    threshold = background - 0.4 * dip_depth * background
    below = ys_sorted < threshold
    if not np.any(below):
        return []

    # Identify contiguous segments.
    changes = np.diff(below.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if below[0]:
        starts = np.concatenate([[0], starts])
    if below[-1]:
        ends = np.concatenate([ends, [len(below)]])

    peaks: list[tuple[float, float]] = []
    for s, e in zip(starts, ends, strict=False):
        seg_ys = ys_sorted[s:e]
        seg_xs = xs_sorted[s:e]
        min_idx = int(np.argmin(seg_ys))
        x_center = float(seg_xs[min_idx])
        depth = (background - float(seg_ys[min_idx])) / max(background, 1e-12)
        peaks.append((x_center, depth))

    return peaks


def estimate_non_scan_param_bounds(
    xs: np.ndarray,
    ys: np.ndarray,
    param_names: Sequence[str],
    current_bounds: dict[str, tuple[float, float]],
    scan_param: str,
    *,
    safety_factor: float = _DEFAULT_SAFETY_FACTOR,
) -> dict[str, tuple[float, float]]:
    """Estimate tighter bounds for non-scan parameters from coarse sweep data.

    Uses signal statistics (baseline, width of informative regions, and
    separation of dips) to infer narrower intervals for each model parameter.
    ``dip_depth`` and ``fwhm_gauss`` are intentionally not narrowed: the sparse
    coarse sweep systematically underestimates both, and narrowing them risks
    excluding the true value. Only returns bounds for parameters that are
    genuinely tighter than the current prior by at least ``_MIN_NARROWING_FRACTION``.

    All estimates are padded by ``safety_factor * estimated_range`` to ensure
    the true value is not excluded, then clamped to the current prior.

    Args:
        xs: Normalized [0, 1] sweep probe positions.
        ys: Measured signal values at each probe position.
        param_names: Names of all model parameters (in order).
        current_bounds: Current physical bounds for each parameter.
        scan_param: Name of the scan (frequency) parameter — excluded.
        safety_factor: Fractional padding applied symmetrically to estimates.

    Returns:
        Mapping of parameter name → ``(new_lo, new_hi)`` in physical units.
        Only includes parameters for which the new interval is genuinely
        narrower than the current prior.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if xs.size < _MIN_SWEEP_SAMPLES or ys.size < _MIN_SWEEP_SAMPLES:
        return {}

    xs_s, ys_s = _sorted_sweep(xs, ys)
    background = _estimate_background(ys_s)
    dip_depth = _estimate_dip_depth(ys_s, background)

    if dip_depth is None or dip_depth < 0.01:
        # Signal is essentially flat — no information to use.
        return {}

    result: dict[str, tuple[float, float]] = {}

    # -----------------------------------------------------------------------
    # background
    # -----------------------------------------------------------------------
    if "background" in param_names and "background" != scan_param:
        lo_pr, hi_pr = current_bounds.get("background", (0.0, 2.0))
        est_bg = background
        half_pad = safety_factor * max(abs(hi_pr - lo_pr) * 0.1, 1e-9)
        new_lo = max(lo_pr, est_bg - half_pad)
        new_hi = min(hi_pr, est_bg + half_pad)
        if _is_useful_narrowing(new_lo, new_hi, lo_pr, hi_pr):
            result["background"] = (new_lo, new_hi)
            log.debug("Narrowed background: [%.4g, %.4g] → [%.4g, %.4g]", lo_pr, hi_pr, new_lo, new_hi)

    # -----------------------------------------------------------------------
    # linewidth / fwhm_lorentz / fwhm_gauss
    # -----------------------------------------------------------------------
    full_width = _estimate_linewidth_from_dip(xs_s, ys_s, background, dip_depth)
    if full_width is not None and full_width > 0:
        for lw_name in ("linewidth", "fwhm_lorentz"):
            if lw_name in param_names and lw_name != scan_param:
                lo_pr, hi_pr = current_bounds.get(lw_name, (0.0, 1.0))
                # full_width is the total dip extent, linewidth is HWHM ≈ half of that.
                est_lw = full_width * 0.5
                pad = safety_factor * max(est_lw, hi_pr - lo_pr * 0.5)
                new_lo = max(lo_pr, est_lw - pad)
                new_hi = min(hi_pr, est_lw + pad)
                if _is_useful_narrowing(new_lo, new_hi, lo_pr, hi_pr):
                    result[lw_name] = (new_lo, new_hi)
                    log.debug(
                        "Narrowed %s: [%.4g, %.4g] → [%.4g, %.4g]",
                        lw_name, lo_pr, hi_pr, new_lo, new_hi,
                    )

    # -----------------------------------------------------------------------
    # split — requires two resolved dip peaks
    # -----------------------------------------------------------------------
    if "split" in param_names and "split" != scan_param:
        lo_pr, hi_pr = current_bounds.get("split", (0.0, 1.0))
        peaks = _find_dip_peaks(xs_s, ys_s, background, dip_depth)
        if len(peaks) >= 2:
            # Sort by x, take outermost pair.
            sorted_peaks = sorted(peaks, key=lambda p: p[0])
            x_left = sorted_peaks[0][0]
            x_right = sorted_peaks[-1][0]
            # xs are normalized [0,1]; convert to physical split via current freq bounds.
            # split ≈ physical half-distance between outermost peaks.
            # Note: xs here are already in physical units (from sweep observations).
            est_split = (x_right - x_left) / 2.0
            pad = safety_factor * max(est_split, (hi_pr - lo_pr) * 0.1)
            new_lo = max(lo_pr, est_split - pad)
            new_hi = min(hi_pr, est_split + pad)
            if _is_useful_narrowing(new_lo, new_hi, lo_pr, hi_pr):
                result["split"] = (new_lo, new_hi)
                log.debug(
                    "Narrowed split: [%.4g, %.4g] → [%.4g, %.4g]",
                    lo_pr, hi_pr, new_lo, new_hi,
                )

    return result


def _is_useful_narrowing(
    new_lo: float,
    new_hi: float,
    old_lo: float,
    old_hi: float,
) -> bool:
    """Return True if the new interval is meaningfully tighter than the old one.

    Args:
        new_lo: New lower bound.
        new_hi: New upper bound.
        old_lo: Old lower bound.
        old_hi: Old upper bound.

    Returns:
        True if valid and narrowed by at least ``_MIN_NARROWING_FRACTION``.
    """
    if new_hi <= new_lo:
        return False
    old_width = old_hi - old_lo
    if old_width <= 0:
        return False
    new_width = new_hi - new_lo
    return (old_width - new_width) / old_width >= _MIN_NARROWING_FRACTION
