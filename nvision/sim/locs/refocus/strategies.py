"""Per-dip refocus strategies (A and B) and shape-aware dip detection.

Strategy A — Single dip bounded by background
    Walk outward from a dip minimum until the signal crosses the noise
    threshold (background) on each side.

Strategy B — Monotonic recovery to baseline or to next dip
    Walk outward along monotonic segments.  Stop at baseline crossing or
    when another dip is encountered.
"""

from __future__ import annotations

import numpy as np


def detect_dips(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    noise_threshold: float,
    min_points: int = 3,
    monotonic_tol: float = 1e-9,
) -> list[tuple[float, float]]:
    """Return ``(lo, hi)`` bounds for each double-monotonic dip.

    A *dip* is a region that is monotonically decreasing toward a local
    minimum, then monotonically increasing away from it.  Small wiggles
    within ``monotonic_tol`` are tolerated so noise does not break the
    shape test.

    Parameters
    ----------
    xs, ys :
        Observation coordinates and values.  *Need not be sorted*.
    noise_threshold :
        Points with ``y < noise_threshold`` are considered *potential*
        signal.  Must be provided by the caller.
    min_points :
        Minimum number of points that must participate in the decreasing
        *and* increasing halves of the dip (total, not per-half).
    monotonic_tol :
        Tolerance for declaring a sequence monotonic.  A step ``Δy`` is
        accepted as "decreasing" when ``Δy <= monotonic_tol``.

    Returns
    -------
    list[tuple[float, float]]
        ``(lo, hi)`` in the same coordinate system as *xs*, one entry
        per resolved dip.
    """
    if len(xs) < min_points or len(ys) < min_points:
        return []

    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]

    # Candidate minima: below threshold and locally minimum-ish
    below = ys_s < noise_threshold
    if not np.any(below):
        return []

    # Simple local-minimum test among below-threshold points
    minima_idx: list[int] = []
    for i in range(len(ys_s)):
        if not below[i]:
            continue
        left = ys_s[i - 1] if i > 0 else np.inf
        right = ys_s[i + 1] if i < len(ys_s) - 1 else np.inf
        if ys_s[i] <= left and ys_s[i] <= right:
            minima_idx.append(i)

    if not minima_idx:
        return []

    # For each minimum, walk outward verifying double-monotonicity,
    # stopping when we return to background (noise_threshold)
    dips: list[tuple[float, float]] = []
    for idx in minima_idx:
        left = _walk_monotonic(ys_s, idx, direction=-1, tol=monotonic_tol, noise_threshold=noise_threshold)
        right = _walk_monotonic(ys_s, idx, direction=1, tol=monotonic_tol, noise_threshold=noise_threshold)
        total_pts = (idx - left) + (right - idx) + 1
        if total_pts >= min_points:
            dips.append((float(xs_s[left]), float(xs_s[right])))

    # Merge overlapping dips (can happen when minima are very close)
    return _merge_dips(dips)


def _walk_monotonic(
    ys: np.ndarray,
    start: int,
    *,
    direction: int,
    tol: float,
    noise_threshold: float | None = None,
) -> int:
    """Walk from *start* in *direction* while monotonic away from the minimum.

    Stops when the signal returns to background (``ys[i] >= noise_threshold``)
    or when monotonicity breaks.

    When ``direction == -1`` we require ``ys[i] <= ys[i+1]`` (increasing
    as we move left, i.e. decreasing toward the minimum).
    When ``direction == 1``  we require ``ys[i] >= ys[i-1]`` (increasing
    as we move right, i.e. again decreasing toward the minimum).
    """
    i = start
    if direction == -1:
        while i > 0 and ys[i - 1] >= ys[i] - tol:
            if noise_threshold is not None and ys[i - 1] >= noise_threshold:
                i -= 1
                break
            i -= 1
    else:
        while i < len(ys) - 1 and ys[i + 1] >= ys[i] - tol:
            if noise_threshold is not None and ys[i + 1] >= noise_threshold:
                i += 1
                break
            i += 1
    return i


def _merge_dips(dips: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge dips whose ranges overlap."""
    if len(dips) <= 1:
        return dips
    dips = sorted(dips, key=lambda d: d[0])
    merged: list[tuple[float, float]] = [dips[0]]
    for lo, hi in dips[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def infer_dip_widths(
    xs: np.ndarray,
    ys: np.ndarray,
    dips: list[tuple[float, float]],
    *,
    noise_threshold: float,
) -> list[float]:
    """Per-dip facade: apply Strategy A or B to each dip and return widths.

    For each ``(lo, hi)`` dip returned by :func:`detect_dips`:

    * **Strategy A** — if both sides of the dip contain background points
      (``y >= noise_threshold``), the width is bounded by the distance
      from the dip minimum to the nearest background points.
    * **Strategy B** — otherwise, walk outward along monotonic segments
      until baseline crossing or the next dip boundary is reached.

    The *smallest* width across all dips is the tightest upper bound on
    the true dip width (all dips in a given signal model are assumed to
    share the same width).

    Parameters
    ----------
    xs, ys :
        Full observation arrays (not limited to the dip window).
    dips :
        List of ``(lo, hi)`` as returned by :func:`detect_dips`.
    noise_threshold :
        Same threshold passed to :func:`detect_dips`.

    Returns
    -------
    list[float]
        Inferred width for each dip, in the same coordinate units as *xs*.
    """
    if not dips:
        return []

    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]

    widths: list[float] = []
    for lo, hi in dips:
        # Find the index range of this dip in sorted data
        left_idx = int(np.searchsorted(xs_s, lo, side="left"))
        right_idx = int(np.searchsorted(xs_s, hi, side="right")) - 1
        if left_idx >= right_idx:
            widths.append(hi - lo)
            continue

        min_idx = left_idx + int(np.argmin(ys_s[left_idx : right_idx + 1]))

        # --- Strategy A: look for background on both sides -----------------
        bg_left_idx = None
        for i in range(min_idx - 1, -1, -1):
            if ys_s[i] >= noise_threshold:
                bg_left_idx = i
                break

        bg_right_idx = None
        for i in range(min_idx + 1, len(ys_s)):
            if ys_s[i] >= noise_threshold:
                bg_right_idx = i
                break

        if bg_left_idx is not None and bg_right_idx is not None:
            # Both sides hit background → width bounded by nearest bg points
            width = float(xs_s[bg_right_idx] - xs_s[bg_left_idx])
            widths.append(width)
            continue

        # --- Strategy B: walk outward monotonically -------------------------
        left = _walk_monotonic(ys_s, min_idx, direction=-1, tol=1e-9)
        right = _walk_monotonic(ys_s, min_idx, direction=1, tol=1e-9)

        # If we hit another dip before background, bound by that dip
        if bg_left_idx is None and left > 0:
            # Walk further left to see if another dip is adjacent
            for i in range(left - 1, -1, -1):
                if ys_s[i] >= noise_threshold:
                    left = i
                    break
        if bg_right_idx is None and right < len(ys_s) - 1:
            for i in range(right + 1, len(ys_s)):
                if ys_s[i] >= noise_threshold:
                    right = i
                    break

        width = float(xs_s[right] - xs_s[left])
        widths.append(width)

    return widths
