"""Deterministic, shape-agnostic dip detection from the measured scan alone: a boxcar GLRT scan.

A generalized likelihood ratio test for "a stretch of the scan sits below the baseline", with no
assumption about the dip's lineshape (Lorentzian, Voigt, saturated, hyperfine-merged...):

1. Per-observation dip z-scores ``z_i = (b - y_i) / sigma``. ``sigma`` is the belief's conjugate
   noise estimate; the baseline ``b`` is the 70th percentile of ``y`` minus its Gaussian offset
   ``Phi^-1(0.7) sigma``, so ``z_i ~ N(0, 1)`` under "no dip" while points inside dips (all below
   the baseline) cannot drag ``b`` down until they outnumber the points above it -- SBED piles its
   measurements onto a dip once it has found one, which would sink a plain median.
2. Every contiguous window (from one distinct measured ``drive_freq`` to another, repeats included)
   whose span lies in the linewidth prior's ``gamma_mean +/- 2 gamma_std`` (clipped to the prior's
   FWHM range) gets the boxcar GLRT score ``sum(z_i) / sqrt(n)``, standard normal under "no dip",
   from prefix sums in O(1) per window. The prior only sets which widths are tried.
3. Family-wise error control: with ``N`` windows tested, a window is significant when
   ``score > Phi^-1(1 - p_fa / N)`` (Bonferroni).
4. Peeling: the best significant window is reported, every window reaching within one window width of
   it (the dip's wings, which a deep dip also pushes below the baseline) is suppressed as its echo, and
   the next best is taken -- one :class:`DipCandidate` per significant dip.

On a uniform sweep this is exactly the prefix-sum scan over index widths. The window-in-Hz form
also handles SBED's adaptive, irregular and repeated positions.

The result depends only on the observations, one scalar noise sigma -- the conjugate
Inverse-Gamma posterior estimate supplied by the belief (see
``SMCMarginalDistribution.estimated_noise_std``) -- and the linewidth *prior* (never the SMC
particles), so it is an independent, reproducible statement about where the data show dips. The
belief uses it to place acquisition candidates on the dips it finds, and the SBED locator builds
its focus sections from the same dips.

This is distinct from ``nvision.sim.locs.refocus``, which is geometric (double-monotonic
regions, no statistics) and is used by the coarse sweep locators to narrow their scan window.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from nvision.sim.defaults import NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD, NVISION_DIP_P_FA

# Baseline percentile of the observed signal (see the module docstring, step 1).
_BASELINE_PERCENTILE = 70.0


@dataclass(frozen=True)
class DipCandidate:
    """A significant resonance dip: a stretch of the scan the noise cannot explain as baseline."""

    centroid_hz: float  # middle of the winning window (Hz)
    significance: float  # boxcar GLRT score (noise sigmas; above the FWER threshold, so > 0)
    n_points: int  # observations inside the window
    f_min: float  # left window edge: a measured drive_freq (Hz)
    f_max: float  # right window edge: a measured drive_freq (Hz)
    confidence: float  # 1 - Bonferroni-corrected p-value of the score
    width_hz: float  # window span f_max - f_min (Hz), within the prior's FWHM range


def linewidth_fwhm_prior_hz(phys_bounds: Mapping[str, tuple[float, float]]) -> tuple[float, float, float, float]:
    """Mean, standard deviation, minimum and maximum (Hz) of the dip FWHM implied by the *prior bounds*.

    Lineshape-agnostic: the effective HWHM (``effective_hwhm``) at the lower and at the upper
    bounds spans a uniform prior on the HWHM; the FWHM is twice it, so
    ``mean = hwhm_lo + hwhm_hi``, ``std = 2 (hwhm_hi - hwhm_lo) / sqrt(12)``, ``min = 2 hwhm_lo`` and
    ``max = 2 hwhm_hi``.
    Uses the prior bounds, never an inferred estimate.
    """
    from nvision.spectra.nv_center import effective_hwhm
    from nvision.spectra.nv_center_drive import is_drive_param_dict

    narrowest = {name: lo for name, (lo, _) in phys_bounds.items()}
    widest = {name: hi for name, (_, hi) in phys_bounds.items()}
    if is_drive_param_dict(widest):
        from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT

        # Drive-controlled: widest at full amplitude with the strongest coupling (smallest
        # drive_amp_sat / largest rabi_per_amp), narrowest at the minimum amplitude with the weakest.
        if "drive_amp_sat" in phys_bounds:
            narrowest["drive_amp_sat"], widest["drive_amp_sat"] = phys_bounds["drive_amp_sat"][::-1]
        hwhm_lo = float(effective_hwhm(narrowest, drive_amp_unit=DRIVE_AMP_MIN_UNIT))
        hwhm_hi = float(effective_hwhm(widest, drive_amp_unit=1.0))
    else:
        hwhm_lo = float(effective_hwhm(narrowest))
        hwhm_hi = float(effective_hwhm(widest))
    if not (0.0 < hwhm_lo <= hwhm_hi) or not math.isfinite(hwhm_hi):
        raise ValueError(f"linewidth_fwhm_prior_hz: invalid HWHM prior range [{hwhm_lo}, {hwhm_hi}] Hz")
    return hwhm_lo + hwhm_hi, 2.0 * (hwhm_hi - hwhm_lo) / math.sqrt(12.0), 2.0 * hwhm_lo, 2.0 * hwhm_hi


def min_linewidth_hz(phys_bounds: Mapping[str, tuple[float, float]]) -> float:
    """Lower bound (Hz) of the linewidth prior: the narrowest dip the prior allows.

    Sets how finely a uniform sweep must sample to be sure of hitting a dip.
    """
    for name in ("linewidth", "homogeneous_linewidth"):
        if name in phys_bounds:
            return float(phys_bounds[name][0])
    from nvision.spectra.nv_center_drive import BASELINE_DRIVE_AMP_UNIT, drive_effective_hwhm, is_drive_param_dict

    if is_drive_param_dict(phys_bounds):
        # Drive-controlled: the narrowest dip at the baseline drive amplitude (the drive a fixed-drive
        # sweep uses), with the weakest coupling and no inhomogeneous broadening. Every locator on the
        # same generator gets the same budget from this, so drive-choosing and fixed-drive runs compare
        # at equal step counts.
        narrowest = {"sigma_inhom": 0.0}
        if "drive_amp_sat" in phys_bounds:
            narrowest["drive_amp_sat"] = phys_bounds["drive_amp_sat"][1]
        else:
            narrowest["rabi_per_amp"] = phys_bounds["rabi_per_amp"][0]
        return float(drive_effective_hwhm(narrowest, BASELINE_DRIVE_AMP_UNIT))
    raise ValueError(
        f"min_linewidth_hz: bounds define no 'linewidth' or 'homogeneous_linewidth'; got {sorted(phys_bounds)}"
    )


def _window_scores(
    xs: np.ndarray, z: np.ndarray, width_lo_hz: float, width_hi_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boxcar GLRT score of every contiguous window whose span lies in ``[width_lo_hz, width_hi_hz]``.

    A window runs from one distinct measured position to another (inclusive, all repeats at both
    ends included); its score is ``sum(z_i) / sqrt(n)`` over its ``n`` observations, computed from
    prefix sums in O(1) per window.

    Args:
        xs: Ascending measured drive frequencies (Hz). shape: (n_observations,)
        z: Per-observation dip z-scores ``(baseline - y) / sigma``. shape: (n_observations,)
        width_lo_hz: Narrowest window span tested (Hz).
        width_hi_hz: Widest window span tested (Hz).

    Returns:
        ``(score, lo_hz, hi_hz, n_points)``, each shape (n_windows,): the GLRT score, left and right
        edge (Hz) and observation count of every tested window.
    """
    prefix = np.concatenate(([0.0], np.cumsum(z)))
    xu, first = np.unique(xs, return_index=True)  # distinct positions and their first index
    end = np.append(first[1:], len(xs))  # one past the last index of each distinct position
    b_lo = np.searchsorted(xu, xu + width_lo_hz, side="left")
    b_hi = np.searchsorted(xu, xu + width_hi_hz, side="right")
    counts = np.maximum(b_hi - b_lo, 0)
    total = int(counts.sum())
    if total == 0:
        empty = np.empty(0)
        return empty, empty, empty, np.empty(0, dtype=int)
    # Flatten every (start a, end b) pair: a repeated counts[a] times, b running b_lo[a] .. b_hi[a]-1.
    a = np.repeat(np.arange(xu.size), counts)
    b = b_lo[a] + np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    n_points = end[b] - first[a]
    score = (prefix[end[b]] - prefix[first[a]]) / np.sqrt(n_points)
    return score, xu[a], xu[b], n_points


def find_dips(
    obs_drive_freqs_phys: np.ndarray,
    obs_ys: np.ndarray,
    noise_std: float,
    gamma_mean_hz: float,
    gamma_std_hz: float,
    gamma_min_hz: float,
    gamma_max_hz: float,
    *,
    noise_std_unc: float | None = None,
    p_fa: float | None = None,
    assume_sorted: bool = False,
) -> list[DipCandidate]:
    """Return the significant dips the observations show, most significant first (see module docstring).

    Args:
        obs_drive_freqs_phys: Measured drive frequencies (Hz). shape: (n_observations,)
        obs_ys: Measured signal values. shape: (n_observations,)
        noise_std: Scalar noise standard deviation of a single observation.
        gamma_mean_hz: Prior mean of the dip FWHM (Hz), e.g. from :func:`linewidth_fwhm_prior_hz`.
        gamma_std_hz: Prior standard deviation of the dip FWHM (Hz, >= 0).
        gamma_min_hz: Narrowest FWHM the prior allows (Hz, in (0, gamma_mean_hz]); no window is
            narrower.
        gamma_max_hz: Widest FWHM the prior allows (Hz, >= gamma_mean_hz); no window is wider.
        noise_std_unc: Uncertainty of ``noise_std``. When its ratio to ``noise_std`` is at
            least ``NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD`` the noise level is too poorly
            known to form z-scores, and no dips are reported.
        p_fa: Family-wise false-alarm probability over all windows tested
            (default ``NVISION_DIP_P_FA``).
        assume_sorted: ``obs_drive_freqs_phys`` is already ascending (caller contract, checked): skips
            the sort. Passing True with unsorted input raises ``ValueError``.
    """
    if noise_std <= 0 or not math.isfinite(noise_std):
        raise ValueError(f"find_dips: noise_std must be positive and finite, got {noise_std!r}")
    if not (gamma_mean_hz > 0) or not math.isfinite(gamma_mean_hz):
        raise ValueError(f"find_dips: gamma_mean_hz must be positive and finite, got {gamma_mean_hz!r}")
    if not (gamma_std_hz >= 0) or not math.isfinite(gamma_std_hz):
        raise ValueError(f"find_dips: gamma_std_hz must be non-negative and finite, got {gamma_std_hz!r}")
    if not (0.0 < gamma_min_hz <= gamma_mean_hz):
        raise ValueError(
            f"find_dips: gamma_min_hz must lie in (0, gamma_mean_hz={gamma_mean_hz}], got {gamma_min_hz!r}"
        )
    if not (gamma_max_hz >= gamma_mean_hz) or not math.isfinite(gamma_max_hz):
        raise ValueError(
            f"find_dips: gamma_max_hz must be finite and >= gamma_mean_hz={gamma_mean_hz}, got {gamma_max_hz!r}"
        )
    p_fa = NVISION_DIP_P_FA if p_fa is None else p_fa
    if not (0.0 < p_fa < 1.0):
        raise ValueError(f"find_dips: p_fa must lie in (0, 1), got {p_fa!r}")
    if len(obs_drive_freqs_phys) != len(obs_ys):
        raise ValueError(f"find_dips: {len(obs_drive_freqs_phys)} frequencies but {len(obs_ys)} signal values")
    if len(obs_drive_freqs_phys) == 0:
        return []

    if noise_std_unc is not None and noise_std_unc / noise_std >= NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD:
        return []

    xs = np.asarray(obs_drive_freqs_phys, dtype=float)
    ys = np.asarray(obs_ys, dtype=float)
    if assume_sorted:
        if len(xs) > 1 and not np.all(xs[:-1] <= xs[1:]):
            raise ValueError("find_dips: assume_sorted=True but obs_drive_freqs_phys is not ascending.")
    else:
        order = np.argsort(xs, kind="stable")
        xs, ys = xs[order], ys[order]

    # Baseline: the upper percentile minus its Gaussian offset, an unbiased baseline estimate under
    # "no dip" that dip points (all below it) cannot drag down until they outnumber the points above it.
    baseline = (
        float(np.percentile(ys, _BASELINE_PERCENTILE)) - float(norm.ppf(_BASELINE_PERCENTILE / 100.0)) * noise_std
    )
    z = (baseline - ys) / noise_std

    # gamma_mean +/- 2 gamma_std overshoots both ends of a uniform prior (the low end even below 0); a
    # window outside the allowed widths only spends hypotheses (raising the threshold).
    width_lo_hz = max(gamma_min_hz, gamma_mean_hz - 2.0 * gamma_std_hz)
    width_hi_hz = min(gamma_max_hz, gamma_mean_hz + 2.0 * gamma_std_hz)
    score, lo_hz, hi_hz, n_points = _window_scores(xs, z, width_lo_hz, width_hi_hz)
    n_tests = score.size
    if n_tests == 0:
        return []

    z_thresh = float(norm.isf(p_fa / n_tests))
    ranked = np.argsort(-score, kind="stable")
    ranked = ranked[score[ranked] > z_thresh]
    if ranked.size == 0:
        logging.debug(
            "find_dips: best boxcar z %.2f <= FWER threshold %.2f (%d windows).",
            float(np.max(score)),
            z_thresh,
            n_tests,
        )
        return []

    candidates: list[DipCandidate] = []
    while ranked.size:
        k = int(ranked[0])
        f_min, f_max = float(lo_hz[k]), float(hi_hz[k])
        # Peeling: the window covers only the dip's core, and a deep dip's wings sit significantly below
        # the baseline too, so every remaining window reaching within one window width of it is its echo.
        margin = f_max - f_min
        rest = ranked[1:]
        ranked = rest[(lo_hz[rest] > f_max + margin) | (hi_hz[rest] < f_min - margin)]
        z_k = float(score[k])
        candidates.append(
            DipCandidate(
                centroid_hz=0.5 * (f_min + f_max),
                significance=z_k,
                n_points=int(n_points[k]),
                f_min=f_min,
                f_max=f_max,
                confidence=float(1.0 - min(1.0, n_tests * norm.sf(z_k))),
                width_hz=f_max - f_min,
            )
        )
    return candidates
