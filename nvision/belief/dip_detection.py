"""Deterministic dip detection from the measured scan alone: a zero-sum Lorentzian matched filter.

A generalized likelihood ratio test (GLRT) for a Lorentzian dip on an unknown local baseline:

1. For each linewidth ``gamma`` (FWHM) on a small grid spanning the linewidth prior
   (``gamma_mean +/- 2 gamma_std``, clipped to the prior's FWHM range) and each candidate
   centre ``c`` (every distinct measured ``drive_freq``), take the observations within
   ``c +/- 1.5 gamma`` and build the Lorentzian template
   ``w_i = (gamma/2)^2 / ((x_i - c)^2 + (gamma/2)^2)`` at their *actual* positions.
2. Subtract the template's mean (annihilates the local baseline offset) and L2-normalise it; the
   matched-filter statistic is ``z = -sum(w_i * y_i) / sigma``, standard normal under "no dip".
   On a uniform sweep this is exactly the cross-correlation of the inverted signal with the
   template; the per-window form also handles SBED's adaptive, irregular and repeated positions.
3. Family-wise error control: with ``N`` (centre, gamma) hypotheses tested, a dip is significant
   when ``z > Phi^-1(1 - p_fa / N)`` (Bonferroni).
4. Peeling: the best significant (centre, gamma) is reported, every hypothesis whose window
   overlaps its centre (or whose centre lies in its window) is suppressed as its echo, and the
   next best is taken -- one :class:`DipCandidate` per significant dip, window ``c +/- 1.5 gamma``.

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

# Number of linewidths evaluated across gamma_mean +/- 2 gamma_std.
_N_GAMMA = 5
# Half-width of the matched-filter window, in FWHM units (window = c +/- 1.5 gamma).
_WINDOW_HALF_FWHM = 1.5
# A zero-sum template needs at least this many points in its window to carry any shape.
_MIN_WINDOW_POINTS = 3


@dataclass(frozen=True)
class DipCandidate:
    """A significant resonance dip: a matched-filter window the noise cannot explain."""

    centroid_hz: float  # matched-filter centre: the measured drive_freq maximising z (Hz)
    significance: float  # matched-filter z-score (noise sigmas; above the FWER threshold, so > 0)
    n_points: int  # observations inside the window
    f_min: float  # left window edge, centre - 1.5 FWHM (Hz; may lie outside the measured axis)
    f_max: float  # right window edge, centre + 1.5 FWHM (Hz)
    confidence: float  # 1 - Bonferroni-corrected p-value of the z-score
    fwhm_hz: float  # matched-filter linewidth (FWHM, Hz) that maximised z


def linewidth_fwhm_prior_hz(phys_bounds: Mapping[str, tuple[float, float]]) -> tuple[float, float, float, float]:
    """Mean, standard deviation, minimum and maximum (Hz) of the dip FWHM implied by the *prior bounds*.

    Lineshape-agnostic: the effective HWHM (``effective_hwhm``) at the lower and at the upper
    bounds spans a uniform prior on the HWHM; the FWHM is twice it, so
    ``mean = hwhm_lo + hwhm_hi``, ``std = 2 (hwhm_hi - hwhm_lo) / sqrt(12)``, ``min = 2 hwhm_lo`` and
    ``max = 2 hwhm_hi``.
    Uses the prior bounds, never an inferred estimate.
    """
    from nvision.spectra.nv_center import effective_hwhm

    hwhm_lo = float(effective_hwhm({name: lo for name, (lo, _) in phys_bounds.items()}))
    hwhm_hi = float(effective_hwhm({name: hi for name, (_, hi) in phys_bounds.items()}))
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
    raise ValueError(
        f"min_linewidth_hz: bounds define no 'linewidth' or 'homogeneous_linewidth'; got {sorted(phys_bounds)}"
    )


def _matched_filter_scan(
    xs: np.ndarray, ys: np.ndarray, centers: np.ndarray, gamma: float, noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-sum Lorentzian matched-filter z-scores at every centre for one FWHM ``gamma``.

    Args:
        xs: Ascending measured drive frequencies (Hz). shape: (n_observations,)
        ys: Measured signal at ``xs``. shape: (n_observations,)
        centers: Candidate dip centres (Hz). shape: (n_centers,)
        gamma: Template FWHM (Hz).
        noise_std: Noise sigma of one observation.

    Returns:
        ``(z, n_in_window)``, both shape (n_centers,). ``z`` is NaN where the window holds fewer
        than ``_MIN_WINDOW_POINTS`` observations or the template is flat (all points at one x).
    """
    half = _WINDOW_HALF_FWHM * gamma
    lo = np.searchsorted(xs, centers - half, side="left")
    hi = np.searchsorted(xs, centers + half, side="right")
    counts = hi - lo
    z = np.full(centers.shape, np.nan)
    width = int(counts.max()) if counts.size else 0
    if width < _MIN_WINDOW_POINTS:
        return z, counts
    # Padded (n_centers, width) gather of every window; mask marks the real entries.
    offsets = np.arange(width)
    mask = offsets[None, :] < counts[:, None]
    idx = np.minimum(lo[:, None] + offsets[None, :], len(xs) - 1)
    dx = xs[idx] - centers[:, None]
    hw2 = (0.5 * gamma) ** 2
    w = np.where(mask, hw2 / (dx * dx + hw2), 0.0)
    w_zero = np.where(mask, w - (w.sum(axis=1) / np.maximum(counts, 1))[:, None], 0.0)
    norm_w = np.sqrt(np.sum(w_zero * w_zero, axis=1))
    valid = (counts >= _MIN_WINDOW_POINTS) & (norm_w > 1e-12)
    corr = -np.sum(w_zero * ys[idx], axis=1)
    z[valid] = corr[valid] / (norm_w[valid] * noise_std)
    return z, counts


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
        gamma_min_hz: Narrowest FWHM the prior allows (Hz, in (0, gamma_mean_hz]); the linewidth grid
            never goes below it.
        gamma_max_hz: Widest FWHM the prior allows (Hz, >= gamma_mean_hz); the linewidth grid never
            goes above it.
        noise_std_unc: Uncertainty of ``noise_std``. When its ratio to ``noise_std`` is at
            least ``NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD`` the noise level is too poorly
            known to normalise z-scores, and no dips are reported.
        p_fa: Family-wise false-alarm probability over all (centre, gamma) hypotheses tested
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

    centers = np.unique(xs)
    if centers.size < _MIN_WINDOW_POINTS:
        return []
    # gamma_mean +/- 2 gamma_std overshoots both ends of a uniform prior (the low end even below 0); a
    # template outside the allowed widths only spends hypotheses (raising the threshold).
    gamma_lo = max(gamma_min_hz, gamma_mean_hz - 2.0 * gamma_std_hz)
    gamma_hi = min(gamma_max_hz, gamma_mean_hz + 2.0 * gamma_std_hz)
    gammas = np.unique(np.linspace(gamma_lo, gamma_hi, _N_GAMMA))

    # Every testable (centre, gamma) hypothesis, flattened. shape: (n_hypotheses,) each.
    hyp_z, hyp_c, hyp_g, hyp_n = [], [], [], []
    for gamma in gammas:
        z, counts = _matched_filter_scan(xs, ys, centers, float(gamma), noise_std)
        ok = np.isfinite(z)
        hyp_z.append(z[ok])
        hyp_c.append(centers[ok])
        hyp_g.append(np.full(int(ok.sum()), float(gamma)))
        hyp_n.append(counts[ok])
    z_all = np.concatenate(hyp_z)
    n_tests = z_all.size
    if n_tests == 0:
        return []
    c_all, g_all, n_all = np.concatenate(hyp_c), np.concatenate(hyp_g), np.concatenate(hyp_n)

    z_thresh = float(norm.isf(p_fa / n_tests))
    ranked = np.argsort(-z_all, kind="stable")
    ranked = ranked[z_all[ranked] > z_thresh]
    if ranked.size == 0:
        logging.debug(
            "find_dips: best matched-filter z %.2f <= FWER threshold %.2f (%d hypotheses).",
            float(np.max(z_all)),
            z_thresh,
            n_tests,
        )
        return []

    candidates: list[DipCandidate] = []
    while ranked.size:
        k = int(ranked[0])
        c, g = float(c_all[k]), float(g_all[k])
        # Peeling: every remaining hypothesis overlapping this dip's centre is its echo.
        rest = ranked[1:]
        ranked = rest[np.abs(c_all[rest] - c) > _WINDOW_HALF_FWHM * np.maximum(g_all[rest], g)]
        z = float(z_all[k])
        candidates.append(
            DipCandidate(
                centroid_hz=c,
                significance=z,
                n_points=int(n_all[k]),
                f_min=c - _WINDOW_HALF_FWHM * g,
                f_max=c + _WINDOW_HALF_FWHM * g,
                confidence=float(1.0 - min(1.0, n_tests * norm.sf(z))),
                fwhm_hz=g,
            )
        )
    return candidates
