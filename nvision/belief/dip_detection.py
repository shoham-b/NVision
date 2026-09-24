"""Deterministic dip detection from the measured scan alone.

Finds resonance dips the classic way: threshold the observations ``n_sigma`` noise standard
deviations below the scan's baseline, cluster the below-threshold points, and accept a
cluster only when a binomial test says that many low points among the nearby observations
cannot be noise.

The result depends only on the observations and one scalar noise sigma -- the conjugate
Inverse-Gamma posterior estimate supplied by the belief (see
``SMCMarginalDistribution.estimated_noise_std``). It never reads the SMC particles' inferred
signal parameters, so it is an independent, reproducible statement about where the data show
dips. The belief uses it to place acquisition candidates on the dips it finds, and the SBED
locator reports the same dips as its focus windows.

This is distinct from ``nvision.sim.locs.refocus``, which is geometric (double-monotonic
regions, no statistics) and is used by the coarse sweep locators to narrow their scan window.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, norm

from nvision.sim.defaults import (
    NVISION_DIP_CONFIDENCE,
    NVISION_DIP_MIN_CLUSTER,
    NVISION_DIP_N_SIGMA,
    NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD,
)


@dataclass(frozen=True)
class DipCandidate:
    """A qualified resonance dip: a cluster of low observations that noise cannot explain."""

    centroid_hz: float  # depth-weighted mean frequency of the cluster
    significance: float  # number of below-threshold observations in the cluster
    n_points: int
    f_min: float  # left extent of the cluster (Hz)
    f_max: float  # right extent of the cluster (Hz)
    confidence: float  # binomial confidence that the cluster is not noise
    background: float  # empirical baseline (70th percentile of the observations)


def effective_max_linewidth_hz(phys_bounds: Mapping[str, tuple[float, float]]) -> float:
    """Upper-bound effective HWHM (Hz) implied by the parameter *prior bounds*.

    Lineshape-agnostic: the dip finder only needs one width scale to decide how far apart
    two low points can be and still belong to the same dip. Uses the prior bounds, never an
    inferred estimate.
    """
    if "linewidth" in phys_bounds:
        return phys_bounds["linewidth"][1]
    if "saturation" in phys_bounds and "sigma_inhom" in phys_bounds:
        from nvision.spectra.nv_center import NV_NATURAL_HWHM_HZ

        gamma_hom_hi = NV_NATURAL_HWHM_HZ * math.sqrt(1.0 + phys_bounds["saturation"][1])
        return gamma_hom_hi + math.sqrt(2.0 * math.log(2.0)) * phys_bounds["sigma_inhom"][1]
    if "homogeneous_linewidth" in phys_bounds:
        sigma_inhom_hi = phys_bounds["sigma_inhom"][1] if "sigma_inhom" in phys_bounds else 0.0
        return phys_bounds["homogeneous_linewidth"][1] + math.sqrt(2.0 * math.log(2.0)) * sigma_inhom_hi
    if "fwhm_total" in phys_bounds:
        return phys_bounds["fwhm_total"][1] / 2.0
    raise ValueError(
        "effective_max_linewidth_hz: bounds define no linewidth parameter "
        f"('linewidth', 'homogeneous_linewidth', 'saturation'+'sigma_inhom' or 'fwhm_total'); got {sorted(phys_bounds)}"
    )


def find_dips(
    obs_xs_phys: np.ndarray,
    obs_ys: np.ndarray,
    noise_std: float,
    max_linewidth_hz: float,
    *,
    noise_std_unc: float | None = None,
    n_sigma: float | None = None,
    min_cluster_count: int | None = None,
    confidence_threshold: float | None = None,
    assume_sorted: bool = False,
) -> list[DipCandidate]:
    """Return the dips the observations show, most significant first.

    Args:
        obs_xs_phys: Measured frequencies (Hz). shape: (n_observations,)
        obs_ys: Measured signal values. shape: (n_observations,)
        noise_std: Scalar noise standard deviation of a single observation.
        max_linewidth_hz: Upper bound of the linewidth prior; two below-threshold points
            further apart than ``3 * max_linewidth_hz`` are different dips.
        noise_std_unc: Uncertainty of ``noise_std``. When its ratio to ``noise_std`` is at
            least ``NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD`` the noise level is too poorly
            known to threshold against, and no dips are reported.
        n_sigma: Threshold below the baseline in noise sigmas (default ``NVISION_DIP_N_SIGMA``).
        min_cluster_count: Minimum below-threshold points in a dip (default ``NVISION_DIP_MIN_CLUSTER``).
        confidence_threshold: Minimum binomial confidence (default ``NVISION_DIP_CONFIDENCE``).
        assume_sorted: ``obs_xs_phys`` is already ascending (caller contract, checked): skips
            the sort. Passing True with unsorted input raises ``ValueError``.
    """
    if noise_std <= 0 or not math.isfinite(noise_std):
        raise ValueError(f"find_dips: noise_std must be positive and finite, got {noise_std!r}")
    if len(obs_xs_phys) != len(obs_ys):
        raise ValueError(f"find_dips: {len(obs_xs_phys)} frequencies but {len(obs_ys)} signal values")
    if len(obs_xs_phys) == 0:
        return []

    n_sigma = NVISION_DIP_N_SIGMA if n_sigma is None else n_sigma
    min_cluster_count = NVISION_DIP_MIN_CLUSTER if min_cluster_count is None else min_cluster_count
    confidence_threshold = NVISION_DIP_CONFIDENCE if confidence_threshold is None else confidence_threshold

    if noise_std_unc is not None and noise_std_unc / noise_std >= NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD:
        return []

    if assume_sorted:
        if len(obs_xs_phys) > 1 and not np.all(obs_xs_phys[:-1] <= obs_xs_phys[1:]):
            raise ValueError("find_dips: assume_sorted=True but obs_xs_phys is not ascending.")
        xs, ys = obs_xs_phys, obs_ys
    else:
        order = np.argsort(obs_xs_phys, kind="stable")
        xs, ys = obs_xs_phys[order], obs_ys[order]

    background = float(np.percentile(ys, 70))
    depth = background - ys
    # A point is a dip point when it lies n_sigma noise deviations below the baseline (and is
    # a real, >1%-of-baseline drop).
    is_low = (depth > n_sigma * noise_std) & (depth > 0.01 * max(background, 1e-9))
    if not np.any(is_low):
        logging.debug("find_dips: no observation is %.1f sigma below the baseline.", n_sigma)
        return []

    low_idx = np.flatnonzero(is_low)
    cluster_radius_hz = 3.0 * max_linewidth_hz
    # Consecutive dip points closer than the radius share a cluster.
    breaks = np.flatnonzero(np.diff(xs[low_idx]) > cluster_radius_hz) + 1
    p_false = float(norm.cdf(-n_sigma))

    candidates: list[DipCandidate] = []
    for members in np.split(low_idx, breaks):
        k = len(members)
        if k < min_cluster_count:
            continue
        cx, cy = xs[members], ys[members]
        f_min, f_max = float(cx[0]), float(cx[-1])
        lo = int(np.searchsorted(xs, f_min - cluster_radius_hz, side="left"))
        hi = int(np.searchsorted(xs, f_max + cluster_radius_hz, side="right"))
        # P(at most k-1 of the n_local nearby points fall below the threshold by chance).
        confidence = float(binom.cdf(k - 1, hi - lo, p_false))
        if confidence < confidence_threshold:
            logging.debug(
                "Dip cluster at %.3f Hz dropped by confidence gate: %.4f < %.4f (k=%d, n_local=%d)",
                float(np.mean(cx)),
                confidence,
                confidence_threshold,
                k,
                hi - lo,
            )
            continue
        weights = background - cy
        candidates.append(
            DipCandidate(
                centroid_hz=float(np.sum(cx * weights) / np.sum(weights)),
                significance=float(k),
                n_points=k,
                f_min=f_min,
                f_max=f_max,
                confidence=confidence,
                background=background,
            )
        )

    return sorted(candidates, key=lambda c: c.significance, reverse=True)
