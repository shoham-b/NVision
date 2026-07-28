"""Statistical dip detection for SBED posterior convergence assessment.

Used exclusively by ``SequentialBayesianExperimentDesignLocator`` (via
``compute_focus_window_confidence`` in ``sbed_locator.py``) to decide whether
the SMC posterior has locked onto the correct dip.

The approach is **statistical and particle-aware**: it thresholds observations
using per-particle noise sigmas drawn from the SMC posterior, clusters the
below-threshold points, and runs a binomial confidence test on each cluster.
Returns ``DipCandidate`` objects carrying confidence scores, background
estimates, and cluster extents — rich enough to cross-check against the
particle posterior mean and declare convergence.

This is distinct from ``nvision.sim.locs.refocus``, which is used by the
coarse sweep locators to geometrically narrow the scan window mid-sweep.
The two detectors serve different purposes and are not interchangeable:

- ``refocus.detect_dips``: geometric, returns raw ``(lo, hi)`` edge pairs.
  Used to answer "where should I keep scanning?"
- ``identify_dip_candidates`` (this module): statistical, uses per-particle
  noise sigmas and a binomial confidence test.  Returns ``DipCandidate``
  objects with confidence scores and background estimates.
  Used to answer "has the posterior converged to the right dip?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, norm

from nvision.sim.defaults import (
    NVISION_DIP_CONFIDENCE,
    NVISION_DIP_MIN_CLUSTER,
    NVISION_DIP_N_SIGMA,
    NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD,
)


@dataclass
class DipCandidate:
    """Dataclass holding details of a qualified resonance dip candidate."""

    centroid_hz: float
    significance: float
    n_points: int
    f_min: float = 0.0      # empirical left extent of dip cluster (Hz)
    f_max: float = 0.0      # empirical right extent of dip cluster (Hz)
    confidence: float = 0.0  # binomial confidence of this cluster
    background: float = 1.0  # empirical baseline (70th pct of obs_ys)


def identify_dip_candidates(
    obs_xs: np.ndarray,
    obs_ys: np.ndarray,
    noise_std: float,
    max_linewidth_hz: float,
    *,
    n_sigma: float | None = None,
    min_cluster_count: int | None = None,
    noise_std_unc: float | None = None,
    relative_unc_threshold: float | None = None,
    per_particle_sigmas: np.ndarray | None = None,
    particle_weights: np.ndarray | None = None,
    confidence_threshold: float | None = None,
    max_split_hz: float | None = None,
) -> list[DipCandidate]:
    """Return qualified dip candidates with continuous per-particle support.

    Args:
        obs_xs: Measured frequency positions (Hz). shape: (n_observations,)
        obs_ys: Measured signal values. shape: (n_observations,)
        noise_std: Fallback posterior noise standard deviation.
        max_linewidth_hz: Upper bound of linewidth prior in Hz.
        n_sigma: Threshold sigma. If None, uses NVISION_DIP_N_SIGMA.
        min_cluster_count: Minimum points in a cluster to qualify. If None, uses NVISION_DIP_MIN_CLUSTER.
        noise_std_unc: Uncertainty in noise estimate.
        relative_unc_threshold: Relative uncertainty gating threshold.
        per_particle_sigmas: Noise standard deviation per particle. shape: (n_particles,)
        particle_weights: Particle weights in SMC filter. shape: (n_particles,)
        confidence_threshold: Binomial test confidence gate. If None, uses NVISION_DIP_CONFIDENCE.
        max_split_hz: Physical split upper bound for single-dip prior gating.

    Returns:
        List of DipCandidate objects, sorted by significance descending.
    """
    if len(obs_xs) == 0 or len(obs_ys) == 0:
        return []

    # 1. Uncertainty Gating Check
    if noise_std_unc is not None:
        if relative_unc_threshold is None:
            relative_unc_threshold = NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD

        rel_unc = (noise_std_unc / noise_std) if noise_std > 1e-9 else 0.0
        if rel_unc >= relative_unc_threshold:
            # Noise estimate is still too uncertain
            return []

    # 2. Defaults setup
    if n_sigma is None:
        n_sigma = NVISION_DIP_N_SIGMA
    if min_cluster_count is None:
        min_cluster_count = NVISION_DIP_MIN_CLUSTER
    if confidence_threshold is None:
        confidence_threshold = NVISION_DIP_CONFIDENCE

    # 3. Setup per-particle representation (fallback if none provided)
    if per_particle_sigmas is None:
        per_particle_sigmas = np.array([noise_std])
    if particle_weights is None:
        particle_weights = np.ones_like(per_particle_sigmas)

    # Normalize weights to sum to 1.0
    w_sum = np.sum(particle_weights)
    if w_sum > 0:
        particle_weights = particle_weights / w_sum
    else:
        particle_weights = np.ones_like(particle_weights) / len(particle_weights)

    # 4. Background signal level (70th percentile)
    background = float(np.percentile(obs_ys, 70))
    bg_clamp = max(background, 1e-9)

    # 5. Continuous dip support calculation using sorted sigmas and binary search
    sort_idx = np.argsort(per_particle_sigmas)
    sorted_sigmas = per_particle_sigmas[sort_idx]
    sorted_weights = particle_weights[sort_idx]
    cumulative_w = np.cumsum(sorted_weights)

    dip_support = np.zeros_like(obs_ys, dtype=np.float64)
    required_sigmas = (background - obs_ys) / n_sigma

    # Particle i votes "dip" for point j iff background - y_j > 0.01 * background AND
    # sigma_i < (background - y_j) / n_sigma. Vectorized over j: searchsorted natively
    # takes an array of query values, so the whole per-point binary search + cumulative-
    # weight lookup below is equivalent to the old per-j Python loop, just without the
    # per-iteration interpreter overhead (this was the hottest line in the function).
    eligible = (background - obs_ys) > 0.01 * bg_clamp
    if np.any(eligible):
        # Find, for each eligible point, the first index where sigma_i >= required_sigma_j.
        k = np.searchsorted(sorted_sigmas, required_sigmas[eligible], side="left")
        # Particles with indices < k have sigma_i < required_sigma_j -> they vote "dip".
        dip_support[eligible] = np.where(k > 0, cumulative_w[np.maximum(k - 1, 0)], 0.0)

    # 6. Filter candidate points with dip_support > 0
    candidate_mask = dip_support > 0.0
    cand_xs = obs_xs[candidate_mask]
    cand_ys = obs_ys[candidate_mask]
    cand_support = dip_support[candidate_mask]

    if len(cand_xs) == 0:
        logging.debug("identify_dip_candidates: no candidates with non-zero dip support.")
        return []

    # 7. Sort by frequency before greedy clustering
    sort_idx = np.argsort(cand_xs)
    sorted_xs = cand_xs[sort_idx]
    sorted_ys = cand_ys[sort_idx]
    sorted_support = cand_support[sort_idx]

    cluster_radius_hz = 3.0 * max_linewidth_hz
    clusters: list[list[tuple[float, float, float]]] = []
    current_cluster: list[tuple[float, float, float]] = []

    for x, y, sup in zip(sorted_xs, sorted_ys, sorted_support):
        if not current_cluster:
            current_cluster.append((float(x), float(y), float(sup)))
        else:
            if float(x) - current_cluster[-1][0] <= cluster_radius_hz:
                current_cluster.append((float(x), float(y), float(sup)))
            else:
                clusters.append(current_cluster)
                current_cluster = [(float(x), float(y), float(sup))]

    if current_cluster:
        clusters.append(current_cluster)

    # 8. Filter and compute centroids of qualifying clusters
    p_false = float(norm.cdf(-n_sigma))
    candidates: list[DipCandidate] = []

    for cluster in clusters:
        # Check absolute minimum count first
        if len(cluster) < min_cluster_count:
            continue

        xs = np.array([pt[0] for pt in cluster])
        ys = np.array([pt[1] for pt in cluster])
        sups = np.array([pt[2] for pt in cluster])

        # Binomial confidence gate
        f_min = float(np.min(xs))
        f_max = float(np.max(xs))
        in_window = (obs_xs >= f_min - cluster_radius_hz) & (obs_xs <= f_max + cluster_radius_hz)
        n_local = int(np.sum(in_window))

        S = float(np.sum(sups))
        k_binom = int(np.round(S)) - 1

        if k_binom < 0:
            p_val = 1.0
        else:
            p_val = float(1.0 - binom.cdf(k_binom, n_local, p_false))

        confidence = 1.0 - p_val
        if confidence < confidence_threshold:
            logging.debug(
                f"Cluster around {np.mean(xs):.3f} Hz dropped by confidence gate: "
                f"confidence={confidence:.4f} < threshold={confidence_threshold:.4f} "
                f"(S={S:.2f}, n_local={n_local})"
            )
            continue

        # Centroid weighted by dip_support[j] * (background - y_j)
        weights = sups * (background - ys)
        if np.sum(weights) <= 0:
            centroid = float(np.mean(xs))
        else:
            centroid = float(np.sum(xs * weights) / np.sum(weights))

        candidates.append(
            DipCandidate(
                centroid_hz=centroid,
                significance=S,
                n_points=len(cluster),
                f_min=float(np.min(xs)),
                f_max=float(np.max(xs)),
                confidence=confidence,
                background=background,
            )
        )

    # 9. Layer 1 Single-Dip prior physical split gating
    # Sort candidates by significance (highest first)
    candidates = sorted(candidates, key=lambda c: c.significance, reverse=True)

    if len(candidates) > 0 and max_split_hz is not None:
        center = candidates[0].centroid_hz
        filtered_candidates = []
        for c in candidates:
            if abs(c.centroid_hz - center) <= max_split_hz:
                filtered_candidates.append(c)
            else:
                logging.debug(
                    f"Dip candidate at {c.centroid_hz:.3f} Hz dropped by single-dip prior "
                    f"(distance {abs(c.centroid_hz - center):.3f} Hz > max_split_hz {max_split_hz:.3f})"
                )
        candidates = filtered_candidates

    return candidates
