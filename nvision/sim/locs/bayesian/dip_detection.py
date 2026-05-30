"""Robust dip detection for SBED locators and belief updates.

Provides utilities to identify likely resonance dip candidate frequencies
by filtering observations using posterior noise estimates and clustering them.
"""

from __future__ import annotations

import numpy as np

from nvision.sim.defaults import NVISION_DIP_MIN_CLUSTER, NVISION_DIP_N_SIGMA, NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD


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
) -> list[float]:
    """Return cluster centroids of likely dip locations.

    The cluster merge radius is set to ``3 × max_linewidth_hz``, covering the
    full width of the widest possible dip so that measurements on both flanks
    of a single dip are merged into one candidate rather than treated as two.

    Args:
        obs_xs: Measured frequency positions (Hz). shape: (n_observations,)
        obs_ys: Measured signal values (dimensionless). shape: (n_observations,)
        noise_std: Posterior noise σ estimate (from SMC or heuristic).
        max_linewidth_hz: Upper bound of the linewidth prior in Hz.
            Drives the cluster merge radius as ``3 × max_linewidth_hz``.
        n_sigma: Sigma threshold for dip classification. If None, uses
            NVISION_DIP_N_SIGMA.
        min_cluster_count: Minimum observations per cluster to qualify. If None,
            uses NVISION_DIP_MIN_CLUSTER.
        noise_std_unc: Posterior standard deviation of the noise estimate. If provided,
            used to gate dip detection.
        relative_unc_threshold: Relative uncertainty threshold (noise_std_unc / noise_std).
            If None, uses NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD.

    Returns:
        List of centroid frequencies (Hz) for clusters that qualify.
        Empty list when no dips are identified.
    """
    if len(obs_xs) == 0 or len(obs_ys) == 0:
        return []

    if noise_std_unc is not None:
        if relative_unc_threshold is None:
            relative_unc_threshold = NVISION_DIP_NOISE_UNCERTAINTY_THRESHOLD
        
        # Determine relative uncertainty of the noise standard deviation.
        # If noise_std is extremely close to 0, the noise is perfectly known,
        # meaning relative uncertainty is effectively 0.
        rel_unc = (noise_std_unc / noise_std) if noise_std > 1e-9 else 0.0
        if rel_unc >= relative_unc_threshold:
            # Noise estimate is still too uncertain; do not focus or target dips yet.
            return []

    if n_sigma is None:
        n_sigma = NVISION_DIP_N_SIGMA
    if min_cluster_count is None:
        min_cluster_count = NVISION_DIP_MIN_CLUSTER


    # 1. Compute background as 70th percentile of signals
    background = float(np.percentile(obs_ys, 70))

    # 2. Compute threshold below background.
    dip_thresh = background - max(n_sigma * noise_std, 0.01 * background)

    rel_unc = (noise_std_unc / noise_std) if noise_std_unc is not None and noise_std > 1e-9 else 0.0
    print(f"DEBUG: identify_dip_candidates | n_sigma={n_sigma} | noise_std={noise_std:.5f} | unc={noise_std_unc} | rel_unc={rel_unc:.3f} | thresh={dip_thresh:.5f} | min_y={np.min(obs_ys):.5f}")

    # 3. Find candidate points below threshold
    candidate_mask = obs_ys < dip_thresh
    cand_xs = obs_xs[candidate_mask]
    cand_ys = obs_ys[candidate_mask]

    if len(cand_xs) == 0:
        print("DEBUG: identify_dip_candidates | found NO candidates!")
        return []

    # 4. Greedy clustering by frequency
    # Sort by frequency first
    sort_idx = np.argsort(cand_xs)
    sorted_xs = cand_xs[sort_idx]
    sorted_ys = cand_ys[sort_idx]

    cluster_radius_hz = 3.0 * max_linewidth_hz

    clusters: list[list[tuple[float, float]]] = []
    current_cluster: list[tuple[float, float]] = []

    for x, y in zip(sorted_xs, sorted_ys):
        if not current_cluster:
            current_cluster.append((float(x), float(y)))
        else:
            if float(x) - current_cluster[-1][0] <= cluster_radius_hz:
                current_cluster.append((float(x), float(y)))
            else:
                clusters.append(current_cluster)
                current_cluster = [(float(x), float(y))]

    if current_cluster:
        clusters.append(current_cluster)

    # 5. Keep only clusters with >= min_cluster_count points, and compute depth-weighted centroids
    centroids: list[float] = []
    for cluster in clusters:
        if len(cluster) < min_cluster_count:
            continue

        # Compute depth below background for each point in the cluster as its weight.
        # This biases the centroid toward the deepest part of the dip.
        xs = np.array([pt[0] for pt in cluster])
        ys = np.array([pt[1] for pt in cluster])
        weights = background - ys

        # Safety fallback (if weights are <= 0 or sum to 0, though mathematically
        # they must be > 0 since y < dip_thresh < background).
        if np.sum(weights) <= 0:
            centroid = float(np.mean(xs))
        else:
            centroid = float(np.sum(xs * weights) / np.sum(weights))

        centroids.append(centroid)

    return centroids
