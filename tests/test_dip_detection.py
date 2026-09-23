"""Unit tests for robust dip detection and noise estimation."""

from __future__ import annotations

import numpy as np
import pytest

from nvision import nv_center_smc_belief
from nvision.sim.locs.bayesian.dip_detection import identify_dip_candidates


def test_identify_dip_candidates_empty():
    # Empty inputs
    assert identify_dip_candidates(np.array([]), np.array([]), 0.02, 5e6) == []


def test_identify_dip_candidates_no_dips():
    # Signals all around background level (1.0), with noise std of 0.02, min cluster count 2.
    obs_xs = np.linspace(2.8e9, 2.9e9, 10)
    obs_ys = np.ones(10) + np.random.uniform(-0.01, 0.01, 10)
    # The threshold will be background - max(3 * 0.02, 0.01) = 1.0 - 0.06 = 0.94.
    # All signals are ~1.0, so no dips are found.
    assert identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2) == []


def test_identify_dip_candidates_single_isolated_spike():
    # A single isolated point below threshold.
    # With min_cluster_count=2, it should be filtered out.
    obs_xs = np.linspace(2.8e9, 2.9e9, 10)
    obs_ys = np.ones(10)
    obs_ys[5] = 0.8  # Deep dip at index 5, but only 1 point.
    # Threshold background is ~1.0. dip_thresh = 1.0 - 0.06 = 0.94.
    # The point at index 5 is 0.8 < 0.94, but it is alone.
    assert identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2) == []


def test_identify_dip_candidates_cluster_detection():
    # Two points close to each other below threshold.
    # Should be detected as a cluster, and return its signal-depth-weighted centroid.
    obs_xs = np.array([2.8e9, 2.81e9, 2.82e9, 2.822e9, 2.824e9, 2.83e9, 2.84e9, 2.85e9, 2.86e9, 2.87e9])
    obs_ys = np.array([1.0, 1.0, 0.8, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])
    # Points at 2.82e9, 2.822e9, 2.824e9 are below 0.94.
    # They are within 3 * 5 MHz = 15 MHz of each other, so they form a single cluster.
    # Depth weights: background = 1.0.
    # weights: 1.0 - 0.8 = 0.2, 1.0 - 0.7 = 0.3, 1.0 - 0.8 = 0.2.
    # centroid = (2.82e9 * 0.2 + 2.822e9 * 0.3 + 2.824e9 * 0.2) / 0.7 = 2.822e9.
    centroids = identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2)
    assert len(centroids) == 1
    assert np.isclose(centroids[0].centroid_hz, 2.822e9)


def test_identify_dip_candidates_per_particle_voting():
    obs_xs = np.array([2.8e9, 2.802e9, 2.9e9, 2.91e9, 2.92e9, 2.93e9])
    obs_ys = np.array([0.7, 0.7, 1.0, 1.0, 1.0, 1.0])

    # 2 particles:
    # Particle 0: sigma = 0.01, weight = 0.8
    # Particle 1: sigma = 0.15, weight = 0.2
    per_particle_sigmas = np.array([0.01, 0.15])
    particle_weights = np.array([0.8, 0.2])

    candidates = identify_dip_candidates(
        obs_xs,
        obs_ys,
        noise_std=0.05,
        max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
    )
    assert len(candidates) == 1
    assert np.isclose(candidates[0].significance, 1.6)


def test_identify_dip_candidates_binomial_confidence_gate():
    obs_xs = np.array([2.8e9, 2.802e9, 2.9e9, 2.91e9, 2.92e9, 2.93e9, 2.94e9, 2.95e9, 2.96e9, 2.97e9])
    obs_ys = np.array([0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # 2 particles:
    # Particle 0: sigma = 0.01, weight = 0.1
    # Particle 1: sigma = 0.04, weight = 0.9
    per_particle_sigmas = np.array([0.01, 0.04])
    particle_weights = np.array([0.1, 0.9])

    cands_gated = identify_dip_candidates(
        obs_xs,
        obs_ys,
        noise_std=0.03,
        max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.99,
    )
    assert cands_gated == []

    cands_ungated = identify_dip_candidates(
        obs_xs,
        obs_ys,
        noise_std=0.03,
        max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
    )
    assert len(cands_ungated) == 1


def test_identify_dip_candidates_single_dip_prior():
    obs_xs = np.array([2.82e9, 2.822e9, 2.92e9, 2.922e9, 2.85e9, 2.86e9, 2.87e9, 2.88e9])
    obs_ys = np.array([0.7, 0.7, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0])

    per_particle_sigmas = np.array([0.01, 0.05])
    particle_weights = np.array([0.8, 0.2])

    cands_gated = identify_dip_candidates(
        obs_xs,
        obs_ys,
        noise_std=0.03,
        max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
        max_split_hz=10e6,
    )
    assert len(cands_gated) == 1
    assert np.isclose(cands_gated[0].centroid_hz, 2.821e9)

    cands_ungated = identify_dip_candidates(
        obs_xs,
        obs_ys,
        noise_std=0.03,
        max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
        max_split_hz=150e6,
    )
    assert len(cands_ungated) == 2


def test_identify_dip_candidates_assume_sorted_matches_default():
    # Same fixture as test_identify_dip_candidates_cluster_detection, pre-sorted
    # (it already happens to be ascending, but sort explicitly to state the
    # assume_sorted=True precondition rather than relying on that coincidence).
    obs_xs_raw = np.array([2.8e9, 2.81e9, 2.82e9, 2.822e9, 2.824e9, 2.83e9, 2.84e9, 2.85e9, 2.86e9, 2.87e9])
    obs_ys_raw = np.array([1.0, 1.0, 0.8, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])
    order = np.argsort(obs_xs_raw)
    obs_xs, obs_ys = obs_xs_raw[order], obs_ys_raw[order]

    default = identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2)
    fast = identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2, assume_sorted=True)

    assert len(default) == len(fast) == 1
    assert np.isclose(default[0].centroid_hz, fast[0].centroid_hz)
    assert np.isclose(default[0].significance, fast[0].significance)
    assert default[0].n_points == fast[0].n_points
    assert np.isclose(default[0].confidence, fast[0].confidence)
    assert np.isclose(default[0].background, fast[0].background)


def test_identify_dip_candidates_assume_sorted_matches_default_multi_cluster():
    # Same fixture as test_identify_dip_candidates_single_dip_prior, sorted --
    # exercises the multi-cluster in_window/searchsorted path.
    obs_xs_raw = np.array([2.82e9, 2.822e9, 2.92e9, 2.922e9, 2.85e9, 2.86e9, 2.87e9, 2.88e9])
    obs_ys_raw = np.array([0.7, 0.7, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0])
    order = np.argsort(obs_xs_raw)
    obs_xs, obs_ys = obs_xs_raw[order], obs_ys_raw[order]

    per_particle_sigmas = np.array([0.01, 0.05])
    particle_weights = np.array([0.8, 0.2])
    kwargs = dict(
        noise_std=0.03,
        max_linewidth_hz=5e6,
        per_particle_sigmas=per_particle_sigmas,
        particle_weights=particle_weights,
        min_cluster_count=2,
        confidence_threshold=0.0,
        max_split_hz=150e6,
    )

    default = identify_dip_candidates(obs_xs, obs_ys, **kwargs)
    fast = identify_dip_candidates(obs_xs, obs_ys, assume_sorted=True, **kwargs)

    assert len(default) == len(fast) == 2
    for d, f in zip(default, fast, strict=True):
        assert np.isclose(d.centroid_hz, f.centroid_hz)
        assert np.isclose(d.significance, f.significance)
        assert d.n_points == f.n_points
        assert np.isclose(d.confidence, f.confidence)


def test_identify_dip_candidates_assume_sorted_raises_on_unsorted_input():
    obs_xs = np.array([2.8e9, 2.9e9, 2.85e9])  # not ascending
    obs_ys = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="assume_sorted"):
        identify_dip_candidates(obs_xs, obs_ys, 0.02, 5e6, assume_sorted=True)


def test_identify_dip_candidates_uncertainty_gating():
    # Setup observations with a valid dip cluster
    obs_xs = np.array([2.8e9, 2.81e9, 2.82e9, 2.822e9, 2.824e9, 2.83e9, 2.84e9, 2.85e9, 2.86e9, 2.87e9])
    obs_ys = np.array([1.0, 1.0, 0.8, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Case A: High relative uncertainty (noise_std_unc = 0.01, noise_std = 0.05 => 20% relative unc).
    # Since 20% >= 15% (default threshold), it should gate the dip and return []
    centroids_gated = identify_dip_candidates(obs_xs, obs_ys, noise_std=0.05, max_linewidth_hz=5e6, noise_std_unc=0.01)
    assert centroids_gated == []

    # Case B: Low relative uncertainty (noise_std_unc = 0.002, noise_std = 0.05 => 4% relative unc).
    # Since 4% < 15%, it should successfully return the cluster centroid
    centroids_ungated = identify_dip_candidates(
        obs_xs, obs_ys, noise_std=0.05, max_linewidth_hz=5e6, noise_std_unc=0.002
    )
    assert len(centroids_ungated) == 1
    assert np.isclose(centroids_ungated[0].centroid_hz, 2.822e9)


def test_sorted_observation_arrays_matches_full_sort():
    # _append_observation() doesn't touch _obs_sort_order (maintained lazily by
    # sorted_observation_arrays()), so exercise it directly without going
    # through the full model/likelihood update path.
    b = nv_center_smc_belief(num_particles=50)
    rng = np.random.default_rng(42)
    xs = rng.uniform(0.0, 1.0, 200)
    for x in xs:
        b._append_observation(float(x), float(rng.normal()))

    xs_sorted, ys_sorted = b.sorted_observation_arrays()
    xs_raw, ys_raw = b.observation_arrays()
    expected_order = np.argsort(xs_raw)

    assert np.array_equal(xs_sorted, xs_raw[expected_order])
    assert np.array_equal(ys_sorted, ys_raw[expected_order])
    assert np.all(np.diff(xs_sorted) >= 0)


def test_sorted_observation_arrays_lazy_incremental_across_calls():
    # Interleave appends with reads to exercise the lazy catch-up loop running
    # more than once (each call should only re-sort points added since the
    # previous call, not the whole history).
    b = nv_center_smc_belief(num_particles=50)
    rng = np.random.default_rng(7)
    seen: list[float] = []
    for _ in range(5):
        for _ in range(20):
            x = float(rng.uniform(0.0, 1.0))
            b._append_observation(x, float(rng.normal()))
            seen.append(x)
        xs_sorted, _ = b.sorted_observation_arrays()
        assert np.all(np.diff(xs_sorted) >= 0)
        assert np.array_equal(xs_sorted, np.sort(np.array(seen)))
        # A call with no new observations in between must be a stable no-op.
        xs_sorted_again, _ = b.sorted_observation_arrays()
        assert np.array_equal(xs_sorted, xs_sorted_again)


def test_resync_sort_position_after_stale_insertion():
    # Reproduces the UnitCubeSMCMarginalDistribution coordinate-frame case:
    # dip detection (via sorted_observation_arrays()) runs mid-update using a
    # provisional value, then the caller overwrites _obs_x_arr with the real
    # one afterwards. _resync_sort_position must restore global sortedness.
    b = nv_center_smc_belief(num_particles=50)
    for x in (0.1, 0.5, 0.9):
        b._append_observation(x, 1.0)
    b.sorted_observation_arrays()  # finalizes _obs_sort_valid_count == 3

    b._append_observation(0.99, 1.0)  # stale/provisional value, e.g. narrowed-frame
    b.sorted_observation_arrays()  # incorporates it at the (wrong) high end

    b._obs_x_arr[3] = 0.2  # caller's post-hoc correction to the real value
    b._resync_sort_position(3)

    xs_sorted, _ = b.sorted_observation_arrays()
    assert np.array_equal(xs_sorted, np.array([0.1, 0.2, 0.5, 0.9]))


def test_resync_sort_position_noop_before_first_read():
    # If sorted_observation_arrays() never ran, the newly-appended index isn't
    # in the sort order yet -- resync must be a no-op, not raise or corrupt state.
    b = nv_center_smc_belief(num_particles=50)
    b._append_observation(0.5, 1.0)
    b._append_observation(0.99, 1.0)
    b._obs_x_arr[1] = 0.1  # correct before it was ever read
    b._resync_sort_position(1)  # no-op: valid_count is still 0

    xs_sorted, _ = b.sorted_observation_arrays()
    assert np.array_equal(xs_sorted, np.array([0.1, 0.5]))


def test_unit_cube_belief_narrowing_dip_detection_stays_consistent():
    # End-to-end: force narrowing + resampling (which triggers dip detection
    # via _generate_epoch_candidates -> sorted_observation_arrays) across many
    # updates, so _resync_sort_position is actually exercised through the real
    # UnitCubeSMCMarginalDistribution.update() path, not just called directly.
    from nvision.models.observation import Observation
    from nvision.spectra.noise_model import GaussianNoiseSignalModel

    # frequency must be a free particle dimension for _resample() to narrow it
    # (narrow_scan_parameter_physical_bounds requires it in _param_names) --
    # nv_center_smc_belief defaults to with_fixed_frequency=True, which skips
    # narrowing entirely, so no coordinate-frame divergence would ever occur.
    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.05)})
    b = nv_center_smc_belief(num_particles=100, noise_model=noise_model, with_fixed_frequency=False)
    initial_width = b.physical_x_bounds[1] - b.physical_x_bounds[0]

    rng = np.random.default_rng(123)
    for i in range(60):
        # Draw from the CURRENT (possibly already-narrowed) window each time,
        # matching how a real locator only ever probes inside its live bounds.
        lo, hi = b.physical_x_bounds
        x = float(rng.uniform(lo, hi))
        b.update(Observation(x=x, signal_value=float(rng.uniform(0.5, 1.0))))
        if i == 20:
            # One explicit narrowing partway through -- this is exactly what
            # makes update()'s post-hoc x-correction diverge from the value
            # _append_observation saw (_resync_sort_position's reason to exist).
            lo0, hi0 = b.physical_x_bounds
            b.narrow_scan_parameter_physical_bounds("frequency", lo0 + 0.1 * (hi0 - lo0), hi0 - 0.1 * (hi0 - lo0))

    final_width = b.physical_x_bounds[1] - b.physical_x_bounds[0]
    assert final_width < initial_width, "test didn't actually exercise narrowing -- strengthen the setup"

    xs_sorted, _ = b.sorted_observation_arrays()
    xs_raw, _ = b.observation_arrays()
    assert np.array_equal(xs_sorted, np.sort(xs_raw))
    # Also confirms the belief's own dip-detection call site (assume_sorted=True)
    # never hit the fail-fast ValueError across this whole run.


def test_noise_std_uncertainty_unconfigured_raises_error():
    # Fresh SMC belief without noise model should raise ValueError
    b = nv_center_smc_belief(num_particles=200)
    with pytest.raises(ValueError, match="no active noise model"):
        b.noise_std_uncertainty()


def test_noise_std_uncertainty_configured():
    # SMC belief with noise model should return a valid float
    from nvision.spectra.noise_model import GaussianNoiseSignalModel

    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    b = nv_center_smc_belief(num_particles=200, noise_model=noise_model)
    b_unc = b.noise_std_uncertainty()
    assert isinstance(b_unc, float)
    assert b_unc >= 0.0


def test_noise_sigma_updates_every_update_but_dip_detection_only_upon_resampling():
    from nvision.models.observation import Observation
    from nvision.spectra.noise_model import GaussianNoiseSignalModel

    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    b = nv_center_smc_belief(num_particles=200, noise_model=noise_model)
    b.auto_resample = False

    initial_noise_std = b.estimated_noise_std()

    # 1. Update with a high-noise observation.
    # The noise estimate MUST change immediately because noise is updated on every update().
    # However, dip centers must remain empty because dip detection only runs upon resampling.
    obs = Observation(x=0.5, signal_value=0.5)
    b.update(obs)

    post_update_noise_std = b.estimated_noise_std()
    assert not np.isclose(initial_noise_std, post_update_noise_std)
    assert getattr(b, "_dip_centers", []) == []


def test_estimated_noise_std_unconfigured_raises_error():
    # Fresh SMC belief without noise model should raise ValueError
    b = nv_center_smc_belief(num_particles=200)
    with pytest.raises(ValueError, match="no active noise model"):
        b.estimated_noise_std()


def test_estimated_noise_std_configured():
    # SMC belief with noise model should return a valid float
    from nvision.spectra.noise_model import GaussianNoiseSignalModel

    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    b = nv_center_smc_belief(num_particles=200, noise_model=noise_model)
    noise_std = b.estimated_noise_std()
    assert isinstance(noise_std, float)
    assert 0.01 <= noise_std <= 0.1


def test_estimated_noise_std_is_90th_percentile():
    from nvision.spectra.noise_model import GaussianNoiseSignalModel

    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.1)})
    b = nv_center_smc_belief(num_particles=10, noise_model=noise_model)

    if getattr(b, "_use_rao_blackwell_noise", False):
        b._noise_alphas = np.full(10, 9.5, dtype=np.float32)
        sigmas = np.linspace(0.01, 0.10, 10)
        b._noise_betas = (sigmas**2 * 10.0).astype(np.float32)
        b._weights = np.full(10, 0.10, dtype=np.float32)

        est = b.estimated_noise_std()
        assert np.isclose(est, 0.09)
