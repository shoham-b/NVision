"""Unit tests for deterministic dip detection and the belief's conjugate noise estimate."""

from __future__ import annotations

import numpy as np
import pytest

from nvision import nv_center_smc_belief
from nvision.belief.dip_detection import effective_max_linewidth_hz, find_dips
from tests.noise import gaussian_noise

# A clean scan: baseline 1.0 with one dip (three points ~0.2-0.3 below it) around 2.822 GHz.
_DIP_XS = np.array([2.8e9, 2.81e9, 2.82e9, 2.822e9, 2.824e9, 2.83e9, 2.84e9, 2.85e9, 2.86e9, 2.87e9])
_DIP_YS = np.array([1.0, 1.0, 0.8, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_find_dips_empty():
    assert find_dips(np.array([]), np.array([]), 0.02, 5e6) == []


def test_find_dips_no_dips():
    # Every point is ~1.0, within noise of the baseline: nothing sits 3 sigma below it.
    obs_xs = np.linspace(2.8e9, 2.9e9, 10)
    obs_ys = np.ones(10) + np.random.default_rng(0).uniform(-0.01, 0.01, 10)
    assert find_dips(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2) == []


def test_find_dips_single_isolated_low_point_is_filtered():
    obs_xs = np.linspace(2.8e9, 2.9e9, 10)
    obs_ys = np.ones(10)
    obs_ys[5] = 0.8  # deep, but alone: below min_cluster_count
    assert find_dips(obs_xs, obs_ys, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2) == []


def test_find_dips_cluster_centroid_is_depth_weighted():
    # 2.82/2.822/2.824 GHz are all > 3 sigma below the 1.0 baseline and within 3 * 5 MHz of each
    # other, so they form one dip; depth weights are 0.2, 0.3, 0.2.
    dips = find_dips(_DIP_XS, _DIP_YS, 0.02, 5e6, n_sigma=3.0, min_cluster_count=2)
    assert len(dips) == 1
    assert np.isclose(dips[0].centroid_hz, 2.822e9)
    assert dips[0].n_points == 3
    assert dips[0].significance == 3.0
    assert (dips[0].f_min, dips[0].f_max) == (2.82e9, 2.824e9)
    assert np.isclose(dips[0].background, 1.0)


def test_find_dips_separates_distant_dips_and_ranks_by_significance():
    xs = np.array([2.80e9, 2.802e9, 2.804e9, 2.85e9, 2.90e9, 2.902e9, 2.95e9, 2.96e9])
    ys = np.array([0.7, 0.7, 0.7, 1.0, 0.7, 0.7, 1.0, 1.0])
    dips = find_dips(xs, ys, 0.03, 5e6, n_sigma=3.0, min_cluster_count=2, confidence_threshold=0.0)
    assert [d.n_points for d in dips] == [3, 2]
    assert dips[0].centroid_hz < 2.81e9 < 2.89e9 < dips[1].centroid_hz


def test_find_dips_is_deterministic_and_ignores_input_order():
    order = np.random.default_rng(1).permutation(len(_DIP_XS))
    a = find_dips(_DIP_XS, _DIP_YS, 0.02, 5e6)
    b = find_dips(_DIP_XS, _DIP_YS, 0.02, 5e6)
    c = find_dips(_DIP_XS[order], _DIP_YS[order], 0.02, 5e6)
    assert a == b == c


def test_find_dips_binomial_confidence_gate():
    # Two low points (0.1 below the baseline, ~1 sigma at noise 0.1) among ten nearby points: at
    # a 1-sigma threshold a pair of low points is quite likely to be chance, so the 0.99 gate
    # rejects it; with no gate it is reported.
    xs = np.array([2.800e9, 2.802e9] + [2.805e9 + i * 1e6 for i in range(8)])
    ys = np.array([0.9, 0.9] + [1.0] * 8)
    kwargs = dict(noise_std=0.09, max_linewidth_hz=5e6, n_sigma=1.0, min_cluster_count=2)
    assert find_dips(xs, ys, confidence_threshold=0.99, **kwargs) == []
    assert len(find_dips(xs, ys, confidence_threshold=0.0, **kwargs)) == 1


def test_find_dips_assume_sorted_matches_default():
    order = np.argsort(_DIP_XS)
    xs, ys = _DIP_XS[order], _DIP_YS[order]
    assert find_dips(xs, ys, 0.02, 5e6, assume_sorted=True) == find_dips(xs, ys, 0.02, 5e6)


def test_find_dips_assume_sorted_raises_on_unsorted_input():
    with pytest.raises(ValueError, match="assume_sorted"):
        find_dips(np.array([2.8e9, 2.9e9, 2.85e9]), np.array([1.0, 1.0, 1.0]), 0.02, 5e6, assume_sorted=True)


@pytest.mark.parametrize("noise_std", [0.0, -0.01, float("nan")])
def test_find_dips_rejects_a_non_positive_noise_sigma(noise_std):
    with pytest.raises(ValueError, match="noise_std"):
        find_dips(_DIP_XS, _DIP_YS, noise_std, 5e6)


def test_find_dips_uncertainty_gating():
    # noise_std_unc / noise_std = 20% >= the 15% default threshold: the noise level is too poorly
    # known to threshold against, so nothing is reported. At 4% the dip is found.
    assert find_dips(_DIP_XS, _DIP_YS, 0.05, 5e6, noise_std_unc=0.01) == []
    dips = find_dips(_DIP_XS, _DIP_YS, 0.05, 5e6, noise_std_unc=0.002)
    assert len(dips) == 1
    assert np.isclose(dips[0].centroid_hz, 2.822e9)


def test_effective_max_linewidth_hz_uses_prior_bounds():
    assert effective_max_linewidth_hz({"linewidth": (1e6, 5e6)}) == 5e6
    assert effective_max_linewidth_hz({"homogeneous_linewidth": (1e6, 4e6)}) == 4e6
    with pytest.raises(ValueError, match="no linewidth parameter"):
        effective_max_linewidth_hz({"frequency": (2.8e9, 2.9e9)})


def test_sorted_observation_arrays_matches_full_sort():
    # _append_observation() doesn't touch _obs_sort_order (maintained lazily by
    # sorted_observation_arrays()), so exercise it directly without going
    # through the full model/likelihood update path.
    b = nv_center_smc_belief(num_particles=50, noise_model=gaussian_noise())
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
    b = nv_center_smc_belief(num_particles=50, noise_model=gaussian_noise())
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
    b = nv_center_smc_belief(num_particles=50, noise_model=gaussian_noise())
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
    b = nv_center_smc_belief(num_particles=50, noise_model=gaussian_noise())
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


def test_belief_requires_a_noise_model():
    """The noise level is always inferred through the conjugate prior, so a belief cannot exist without one."""
    with pytest.raises(ValueError, match="requires a noise model"):
        nv_center_smc_belief(num_particles=50, noise_model=None)


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
    assert b.dip_candidates == []


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

    b._noise_alphas = np.full(10, 9.5, dtype=np.float32)
    sigmas = np.linspace(0.01, 0.10, 10)
    b._noise_betas = (sigmas**2 * 10.0).astype(np.float32)
    b._weights = np.full(10, 0.10, dtype=np.float32)

    est = b.estimated_noise_std()
    assert np.isclose(est, 0.09)
