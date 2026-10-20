"""Unit tests for the matched-filter dip detector and the belief's conjugate noise estimate."""

from __future__ import annotations

import numpy as np
import pytest

from nvision import nv_center_smc_belief
from nvision.belief.dip_detection import find_dips, linewidth_fwhm_prior_hz
from tests.noise import gaussian_noise

# FWHM prior used throughout: 10 MHz +/- 2 MHz, allowed range 4-16 MHz.
_G_MEAN, _G_STD, _G_MIN, _G_MAX = 10e6, 2e6, 4e6, 16e6


def _lorentzian_scan(xs, centers, depth=1.0, fwhm=10e6):
    """Baseline 1.0 minus a Lorentzian dip of the given depth (and FWHM) at each centre. shape: like xs."""
    hw = 0.5 * fwhm
    ys = np.ones_like(xs, dtype=float)
    for c in centers:
        ys -= depth * hw**2 / ((xs - c) ** 2 + hw**2)
    return ys


# Clean uniform sweep, 1 MHz step, one dip at 2.85 GHz.
_DIP_XS = np.arange(2.80e9, 2.90e9 + 1, 1e6)
_DIP_YS = _lorentzian_scan(_DIP_XS, [2.85e9], depth=0.2)


def test_find_dips_empty():
    assert find_dips(np.array([]), np.array([]), 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX) == []


def test_find_dips_too_few_positions():
    xs, ys = np.array([2.8e9, 2.8e9, 2.81e9]), np.array([1.0, 0.5, 1.0])
    assert find_dips(xs, ys, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX) == []


def test_find_dips_pure_noise_has_no_false_alarms():
    # Many independent pure-noise scans: at p_fa = 0.01 the family-wise false alarm rate is <= 1%.
    rng = np.random.default_rng(0)
    xs = np.arange(2.80e9, 2.90e9 + 1, 0.5e6)
    false_alarms = sum(
        bool(find_dips(xs, 1.0 + 0.05 * rng.standard_normal(xs.size), 0.05, _G_MEAN, _G_STD, _G_MIN, _G_MAX))
        for _ in range(200)
    )
    assert false_alarms <= 6


def test_find_dips_locates_a_clean_dip():
    dips = find_dips(_DIP_XS, _DIP_YS, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX)
    assert len(dips) == 1
    d = dips[0]
    assert d.centroid_hz == 2.85e9
    assert d.significance > 0
    assert 0.0 < d.confidence <= 1.0
    assert np.isclose(d.f_max - d.f_min, 3.0 * d.fwhm_hz)
    assert np.isclose(0.5 * (d.f_min + d.f_max), d.centroid_hz)
    assert _G_MEAN - 2 * _G_STD <= d.fwhm_hz <= _G_MEAN + 2 * _G_STD


def test_find_dips_detects_snr_one_dip_on_irregular_repeated_positions():
    # Depth equal to the noise sigma (SNR ~ 1 per point) -- invisible to any per-point threshold --
    # sampled irregularly with repeats, as SBED's adaptive acquisition produces. Only the matched
    # filter's coherent sum over the ~500 points in the window lifts it above the FWER threshold.
    rng = np.random.default_rng(3)
    xs = np.sort(np.concatenate([rng.uniform(2.80e9, 2.90e9, 1500), np.repeat(rng.uniform(2.84e9, 2.86e9, 40), 3)]))
    ys = _lorentzian_scan(xs, [2.85e9], depth=0.05) + 0.05 * rng.standard_normal(xs.size)
    dips = find_dips(xs, ys, 0.05, _G_MEAN, _G_STD, _G_MIN, _G_MAX, assume_sorted=True)
    assert len(dips) >= 1
    assert abs(dips[0].centroid_hz - 2.85e9) < 3e6


def test_find_dips_ignores_a_constant_baseline_offset():
    # The zero-sum template annihilates the local baseline: shifting every value changes nothing.
    a = find_dips(_DIP_XS, _DIP_YS, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX)
    b = find_dips(_DIP_XS, _DIP_YS + 7.0, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX)
    assert [d.centroid_hz for d in a] == [d.centroid_hz for d in b]
    assert np.allclose([d.significance for d in a], [d.significance for d in b])


def test_find_dips_peels_separated_dips_and_ranks_by_significance():
    xs = np.arange(2.80e9, 2.95e9 + 1, 1e6)
    ys = 1.0 - (1.0 - _lorentzian_scan(xs, [2.83e9], depth=0.3)) - (1.0 - _lorentzian_scan(xs, [2.91e9], depth=0.15))
    dips = find_dips(xs, ys, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX)
    assert [d.centroid_hz for d in dips] == [2.83e9, 2.91e9]
    assert dips[0].significance > dips[1].significance


def test_find_dips_is_deterministic_and_ignores_input_order():
    order = np.random.default_rng(1).permutation(len(_DIP_XS))
    a = find_dips(_DIP_XS, _DIP_YS, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX)
    b = find_dips(_DIP_XS, _DIP_YS, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX)
    c = find_dips(_DIP_XS[order], _DIP_YS[order], 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX)
    assert a == b == c


def test_find_dips_p_fa_gate():
    # A shallow dip: its z-score passes a lenient family-wise false-alarm rate but not a strict one.
    ys = _lorentzian_scan(_DIP_XS, [2.85e9], depth=0.02)
    z = find_dips(_DIP_XS, ys, 0.01, _G_MEAN, _G_STD, _G_MIN, _G_MAX, p_fa=0.5)[0].significance
    assert find_dips(_DIP_XS, ys, 0.01 * z / 4.0, _G_MEAN, _G_STD, _G_MIN, _G_MAX, p_fa=0.5)
    assert find_dips(_DIP_XS, ys, 0.01 * z / 4.0, _G_MEAN, _G_STD, _G_MIN, _G_MAX, p_fa=1e-12) == []


def test_find_dips_assume_sorted_matches_default():
    assert find_dips(_DIP_XS, _DIP_YS, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX, assume_sorted=True) == find_dips(
        _DIP_XS, _DIP_YS, 0.02, _G_MEAN, _G_STD, _G_MIN, _G_MAX
    )


def test_find_dips_assume_sorted_raises_on_unsorted_input():
    with pytest.raises(ValueError, match="assume_sorted"):
        find_dips(
            np.array([2.8e9, 2.9e9, 2.85e9]),
            np.array([1.0, 1.0, 1.0]),
            0.02,
            _G_MEAN,
            _G_STD,
            _G_MIN,
            _G_MAX,
            assume_sorted=True,
        )


@pytest.mark.parametrize("noise_std", [0.0, -0.01, float("nan")])
def test_find_dips_rejects_a_non_positive_noise_sigma(noise_std):
    with pytest.raises(ValueError, match="noise_std"):
        find_dips(_DIP_XS, _DIP_YS, noise_std, _G_MEAN, _G_STD, _G_MIN, _G_MAX)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gamma_mean_hz": 0.0}, "gamma_mean_hz"),
        ({"gamma_std_hz": -1.0}, "gamma_std_hz"),
        ({"gamma_min_hz": 0.0}, "gamma_min_hz"),
        ({"gamma_min_hz": 2 * _G_MEAN}, "gamma_min_hz"),
        ({"gamma_max_hz": 0.5 * _G_MEAN}, "gamma_max_hz"),
        ({"p_fa": 1.0}, "p_fa"),
    ],
)
def test_find_dips_rejects_invalid_prior_and_p_fa(kwargs, match):
    args = {
        "gamma_mean_hz": _G_MEAN,
        "gamma_std_hz": _G_STD,
        "gamma_min_hz": _G_MIN,
        "gamma_max_hz": _G_MAX,
        "p_fa": None,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        find_dips(_DIP_XS, _DIP_YS, 0.02, **args)


def test_find_dips_never_tests_a_template_narrower_than_the_prior_minimum():
    # A one-point spike is best matched by the narrowest template. With gamma_mean - 2 gamma_std = 2 MHz
    # that would be 2 MHz, but the prior allows no dip narrower than 8 MHz, so no 2 MHz template is tested.
    ys = np.ones_like(_DIP_XS)
    ys[50] -= 0.5
    unfloored = find_dips(_DIP_XS, ys, 0.02, 10e6, 4e6, 2e6, 30e6)
    assert np.isclose(unfloored[0].fwhm_hz, 2e6)
    floored = find_dips(_DIP_XS, ys, 0.02, 10e6, 4e6, 8e6, 30e6)
    assert all(d.fwhm_hz >= 8e6 for d in floored)


def test_find_dips_never_tests_a_template_wider_than_the_prior_maximum():
    # A broad 30 MHz dip is best matched by the widest template. gamma_mean + 2 gamma_std = 18 MHz, but
    # the prior allows no dip wider than 12 MHz, so no template wider than that is tested.
    xs = np.arange(2.70e9, 3.00e9 + 1, 1e6)
    ys = _lorentzian_scan(xs, [2.85e9], depth=0.3, fwhm=30e6)
    assert np.isclose(find_dips(xs, ys, 0.02, 10e6, 4e6, 2e6, 30e6)[0].fwhm_hz, 18e6)
    assert all(d.fwhm_hz <= 12e6 for d in find_dips(xs, ys, 0.02, 10e6, 4e6, 2e6, 12e6))


def test_find_dips_uncertainty_gating():
    # noise_std_unc / noise_std = 20% >= the 15% default threshold: the noise level is too poorly
    # known to normalise z-scores, so nothing is reported. At 4% the dip is found.
    assert find_dips(_DIP_XS, _DIP_YS, 0.05, _G_MEAN, _G_STD, _G_MIN, _G_MAX, noise_std_unc=0.01) == []
    dips = find_dips(_DIP_XS, _DIP_YS, 0.05, _G_MEAN, _G_STD, _G_MIN, _G_MAX, noise_std_unc=0.002)
    assert len(dips) == 1
    assert dips[0].centroid_hz == 2.85e9


def test_linewidth_fwhm_prior_hz_uses_prior_bounds():
    # Uniform HWHM on [1, 5] MHz -> FWHM mean 6 MHz, std 2 * 4 / sqrt(12) MHz, range 2-10 MHz.
    mean, std, minimum, maximum = linewidth_fwhm_prior_hz({"linewidth": (1e6, 5e6)})
    assert np.isclose(mean, 6e6)
    assert np.isclose(std, 8e6 / np.sqrt(12.0))
    assert np.isclose(minimum, 2e6)
    assert np.isclose(maximum, 10e6)
    mean, _, _, _ = linewidth_fwhm_prior_hz({"homogeneous_linewidth": (1e6, 4e6)})
    assert np.isclose(mean, 5e6)
    with pytest.raises(ValueError, match="no linewidth parameter"):
        linewidth_fwhm_prior_hz({"center_freq": (2.8e9, 2.9e9)})


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


def test_belief_sorted_observations_stay_consistent_across_updates_and_resamples():
    # End-to-end: many updates with forced resamples (each triggers dip detection via
    # _generate_epoch_candidate_drive_freq -> sorted_observation_arrays with assume_sorted=True),
    # so the fail-fast "not ascending" ValueError would surface if sorting ever drifted.
    from nvision.models.observation import Observation
    from nvision.spectra.noise_model import GaussianNoiseSignalModel

    noise_model = GaussianNoiseSignalModel(prior_bounds={"noise_sigma": (0.01, 0.05)})
    b = nv_center_smc_belief(num_particles=100, noise_model=noise_model, with_fixed_center_freq=False)
    axis_before = (b.drive_freq_bounds_phys, dict(b.physical_param_bounds))

    rng = np.random.default_rng(123)
    for i in range(60):
        b.update(Observation(drive_freq_unit=float(rng.uniform(0.0, 1.0)), signal_value=float(rng.uniform(0.5, 1.0))))
        if i % 10 == 9:
            b._resample()

    assert (b.drive_freq_bounds_phys, dict(b.physical_param_bounds)) == axis_before
    xs_sorted, _ = b.sorted_observation_arrays()
    xs_raw, _ = b.observation_arrays()
    assert np.array_equal(xs_sorted, np.sort(xs_raw))


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
    obs = Observation(drive_freq_unit=0.5, signal_value=0.5)
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

    b._noise_alpha = 9.5
    sigmas = np.linspace(0.01, 0.10, 10)
    b._noise_betas = (sigmas**2 * 10.0).astype(np.float32)
    b._weights = np.full(10, 0.10, dtype=np.float32)

    est = b.estimated_noise_std()
    assert np.isclose(est, 0.09)


def test_linewidth_fwhm_prior_hz_spans_the_drive_amplitude_range():
    # Pulsed drive: the pi-pulse line HWHM is 0.7988 * rabi_per_amp * amplitude. Narrowest at the
    # minimum amplitude with the weakest coupling, widest at full amplitude with the strongest.
    from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT
    from nvision.spectra.nv_center_drive import PI_PULSE_HWHM_PER_RABI

    _, _, minimum, maximum = linewidth_fwhm_prior_hz({"rabi_per_amp": (1e6, 10e6), "sigma_inhom": (0.0, 0.0)})
    assert np.isclose(minimum, 2 * PI_PULSE_HWHM_PER_RABI * 1e6 * DRIVE_AMP_MIN_UNIT)
    assert np.isclose(maximum, 2 * PI_PULSE_HWHM_PER_RABI * 10e6)
    # CW: width grows with saturation s = (a / drive_amp_sat)**2, so the largest drive_amp_sat is narrowest.
    _, _, cw_min, cw_max = linewidth_fwhm_prior_hz({"drive_amp_sat": (0.05, 0.5), "sigma_inhom": (0.0, 0.0)})
    assert 0.0 < cw_min < cw_max
