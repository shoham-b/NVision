"""Unit tests for the boxcar-scan dip detector and the belief's conjugate noise estimate."""

from __future__ import annotations

import numpy as np
import pytest

from nvision import nv_center_smc_belief
from nvision.belief.dip_detection import find_dips, hwhm_fwhm_prior_hz
from tests.noise import gaussian_noise

# FWHM prior used throughout: 10 MHz +/- 2 MHz, allowed range 4-16 MHz -> windows span 6-14 MHz.
_G = (10e6, 2e6, 4e6, 16e6)
# Student-t degrees of freedom of a noise sigma known exactly (normal scores).
_KNOWN_SIGMA = np.inf


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
    assert find_dips(np.array([]), np.array([]), 0.02, _KNOWN_SIGMA, *_G) == []


def test_find_dips_no_window_fits_the_scan():
    # Every point lies within 2 MHz: no window can span the 6 MHz minimum width.
    xs, ys = np.array([2.8e9, 2.801e9, 2.802e9]), np.array([1.0, 0.5, 1.0])
    assert find_dips(xs, ys, 0.02, _KNOWN_SIGMA, *_G) == []


@pytest.mark.parametrize("clustered", [False, True])
def test_find_dips_pure_noise_has_no_false_alarms(clustered):
    # Many independent pure-noise scans: at p_fa = 0.01 the family-wise false-alarm rate is <= 1%, also
    # when 70% of the measurements pile up in one place as SBED's do (the baseline must stay unbiased).
    rng = np.random.default_rng(0)
    false_alarms = 0
    for _ in range(200):
        xs = rng.uniform(2.80e9, 2.90e9, 300)
        if clustered:
            xs[:210] = 2.85e9 + rng.uniform(-5e6, 5e6, 210)
        false_alarms += bool(find_dips(xs, 1.0 + 0.05 * rng.standard_normal(xs.size), 0.05, _KNOWN_SIGMA, *_G))
    assert false_alarms <= 6


def test_find_dips_locates_a_clean_dip():
    dips = find_dips(_DIP_XS, _DIP_YS, 0.02, _KNOWN_SIGMA, *_G)
    assert len(dips) == 1
    d = dips[0]
    assert abs(d.centroid_hz - 2.85e9) <= 1e6
    assert d.significance > 0
    assert 0.0 < d.confidence <= 1.0
    assert d.f_min in _DIP_XS
    assert d.f_max in _DIP_XS
    assert np.isclose(d.width_hz, d.f_max - d.f_min)
    assert 6e6 <= d.width_hz <= 14e6


def test_find_dips_is_shape_agnostic():
    # A flat-bottomed (box) dip and a Gaussian dip are found just like a Lorentzian.
    box = np.where(np.abs(_DIP_XS - 2.85e9) <= 4e6, 0.8, 1.0)
    gauss = 1.0 - 0.2 * np.exp(-0.5 * ((_DIP_XS - 2.85e9) / 4e6) ** 2)
    for ys in (box, gauss):
        dips = find_dips(_DIP_XS, ys, 0.02, _KNOWN_SIGMA, *_G)
        assert len(dips) == 1
        assert abs(dips[0].centroid_hz - 2.85e9) <= 1e6


def test_find_dips_detects_snr_one_dip_on_irregular_repeated_positions():
    # Depth equal to the noise sigma (SNR ~ 1 per point) -- invisible to any per-point threshold --
    # sampled irregularly with repeats, as SBED's adaptive acquisition produces. Only the boxcar sum
    # over the many points inside the window lifts it above the FWER threshold.
    rng = np.random.default_rng(3)
    xs = np.sort(np.concatenate([rng.uniform(2.80e9, 2.90e9, 1500), np.repeat(rng.uniform(2.84e9, 2.86e9, 40), 3)]))
    ys = _lorentzian_scan(xs, [2.85e9], depth=0.05) + 0.05 * rng.standard_normal(xs.size)
    dips = find_dips(xs, ys, 0.05, _KNOWN_SIGMA, *_G, assume_sorted=True)
    assert len(dips) >= 1
    assert abs(dips[0].centroid_hz - 2.85e9) < 3e6


def test_find_dips_still_detects_a_dip_holding_most_of_the_measurements():
    # SBED concentrates its measurements on a found dip: 70% of the points here sit inside it.
    rng = np.random.default_rng(5)
    xs = np.sort(np.concatenate([rng.uniform(2.80e9, 2.90e9, 90), 2.85e9 + rng.uniform(-4e6, 4e6, 210)]))
    ys = _lorentzian_scan(xs, [2.85e9], depth=0.1, fwhm=8e6) + 0.05 * rng.standard_normal(xs.size)
    dips = find_dips(xs, ys, 0.05, _KNOWN_SIGMA, *_G, assume_sorted=True)
    assert len(dips) >= 1
    assert abs(dips[0].centroid_hz - 2.85e9) < 3e6


def test_find_dips_ignores_a_constant_baseline_offset():
    a = find_dips(_DIP_XS, _DIP_YS, 0.02, _KNOWN_SIGMA, *_G)
    b = find_dips(_DIP_XS, _DIP_YS + 7.0, 0.02, _KNOWN_SIGMA, *_G)
    assert [(d.f_min, d.f_max) for d in a] == [(d.f_min, d.f_max) for d in b]
    assert np.allclose([d.significance for d in a], [d.significance for d in b])


def test_find_dips_separates_dips_split_by_baseline_and_ranks_by_significance():
    xs = np.arange(2.80e9, 2.95e9 + 1, 1e6)
    ys = _lorentzian_scan(xs, [2.83e9], depth=0.3) + _lorentzian_scan(xs, [2.91e9], depth=0.15) - 1.0
    dips = find_dips(xs, ys, 0.02, _KNOWN_SIGMA, *_G)
    assert len(dips) == 2
    assert abs(dips[0].centroid_hz - 2.83e9) <= 1e6
    assert abs(dips[1].centroid_hz - 2.91e9) <= 1e6
    assert dips[0].significance > dips[1].significance
    assert dips[0].f_max < dips[1].f_min


@pytest.mark.parametrize("snr", [5, 20, 100])
def test_find_dips_reports_one_tight_dip_however_strong(snr):
    # A deep dip's wings are significant too, and every wider window containing the dip scores above the
    # threshold; both must stay one dip whose extent hugs the line, not the widest window tested.
    xs = np.arange(2.80e9, 2.95e9 + 1, 0.5e6)
    ys = _lorentzian_scan(xs, [2.875e9], depth=0.3, fwhm=8e6)
    ys += 0.3 / snr * np.random.default_rng(1).standard_normal(xs.size)
    dips = find_dips(xs, ys, 0.3 / snr, _KNOWN_SIGMA, 7.7e6, 4.2e6, 0.4e6, 15e6)
    assert len(dips) == 1
    assert abs(dips[0].centroid_hz - 2.875e9) <= 1e6
    assert dips[0].width_hz <= 1.5 * 8e6


def test_find_dips_keeps_a_hyperfine_triplet_as_one_dip():
    # N-14 triplet: three 1 MHz-wide lines 2.16 MHz apart, each narrower than their spread.
    xs = np.arange(2.80e9, 2.95e9 + 1, 0.5e6)
    centers = [2.875e9 - 2.16e6, 2.875e9, 2.875e9 + 2.16e6]
    ys = sum(_lorentzian_scan(xs, [c], depth=0.1, fwhm=1e6) for c in centers) - 2.0
    ys += 0.01 * np.random.default_rng(2).standard_normal(xs.size)
    dips = find_dips(xs, ys, 0.01, _KNOWN_SIGMA, 1.2e6, 0.3e6, 1e6, 1.6e6)
    assert len(dips) == 1
    assert dips[0].f_min <= centers[0]
    assert dips[0].f_max >= centers[-1]


def test_find_dips_is_deterministic_and_ignores_input_order():
    order = np.random.default_rng(1).permutation(len(_DIP_XS))
    a = find_dips(_DIP_XS, _DIP_YS, 0.02, _KNOWN_SIGMA, *_G)
    b = find_dips(_DIP_XS, _DIP_YS, 0.02, _KNOWN_SIGMA, *_G)
    c = find_dips(_DIP_XS[order], _DIP_YS[order], 0.02, _KNOWN_SIGMA, *_G)
    assert a == b == c


def test_find_dips_p_fa_gate():
    # Scale the noise so the best window scores exactly 5 sigma: that passes a lenient family-wise
    # false-alarm rate but not a strict one.
    z = find_dips(_DIP_XS, _DIP_YS, 0.02, _KNOWN_SIGMA, *_G, p_fa=0.5)[0].significance
    noise = 0.02 * z / 5.0
    assert find_dips(_DIP_XS, _DIP_YS, noise, _KNOWN_SIGMA, *_G, p_fa=0.5)
    assert find_dips(_DIP_XS, _DIP_YS, noise, _KNOWN_SIGMA, *_G, p_fa=1e-12) == []


def test_find_dips_assume_sorted_matches_default():
    assert find_dips(_DIP_XS, _DIP_YS, 0.02, _KNOWN_SIGMA, *_G, assume_sorted=True) == find_dips(
        _DIP_XS, _DIP_YS, 0.02, _KNOWN_SIGMA, *_G
    )


def test_find_dips_assume_sorted_raises_on_unsorted_input():
    with pytest.raises(ValueError, match="assume_sorted"):
        find_dips(
            np.array([2.8e9, 2.9e9, 2.85e9]), np.array([1.0, 1.0, 1.0]), 0.02, _KNOWN_SIGMA, *_G, assume_sorted=True
        )


@pytest.mark.parametrize("noise_std", [0.0, -0.01, float("nan")])
def test_find_dips_rejects_a_non_positive_noise_sigma(noise_std):
    with pytest.raises(ValueError, match="noise_scale"):
        find_dips(_DIP_XS, _DIP_YS, noise_std, _KNOWN_SIGMA, *_G)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gamma_mean_hz": 0.0}, "gamma_mean_hz"),
        ({"gamma_std_hz": -1.0}, "gamma_std_hz"),
        ({"gamma_min_hz": 0.0}, "gamma_min_hz"),
        ({"gamma_min_hz": 20e6}, "gamma_min_hz"),
        ({"gamma_max_hz": 5e6}, "gamma_max_hz"),
        ({"p_fa": 1.0}, "p_fa"),
    ],
)
def test_find_dips_rejects_invalid_prior_and_p_fa(kwargs, match):
    args = {"gamma_mean_hz": 10e6, "gamma_std_hz": 2e6, "gamma_min_hz": 4e6, "gamma_max_hz": 16e6, "p_fa": None}
    with pytest.raises(ValueError, match=match):
        find_dips(_DIP_XS, _DIP_YS, 0.02, _KNOWN_SIGMA, **{**args, **kwargs})


def test_find_dips_never_tests_a_window_narrower_than_the_prior_minimum():
    # A one-point spike is best fitted by the narrowest window. gamma_mean - 2 gamma_std = 2 MHz, but with
    # the prior allowing nothing narrower than 8 MHz, no window (and so no reported dip) is narrower.
    spike = np.ones_like(_DIP_XS)
    spike[50] -= 1.0
    assert find_dips(_DIP_XS, spike, 0.02, _KNOWN_SIGMA, 10e6, 4e6, 2e6, 12e6)[0].width_hz < 8e6
    assert all(d.width_hz >= 8e6 for d in find_dips(_DIP_XS, spike, 0.02, _KNOWN_SIGMA, 10e6, 4e6, 8e6, 12e6))


def test_find_dips_reports_a_dip_wider_than_every_window_as_one_run():
    # A 30 MHz dip against windows of at most 12 MHz: overlapping significant windows chain into one run.
    ys = _lorentzian_scan(_DIP_XS, [2.85e9], depth=0.3, fwhm=30e6)
    dips = find_dips(_DIP_XS, ys, 0.02, _KNOWN_SIGMA, 10e6, 4e6, 8e6, 12e6)
    assert len(dips) == 1
    assert abs(dips[0].centroid_hz - 2.85e9) <= 2e6


def test_find_dips_rejects_non_positive_noise_dof():
    with pytest.raises(ValueError, match="noise_dof"):
        find_dips(_DIP_XS, _DIP_YS, 0.02, 0.0, *_G)


def test_find_dips_uncertain_sigma_raises_the_threshold():
    # At the largest noise scale where the dip is still detected with sigma known exactly, a sigma known
    # only to 6 degrees of freedom is not enough: its Student-t threshold is higher.
    noises = np.geomspace(0.01, 0.2, 60)
    known = [n for n in noises if find_dips(_DIP_XS, _DIP_YS, n, _KNOWN_SIGMA, *_G)]
    assert known
    assert find_dips(_DIP_XS, _DIP_YS, max(known), 6.0, *_G) == []


def test_find_dips_controls_false_alarms_when_sigma_is_estimated_from_few_samples():
    # Pure noise with sigma estimated from only 6 residuals (Inverse-Gamma posterior alpha = 3,
    # beta = sum(e^2)/2). Plugging that estimate in as if exact over-triggers whenever it comes out low;
    # the Student-t threshold (nu = 2 alpha) keeps the family-wise false-alarm rate at p_fa.
    rng = np.random.default_rng(11)
    xs = np.arange(2.80e9, 2.90e9 + 1, 1e6)
    t_alarms = plug_in_alarms = 0
    for _ in range(400):
        residuals = 0.05 * rng.standard_normal(6)
        alpha, beta = 3.0, 0.5 * float(np.sum(residuals**2))
        scale = np.sqrt(beta / alpha)
        ys = 1.0 + 0.05 * rng.standard_normal(xs.size)
        t_alarms += bool(find_dips(xs, ys, scale, 2 * alpha, *_G, p_fa=0.05))
        plug_in_alarms += bool(find_dips(xs, ys, scale, _KNOWN_SIGMA, *_G, p_fa=0.05))
    assert t_alarms <= 0.05 * 400 * 1.6
    assert plug_in_alarms > 2 * t_alarms


def test_belief_noise_student_t_matches_its_inverse_gamma_posterior():
    b = nv_center_smc_belief(num_particles=50, noise_model=gaussian_noise())
    scale, dof = b.noise_student_t()
    alpha = b._noise_alpha
    assert dof == 2 * alpha
    assert np.isclose(scale, b.estimated_noise_std() * np.sqrt((alpha + 0.5) / alpha))


def test_hwhm_fwhm_prior_hz_uses_prior_bounds():
    # Uniform HWHM on [1, 5] MHz -> FWHM mean 6 MHz, std 2 * 4 / sqrt(12) MHz, range 2-10 MHz.
    mean, std, minimum, maximum = hwhm_fwhm_prior_hz({"hwhm": (1e6, 5e6)})
    assert np.isclose(mean, 6e6)
    assert np.isclose(std, 8e6 / np.sqrt(12.0))
    assert np.isclose(minimum, 2e6)
    assert np.isclose(maximum, 10e6)
    mean, _, _, _ = hwhm_fwhm_prior_hz({"hwhm_lorentz": (1e6, 4e6)})
    assert np.isclose(mean, 5e6)
    with pytest.raises(ValueError, match="no hwhm parameter"):
        hwhm_fwhm_prior_hz({"zero_field_freq": (2.8e9, 2.9e9)})


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
    b = nv_center_smc_belief(num_particles=100, noise_model=noise_model, with_fixed_zero_field_freq=False)
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


def test_hwhm_fwhm_prior_hz_spans_the_drive_amplitude_range():
    # Pulsed drive: the pi-pulse line HWHM is 0.7988 * rabi_per_amp * amplitude. Narrowest at the
    # minimum amplitude with the weakest coupling, widest at full amplitude with the strongest.
    from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT
    from nvision.spectra.nv_center_drive import PI_PULSE_HWHM_PER_RABI

    _, _, minimum, maximum = hwhm_fwhm_prior_hz({"rabi_per_amp": (1e6, 10e6), "sigma_inhom": (0.0, 0.0)})
    assert np.isclose(minimum, 2 * PI_PULSE_HWHM_PER_RABI * 1e6 * DRIVE_AMP_MIN_UNIT)
    assert np.isclose(maximum, 2 * PI_PULSE_HWHM_PER_RABI * 10e6)
    # CW: width grows with saturation s = (a / drive_amp_sat)**2, so the largest drive_amp_sat is narrowest.
    _, _, cw_min, cw_max = hwhm_fwhm_prior_hz({"drive_amp_sat": (0.05, 0.5), "sigma_inhom": (0.0, 0.0)})
    assert 0.0 < cw_min < cw_max
