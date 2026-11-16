"""Run-time posterior metrics (nvision/metrics/posterior.py): KDE Fisher information, negentropy, far mass."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from nvision.metrics.posterior import (
    POSTERIOR_METRIC_KEYS,
    POSTERIOR_SERIES_KEYS,
    compute_posterior_metrics,
    posterior_density,
)

N = 1000


def _mixture_fisher(means, sigmas, weights, lo, hi, n=400_001):
    """Exact J = int p'^2 / p of a Gaussian mixture by dense quadrature of its analytic pdf."""
    x = np.linspace(lo, hi, n)
    p = np.zeros_like(x)
    dp = np.zeros_like(x)
    for m, s, w in zip(means, sigmas, weights, strict=True):
        g = w * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        p += g
        dp += -(x - m) / s**2 * g
    keep = p > 1e-12 * p.max()
    return float(np.sum(dp[keep] ** 2 / p[keep]) * (x[1] - x[0]))


def test_gaussian_posterior_fisher_is_inverse_variance():
    rng = np.random.default_rng(0)
    sigma = 3e3
    x = 50e6 + sigma * rng.standard_normal(N)
    d = posterior_density(x, np.full(N, 1.0 / N), 0.0, 100e6)
    # The KDE of N draws is N(mu, sigma^2 + h^2): J = 1 / (sigma^2 + h^2), a few % below 1 / sigma^2.
    j = d.fisher_information()
    assert j == pytest.approx(1.0 / sigma**2, rel=0.25)
    assert j == pytest.approx(1.0 / (np.var(x) + d.bandwidth_phys**2), rel=0.25)


def test_two_mirror_peaks_fisher_matches_the_mixture_not_the_cloud_variance():
    rng = np.random.default_rng(1)
    sep, peak = 4e6, 0.1e6  # peaks far narrower than their separation
    centres = np.array([20e6, 20e6 + sep])
    x = np.concatenate([c + peak * rng.standard_normal(N // 2) for c in centres])
    d = posterior_density(x, np.full(N, 1.0 / N), 0.0, 100e6)
    sigma_hat = np.sqrt(np.var(x))
    exact = _mixture_fisher(centres, [peak, peak], [0.5, 0.5], 0.0, 100e6)
    j = d.fisher_information()
    assert j > 20.0 / sigma_hat**2  # >> 1 / sigma_hat^2
    assert j == pytest.approx(exact, rel=0.3)  # a global (Silverman) bandwidth would be ~8x low here
    assert d.bandwidth_phys < peak


def test_negentropy_zero_for_gaussian_positive_for_mixture():
    rng = np.random.default_rng(2)
    g = posterior_density(10e6 + 1e5 * rng.standard_normal(N), np.full(N, 1.0 / N), 0.0, 20e6)
    assert g.negentropy() == pytest.approx(0.0, abs=0.05)
    x = np.concatenate([5e6 + 1e4 * rng.standard_normal(N // 2), 15e6 + 1e4 * rng.standard_normal(N // 2)])
    m = posterior_density(x, np.full(N, 1.0 / N), 0.0, 20e6)
    assert m.negentropy() > 0.5


def test_reflection_keeps_mass_at_the_bound():
    rng = np.random.default_rng(3)
    x = np.abs(rng.standard_normal(N)) * 1e5  # half-normal piled against lo = 0
    d = posterior_density(x, np.full(N, 1.0 / N), 0.0, 10e6)
    assert float(np.sum(d.pdf) * d.dx) == pytest.approx(1.0, abs=1e-6)  # nothing leaked past the edge
    assert d.grid_phys.min() >= 0.0


def test_weights_are_respected():
    rng = np.random.default_rng(4)
    x = np.concatenate([5e6 + 1e4 * rng.standard_normal(N), 15e6 + 1e4 * rng.standard_normal(N)])
    w = np.concatenate([np.full(N, 0.9 / N), np.full(N, 0.1 / N)])
    assert posterior_density(x, w, 0.0, 20e6).mode() == pytest.approx(5e6, abs=5e4)


def test_density_fails_fast_on_bad_input():
    with pytest.raises(ValueError, match="outside the bounds"):
        posterior_density(np.array([1.0, 2.0, 30.0]), np.ones(3), 0.0, 10.0)
    with pytest.raises(ValueError, match="no spread"):
        posterior_density(np.full(5, 1.0), np.ones(5), 0.0, 10.0)
    with pytest.raises(ValueError, match="non-negative"):
        posterior_density(np.array([1.0, 2.0]), np.array([1.0, -1.0]), 0.0, 10.0)


# --- run-level metrics on a fake run ---------------------------------------------------------------------------


class _Belief:
    """Particle belief stand-in: a fixed cloud of zeeman_shift (Hz) and a claimed sigma."""

    physical_param_bounds: ClassVar = {"zeeman_shift": (0.0, 100e6)}
    drive_freq_bounds_phys = (2.0e9, 2.1e9)

    def __init__(self, x, sigma=None, num_observations=1):
        self.num_observations = num_observations
        self._x = x
        self._w = np.full(x.size, 1.0 / x.size)
        self._sigma = float(np.sqrt(np.var(x))) if sigma is None else sigma

    def particles_phys(self):
        return {"zeeman_shift": self._x}

    def weights(self):
        return self._w

    def estimates(self):
        return {"zeeman_shift": float(np.mean(self._x))}

    def uncertainty(self):
        return {"zeeman_shift": self._sigma}


def _fake_run(clouds, drive_unit, truth=30e6, zff=2.0e9, n_obs=None):
    signal = SimpleNamespace(
        get_param_value=lambda name: {"zeeman_shift": truth, "zero_field_freq": zff}[name],
        parameter_values=lambda: {"zeeman_shift": truth, "zero_field_freq": zff, "hwhm": 0.5e6},
        model=SimpleNamespace(controls_drive=False),
    )
    snaps = [
        SimpleNamespace(
            belief=_Belief(c, num_observations=1 if n_obs is None else n_obs[i]), obs=SimpleNamespace(drive_freq_unit=u)
        )
        for i, (c, u) in enumerate(zip(clouds, drive_unit, strict=True))
    ]
    return SimpleNamespace(snapshots=snaps, true_signal=signal)


def test_run_metrics_z_far_mass_far_placement_and_j_tau(monkeypatch):
    # FWHM 2 MHz -> far = 6 MHz from the mode. dip drive position = zff + mode.
    monkeypatch.setattr("nvision.metrics.stages._true_fwhm_phys", lambda ts: 2e6)
    rng = np.random.default_rng(5)
    wide = 30e6 + 8e6 * rng.standard_normal(N)
    mirror = np.concatenate([20e6 + 1e5 * rng.standard_normal(N // 2), 40e6 + 1e5 * rng.standard_normal(N // 2)])
    tight = 30e6 + 1e5 * rng.standard_normal(N)
    # step 1 measured at the dip of snapshot 0's mode?  mode of `wide` ~ 30 MHz -> drive 2.03 GHz -> unit 0.3
    run = _fake_run([wide, mirror, tight], drive_unit=[0.0, 0.3, 0.9])
    out = compute_posterior_metrics(run, "zeeman_shift", {"dip_detected_step": 2})
    s, ser = out["scalars"], out["series"]
    assert set(s) == set(POSTERIOR_METRIC_KEYS)
    assert set(ser) == set(POSTERIOR_SERIES_KEYS)
    assert all(len(v) == 3 for v in ser.values())
    # z = (mean - truth) / sigma_hat; the mirror cloud's mean is 30 MHz -> z ~ 0
    assert ser["z"][1] == pytest.approx(0.0, abs=0.05)
    assert ser["x"][0] is None  # no belief before the first measurement
    assert ser["x"][1] == 0.0  # measured on the previous mode
    assert ser["x"][2] == 1.0  # 0.9 -> 2.09 GHz, ~60 MHz from the previous belief's dip
    assert s["posterior_far_placed_steps"] == 1
    assert s["posterior_far_placed_share"] == pytest.approx(0.5)
    # J_tau is the information of the posterior at tau = 2 (the two mirror peaks): >> 1 / sigma_hat^2
    sigma_hat = float(np.sqrt(np.var(mirror)))
    assert s["posterior_J_tau_phys"] > 20.0 / sigma_hat**2
    assert s["posterior_J_tau_phys"] == ser["j"][1]
    # far mass of the mirror cloud: half of it sits 20 MHz (> 6 MHz) from the mode
    assert s["posterior_far_mass_tau"] == pytest.approx(0.5, abs=0.05)
    assert ser["m"][2] == pytest.approx(0.0, abs=1e-9)


def test_run_metrics_none_without_primary_particles_or_detection(monkeypatch):
    monkeypatch.setattr("nvision.metrics.stages._true_fwhm_phys", lambda ts: 2e6)
    rng = np.random.default_rng(6)
    run = _fake_run([30e6 + 1e5 * rng.standard_normal(N)], drive_unit=[0.1])
    s = compute_posterior_metrics(run, "zeeman_shift", {"dip_detected_step": None})["scalars"]
    assert s["posterior_J_tau_phys"] is None
    empty = SimpleNamespace(snapshots=[], true_signal=None)
    assert compute_posterior_metrics(empty, "zeeman_shift", {})["scalars"]["posterior_J_tau_phys"] is None
    grid = SimpleNamespace(snapshots=[SimpleNamespace(belief=object())], true_signal=None)
    assert compute_posterior_metrics(grid, "zeeman_shift", {})["scalars"]["posterior_J_tau_phys"] is None


def test_mirror_peaks_with_a_diffuse_background_and_a_far_straggler():
    """The posterior right after detection also holds some diffuse mass (the tail the far-mass metric counts)."""
    rng = np.random.default_rng(7)
    peak, sep, bg_std, bg_share = 0.1e6, 4e6, 15e6, 0.15
    n_bg = int(bg_share * N)
    n_peak = (N - n_bg) // 2
    bg_centre = 50e6
    x = np.concatenate(
        [
            20e6 + peak * rng.standard_normal(n_peak),
            20e6 + sep + peak * rng.standard_normal(n_peak),
            bg_centre + bg_std * rng.standard_normal(n_bg),
        ]
    )
    w = np.full(x.size, 1.0 / x.size)
    exact = _mixture_fisher(
        [20e6, 20e6 + sep, bg_centre],
        [peak, peak, bg_std],
        [n_peak / x.size, n_peak / x.size, n_bg / x.size],
        0.0,
        100e6,
    )
    j = posterior_density(x, w, 0.0, 100e6).fisher_information()
    assert j == pytest.approx(exact, rel=0.3)
    # One far particle of weight 5e-4 (the span jumps to ~100 MHz) must not move J by much.
    x2 = np.append(x, 99e6)
    w2 = np.append(w, 5e-4)
    j2 = posterior_density(x2, w2, 0.0, 100e6).fisher_information()
    assert j2 == pytest.approx(j, rel=0.3)


def test_belief_without_measurements_is_undefined_not_a_crash_and_is_counted(monkeypatch):
    """A sweep's belief is the flat prior until its first flush: no posterior, so no metrics for that step."""
    monkeypatch.setattr("nvision.metrics.stages._true_fwhm_phys", lambda ts: 2e6)
    flat = np.linspace(0.0, 100e6, N)
    peaked = 30e6 + 1e5 * np.random.default_rng(8).standard_normal(N)
    run = _fake_run([flat, peaked, peaked + 1.0], drive_unit=[0.0, 0.3, 0.3], n_obs=[0, 1, 2])
    out = compute_posterior_metrics(run, "zeeman_shift", {"dip_detected_step": 1})
    assert out["scalars"]["posterior_undefined_steps"] == 1
    assert out["series"]["j"][0] is None
    assert out["series"]["m"][0] is None
    assert out["series"]["z"][0] is not None  # z needs only the estimate and sigma
    assert out["series"]["j"][1] is not None
    assert out["series"]["x"][1] is None  # the previous belief is undefined: no mode to compare with
    assert out["series"]["x"][2] is not None
    assert out["scalars"]["posterior_J_tau_phys"] is None  # tau = 1 is the undefined step


def test_density_is_computed_only_at_the_kept_steps_and_tau(monkeypatch):
    monkeypatch.setattr("nvision.metrics.stages._true_fwhm_phys", lambda ts: 2e6)
    from nvision.metrics.series import MAX_SERIES_POINTS, _downsample_indices

    rng = np.random.default_rng(9)
    n = 200
    clouds = [30e6 + 1e5 * rng.standard_normal(N) for _ in range(n)]
    run = _fake_run(clouds, drive_unit=[0.3] * n)
    out = compute_posterior_metrics(run, "zeeman_shift", {"dip_detected_step": 100})
    expected = set(_downsample_indices(n, MAX_SERIES_POINTS)) | {99}
    assert set(out["density_steps"]) == expected
    assert {t for t, v in enumerate(out["series"]["j"]) if v is not None} == expected
    assert all(v is not None for v in out["series"]["z"])  # the cheap per-step series cover every step
    assert all(v is not None for v in out["series"]["m"])
    assert out["scalars"]["posterior_J_tau_phys"] == out["series"]["j"][99]


def _weighted_draws(target_pdf, centre, proposal_sd, seed):
    """Importance-weighted sample of ``target_pdf`` from a broad Gaussian proposal (weights degenerate: low ESS)."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    x = centre + proposal_sd * rng.standard_normal(N)
    w = target_pdf(x) / norm.pdf(x, centre, proposal_sd)
    w /= w.sum()
    return x, w, 1.0 / float(np.sum(w**2))


def test_low_ess_weighted_clouds_do_not_inflate_j():
    """The posterior before a resample has degenerate weights (ESS ~ 100-500 of 1000): J must still be right."""
    from scipy.stats import norm

    sigma = 3e3
    js, neffs = [], []
    for seed in range(5):
        x, w, neff = _weighted_draws(lambda v: norm.pdf(v, 50e6, sigma), 50e6, 3 * sigma, seed)
        js.append(posterior_density(x, w, 0.0, 100e6).fisher_information())
        neffs.append(neff)
    assert 100 < np.mean(neffs) < 700
    assert np.mean(js) == pytest.approx(1.0 / sigma**2, rel=0.2)

    peak, sep = 0.1e6, 4e6

    def mix(v):
        return 0.5 * norm.pdf(v, 20e6, peak) + 0.5 * norm.pdf(v, 20e6 + sep, peak)

    js, neffs = [], []
    for seed in range(5):
        x, w, neff = _weighted_draws(mix, 22e6, 2.5e6, seed)
        js.append(posterior_density(x, w, 0.0, 100e6).fisher_information())
        neffs.append(neff)
    assert np.mean(neffs) < 300
    # ESS ~ 90 leaves ~45 particles per peak, so J comes out low, never high: undersmoothing would inflate it.
    assert 0.6 / peak**2 < np.mean(js) < 1.2 / peak**2


def test_j_tau_bracket_and_h_series(monkeypatch):
    monkeypatch.setattr("nvision.metrics.stages._true_fwhm_phys", lambda ts: 2e6)
    rng = np.random.default_rng(10)
    cloud = 30e6 + 1e5 * rng.standard_normal(N)
    run = _fake_run([cloud, cloud + 1.0], drive_unit=[0.3, 0.3])
    out = compute_posterior_metrics(run, "zeeman_shift", {"dip_detected_step": 2})
    s = out["scalars"]
    assert s["posterior_J_tau_2h_phys"] < s["posterior_J_tau_phys"]  # more smoothing, less information
    assert out["series"]["h"][1] == s["posterior_kde_bandwidth_tau_phys"]


def test_structureless_cloud_after_measurements_is_undefined_not_an_abort(monkeypatch):
    """A flat cloud after measurements (posterior still the prior) has no ISJ root: record it, do not abort."""
    monkeypatch.setattr("nvision.metrics.stages._true_fwhm_phys", lambda ts: 2e6)
    flat = np.linspace(0.0, 100e6, N)
    peaked = 30e6 + 1e5 * np.random.default_rng(11).standard_normal(N)
    run = _fake_run([flat, peaked], drive_unit=[0.3, 0.3])  # n_obs = 1 for both
    out = compute_posterior_metrics(run, "zeeman_shift", {"dip_detected_step": 1})
    assert out["series"]["j"][0] is None
    assert out["series"]["j"][1] is not None
    assert out["scalars"]["posterior_undefined_steps"] == 1
    assert out["scalars"]["posterior_J_tau_phys"] is None
    assert out["scalars"]["posterior_J_tau_2h_phys"] is None


def test_attach_is_a_no_op_when_nothing_was_computed_and_raises_on_misalignment():
    from nvision.metrics.series import attach_posterior_series

    empty = compute_posterior_metrics(SimpleNamespace(snapshots=[], true_signal=None), "zeeman_shift", {})
    series = {"s": [1, 2, 3]}
    attach_posterior_series(series, empty)  # e.g. a real-data run: no primary, nothing attached, no error
    assert set(series) == {"s"}
    no_primary = compute_posterior_metrics(
        SimpleNamespace(snapshots=[SimpleNamespace(belief=object())], true_signal=None), None, {}
    )
    attach_posterior_series(series, no_primary)
    assert set(series) == {"s"}
    run = _fake_run([30e6 + 1e5 * np.random.default_rng(12).standard_normal(N) for _ in range(3)], [0.3] * 3)
    done = compute_posterior_metrics(run, "zeeman_shift", {})
    with pytest.raises(ValueError, match="not built for this run"):
        attach_posterior_series({"s": [1, 2, 9]}, done)
