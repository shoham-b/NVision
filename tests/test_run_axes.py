"""Dip detection, I_max and the axes are read off the belief snapshots by one function (nvision/metrics/run_axes.py)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from nvision.metrics import run_axes
from nvision.metrics.run_axes import belief_prior_variance_phys, compute_run_axes
from nvision.spectra.drive import DRIVE_AMP_MIN_UNIT, DriveSetting
from nvision.spectra.nv_center_drive import pi_pulse_duration_s, rabi_transition_probability


class _Belief:
    def __init__(self, primary_unc: float, has_dip: bool, estimate: float = 1.0):
        self._unc = primary_unc
        self._has_dip = has_dip
        self._estimate = estimate
        self.physical_param_bounds = {"zeeman_shift": (0.0, 60e6), "hwhm": (0.1e6, 1e6)}
        self.drive_freq_bounds_phys = (0.0, 100e6)

    def uncertainty(self):
        return {"zeeman_shift": self._unc}

    def estimates(self):
        return {"zeeman_shift": self._estimate}

    def current_dips(self, p_fa=None):
        return [object()] if self._has_dip else []


def _run(uncs: list[float], dip_from: int | None, shots: list[int] | None = None) -> SimpleNamespace:
    """A run whose step i has uncertainty uncs[i] and shows a dip from step index ``dip_from`` on."""
    shots = shots or [1] * len(uncs)
    snaps = [
        SimpleNamespace(
            belief=_Belief(u, dip_from is not None and i >= dip_from), obs=SimpleNamespace(n_shots=k, noise_std=1.0)
        )
        for i, (u, k) in enumerate(zip(uncs, shots, strict=True))
    ]
    true_signal = SimpleNamespace(
        parameter_values=lambda: {"zeeman_shift": 1.0, "hwhm": 1.0},
        get_param_value=lambda name: 1.0,
        model=SimpleNamespace(controls_drive=False),
    )
    return SimpleNamespace(snapshots=snaps, true_signal=true_signal)


@pytest.fixture(autouse=True)
def _uniform_prior_on_mock_bounds(monkeypatch):
    """The mock belief is neither an SMC nor a grid belief: give it the uniform prior over its own bounds."""

    def uniform_var(belief, name):
        lo, hi = belief.physical_param_bounds[name]
        return (hi - lo) ** 2 / 12.0

    monkeypatch.setattr(run_axes, "belief_prior_variance_phys", uniform_var)
    monkeypatch.setattr(run_axes, "belief_prior_kind", lambda belief, name: "uniform")


def _with_eta(monkeypatch, eta: list[float]) -> None:
    """Fake the sensitivity of the true signal: ``eta`` per step against a best g^2 of 1."""
    sens = run_axes._Sensitivity(np.array(eta), 1.0)
    monkeypatch.setattr(run_axes, "_primary_sensitivity", lambda run, primary: sens)


def _eta(run: SimpleNamespace, primary: str) -> np.ndarray:
    """Per-step efficiency g_t^2 / max g^2 of a run, shape (n_steps,)."""
    sens = run_axes._primary_sensitivity(run, primary)
    return sens.measured / sens.best


def test_the_dip_marker_needs_the_dip_to_be_held(monkeypatch):
    uncs = [1e6] * 8
    _with_eta(monkeypatch, [0.0] * 8)
    run = _run(uncs, dip_from=0)
    for i in (1, 4):  # the dip disappears at indices 1 and 4
        run.snapshots[i].belief._has_dip = False
    m = compute_run_axes(run)
    assert m["detection_step"] == 6  # first window of 3 consecutive dip steps starts at index 5


def test_the_dip_marker_uses_the_locators_own_false_alarm_level(monkeypatch):
    seen: list[float | None] = []
    _with_eta(monkeypatch, [0.0] * 4)
    run = _run([1e6] * 4, dip_from=0)
    for snap in run.snapshots:
        snap.belief.current_dips = lambda p_fa=None: seen.append(p_fa) or [object()]
    compute_run_axes(run)
    assert seen
    assert set(seen) == {None}  # None = the default NVISION_DIP_P_FA


def test_empty_run_yields_no_stages():
    m = compute_run_axes(SimpleNamespace(snapshots=[], true_signal=None))
    assert all(v is None for v in m.values())


def test_model_without_primary_param_yields_no_stages():
    # Fixed center_freq, no Zeeman splitting: neither a split nor center_freq is a parameter.
    run = _run([1e6] * 3, dip_from=1)
    run.true_signal = SimpleNamespace(parameter_values=lambda: {"linewidth": 1.0, "c_total": 1.0})
    m = compute_run_axes(run)
    assert all(v is None for v in m.values())


class _GaussDipModel:
    """signal(x) = 1 - exp(-(x - c)^2 / 2) with one parameter ``c``; sensitivity peaks on the slopes, 0 at c."""

    controls_drive = False
    spec = SimpleNamespace(
        pack_params=lambda p: [p],
        unpack_params=lambda v: v[0],
    )

    def compute_many_float64(self, xs, c):
        return 1.0 - np.exp(-((xs - c) ** 2) / 2.0)


def test_sensitivity_efficiency_is_one_on_the_slope_and_zero_at_the_dip_centre():
    model = _GaussDipModel()
    true_signal = SimpleNamespace(
        parameter_names=("c",),
        get_param_bounds=lambda name: (-5.0, 5.0),
        model=model,
        typed_parameters=0.0,
        model_at=lambda drive: model,
        get_param_value=lambda name: 0.0,
    )
    belief = SimpleNamespace(drive_freq_bounds_phys=(-10.0, 10.0))

    def snap(x_phys: float):
        return SimpleNamespace(belief=belief, obs=SimpleNamespace(drive=None, drive_freq_unit=(x_phys + 10) / 20))

    run = SimpleNamespace(true_signal=true_signal, snapshots=[snap(0.0), snap(1.0), snap(-1.0), snap(9.0)])
    eta = _eta(run, "c")
    assert eta[0] == pytest.approx(0.0, abs=1e-6)  # dip centre
    assert eta[1] == pytest.approx(1.0, abs=1e-3)  # steepest point of exp(-x^2/2) slope is at |x| = 1
    assert eta[2] == pytest.approx(1.0, abs=1e-3)
    assert eta[3] < 1e-6  # far baseline


class _LorentzDipModel(_GaussDipModel):
    """signal(x) = 1 - 1 / (1 + (x - c)^2): HWHM 1, steepest slope at |x - c| = 1 / sqrt(3)."""

    def compute_many_float64(self, xs, c):
        return 1.0 - 1.0 / (1.0 + (xs - c) ** 2)


def test_lorentzian_efficiency_loss_is_second_order_near_the_optimum():
    # Missing the optimum by eps costs ~4.5 eps^2 (eps in HWHM): the locator need only be *near* it.
    model = _LorentzDipModel()
    true_signal = SimpleNamespace(
        parameter_names=("c",),
        get_param_bounds=lambda name: (-5.0, 5.0),
        model=model,
        typed_parameters=0.0,
        model_at=lambda drive: model,
    )
    belief = SimpleNamespace(drive_freq_bounds_phys=(-10.0, 10.0))
    u0 = 1 / np.sqrt(3)
    xs = [u0, u0 + 0.1, -(u0 - 0.1)]
    snaps = [SimpleNamespace(belief=belief, obs=SimpleNamespace(drive=None, drive_freq_unit=(x + 10) / 20)) for x in xs]
    eta = _eta(SimpleNamespace(true_signal=true_signal, snapshots=snaps), "c")
    assert eta[0] == pytest.approx(1.0, abs=1e-4)
    assert eta[1] == pytest.approx(1 - 4.5 * 0.1**2, abs=0.01)
    assert eta[2] == pytest.approx(1 - 4.5 * 0.1**2, abs=0.01)


class _PulsedRabiModel:
    """Pulsed model with one parameter ``c`` (resonance, Hz): the spin-flip probability of a square pulse."""

    controls_drive = True
    is_pulsed = True
    spec = SimpleNamespace(pack_params=lambda p: [p], unpack_params=lambda v: v[0])

    def __init__(self, rabi_hz: float, duration_s: float | None = None):
        self._rabi_hz = rabi_hz
        self._duration_s = duration_s

    def compute_many_float64(self, xs, c):
        return rabi_transition_probability(xs - c, self._rabi_hz, self._duration_s)


def test_oracle_includes_pulses_longer_than_the_pi_pulse():
    # A pulse 5.6x the pi-pulse is far more sensitive to the resonance than the pi-pulse (Ramsey-like fringes).
    # A locator with a mis-calibrated Rabi frequency can pick one; the oracle must not be beaten by it.
    rabi_per_amp = 5e6
    amp = DRIVE_AMP_MIN_UNIT
    t_long = 10**0.75 * pi_pulse_duration_s(rabi_per_amp * amp)
    model = _PulsedRabiModel(rabi_per_amp * amp)
    x_lo, x_hi = -30e6, 30e6

    def model_at(drive):
        return _PulsedRabiModel(rabi_per_amp * drive.drive_amp_unit, drive.pulse_duration_s)

    true_signal = SimpleNamespace(
        parameter_names=("c",),
        get_param_bounds=lambda name: (-30e6, 30e6),
        model=model,
        typed_parameters=0.0,
        model_at=model_at,
        get_param_value=lambda name: rabi_per_amp,
    )
    belief = SimpleNamespace(drive_freq_bounds_phys=(x_lo, x_hi))
    xs = np.linspace(x_lo, x_hi, 40001)
    h = 6e3
    long_model = model_at(DriveSetting(amp, t_long))
    g2 = ((long_model.compute_many_float64(xs, h) - long_model.compute_many_float64(xs, -h)) / (2 * h)) ** 2
    x_best = float(xs[np.argmax(g2)])
    drive = DriveSetting(amp, t_long)
    snap = SimpleNamespace(
        belief=belief, obs=SimpleNamespace(drive=drive, drive_freq_unit=(x_best - x_lo) / (x_hi - x_lo))
    )
    eta = _eta(SimpleNamespace(true_signal=true_signal, snapshots=[snap]), "c")
    assert 0.0 < eta[0] <= 1.0  # before the fix: ValueError "A measurement beat the grid optimum"


def test_fisher_star_and_axes(monkeypatch):
    # With best g^2 = 1, sigma = 1 and one shot per step, I_max = 1 per step.
    _with_eta(monkeypatch, [0.0] * 6 + [1.0] * 4)
    m = compute_run_axes(_run([2e6] * 10, dip_from=1))
    assert m["i_max"] == pytest.approx(1.0)
    assert m["fwhm_phys"] == pytest.approx(2.0)  # 2 * hwhm of the fake truth
    assert m["window_phys"] == pytest.approx(100e6)
    assert m["window_over_fwhm"] == pytest.approx(50e6)
    assert m["sigma0_phys"] == pytest.approx(60e6 / math.sqrt(12.0))  # W / sqrt(12) of the uniform zeeman_shift prior
    assert m["prior_kind"] == "uniform"
    assert set(m) == set(run_axes.RUN_AXIS_KEYS)


def test_uniform_prior_variance_is_w2_over_12():
    assert run_axes.prior_variance_phys(None, (10.0, 16.0)) == pytest.approx(36.0 / 12.0)


def test_gaussian_prior_variance_matches_the_sampler():
    """Truncated N(mu, std) on the unit cube mixed with a uniform share, scaled to Hz: as the belief draws it."""
    from nvision.belief.smc_marginal import _sample_truncated_normal

    rng = np.random.default_rng(0)
    n, broad, lo, hi = 400_000, 0.1, 2.0, 12.0
    draws = _sample_truncated_normal(0.1, 0.3, 0.0, 1.0, n, rng=rng)
    idx = rng.choice(n, size=round(broad * n), replace=False)
    draws[idx] = rng.uniform(0.0, 1.0, idx.size)
    expected = float(np.var(lo + draws * (hi - lo)))
    assert run_axes.prior_variance_phys((0.1, 0.3), (lo, hi), broad) == pytest.approx(expected, rel=1e-2)


def test_sin2_prior_variance_matches_quadrature():
    from scipy.integrate import quad

    lo, hi, k = 100.0, 130.0, np.pi / 7.0
    norm = quad(lambda f: np.sin(k * (f - lo)) ** 2, lo, hi, limit=200)[0]
    mean = quad(lambda f: f * np.sin(k * (f - lo)) ** 2, lo, hi, limit=200)[0] / norm
    var = quad(lambda f: (f - mean) ** 2 * np.sin(k * (f - lo)) ** 2, lo, hi, limit=200)[0] / norm
    assert run_axes.prior_variance_phys(("sin^2", k), (lo, hi)) == pytest.approx(var, rel=1e-9)


@pytest.mark.parametrize(
    ("entry", "bounds", "match"),
    [
        ((0.5, 0.0), (0.0, 1.0), "std > 0"),  # no spread
        ((-40.0, 1.0), (0.0, 1.0), "far-tail"),  # sampler's far-tail exponential branch
        (("beta", 2.0, 3.0), (0.0, 1.0), "Unsupported prior"),  # unknown shape
        (None, (1.0, 1.0), "hi > lo"),  # empty interval
    ],
)
def test_prior_variance_fails_fast_on_unsupported_priors(entry, bounds, match):
    with pytest.raises(ValueError, match=match):
        run_axes.prior_variance_phys(entry, bounds)


def test_smc_belief_prior_is_uniform_on_zeeman_shift():
    """The real belief: no prior entry on zeeman_shift, so J0 = 12 / W^2 of its bounds, after any measurement."""
    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
    from tests.noise import gaussian_noise

    belief = nv_center_smc_belief(
        {"zero_field_freq": (2.80e9, 2.94e9), "zeeman_shift": (1e6, 41e6)},
        num_particles=200,
        noise_model=gaussian_noise(),
        seed=0,
    )
    assert "zeeman_shift" not in (belief.priors or {})
    lo, hi = belief.physical_param_bounds["zeeman_shift"]
    assert (lo, hi) == (1e6, 41e6)
    assert belief_prior_variance_phys(belief, "zeeman_shift") == pytest.approx((hi - lo) ** 2 / 12.0)
    with pytest.raises(TypeError):
        belief_prior_variance_phys(SimpleNamespace(), "zeeman_shift")


def test_true_error_first_passage(monkeypatch):
    _with_eta(monkeypatch, [1.0] * 10)
    monkeypatch.setattr("nvision.metrics.series.effective_convergence_threshold", lambda run, param: 0.5)
    run = _run([2.0] * 10, dip_from=1)
    for snap, err in zip(run.snapshots, [4.0] * 3 + [0.4] * 7, strict=True):  # truth is 1.0
        snap.belief._estimate = 1.0 + err
    m = compute_run_axes(run)
    assert m["target_cost_step"] == 4  # error <= tau = 0.5 from the 4th measurement on
