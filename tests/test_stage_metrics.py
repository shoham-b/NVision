"""Stage boundaries (search -> focus -> refine) are read off the belief snapshots by one function."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nvision.metrics import stages
from nvision.metrics.stages import compute_stage_metrics, sensitivity_efficiency_trace
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


def _with_eta(monkeypatch, eta: list[float], eps_est: float = 2e6, eps_true: float = 0.1) -> None:
    """Fake the sensitivity of the true signal (``eta``, ``eps_true``) and the tolerance of the estimated line."""
    sens = stages._Sensitivity(np.array(eta), 1.0, eps_true)
    monkeypatch.setattr(stages, "_primary_sensitivity", lambda run, primary: sens)
    monkeypatch.setattr(stages, "_estimated_tolerance_phys", lambda run, primary: eps_est)


def test_all_three_stages_reached(monkeypatch):
    uncs = [10e6, 8e6, 7.5e6, 1e6, 500e3, 200e3, 90e3, 80e3, 70e3, 60e3]
    # Dip shown from index 3 on (held for 3 steps) -> focus at index 3. The claimed sigma is <= eps_est = 100 kHz from
    # index 6 on (held 6..8) -> refine starts with the next measurement, index 7.
    _with_eta(monkeypatch, [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 1.0, 0.9, 1.0], eps_est=100e3)
    m = compute_stage_metrics(_run(uncs, dip_from=3))
    assert m["focus_found_step"] == 4  # sigma <= 0.7 of 10e6 from index 3 on
    assert m["dip_detected_step"] == 4  # the evidence marker is separate (here it happens to coincide)
    assert m["refine_start_step"] == 8
    assert (m["stage1_steps"], m["stage2_steps"], m["stage3_steps"]) == (4, 3, 3)
    assert m["stage1_unc_start"] == 10e6
    assert m["stage2_unc_start"] == 1e6  # belief after the focus measurement
    assert m["stage2_unc_end"] == 90e3
    assert m["stage3_unc_start"] == 90e3  # belief before the first refine measurement
    assert m["stage3_unc_end"] == 60e3
    assert m["stage2_unc_ratio"] == pytest.approx(0.09)
    assert m["stage1_sens_eff"] == pytest.approx(0.0)
    assert m["stage2_sens_eff"] == pytest.approx(0.2)
    assert m["stage3_sens_eff"] == pytest.approx(2.9 / 3)
    assert m["stage_primary_param"] == "zeeman_shift"


def test_error_at_the_end_of_each_stage(monkeypatch):
    uncs = [10e6, 8e6, 7.5e6, 1e6, 500e3, 200e3, 90e3, 80e3, 70e3, 60e3]
    _with_eta(monkeypatch, [0.0] * 10, eps_est=100e3)
    run = _run(uncs, dip_from=3)
    # Estimate error (truth is 1.0) after measurement i: 10 - i, so stages end after indices 3, 6 and 9.
    for i, snap in enumerate(run.snapshots):
        snap.belief._estimate = 1.0 + (10 - i)
    m = compute_stage_metrics(run)
    assert (m["stage1_err_end"], m["stage2_err_end"], m["stage3_err_end"]) == (7.0, 4.0, 1.0)


def test_an_isolated_drop_in_sigma_starts_neither_focus_nor_refine(monkeypatch):
    # Sigma touches the focus level (0.7 of 1e6) and the tolerance at index 2 only; each must hold for 3 steps.
    uncs = [1e6, 1e6, 50e3, 1e6, 1e6, 1e6, 50e3, 40e3, 30e3, 20e3]
    _with_eta(monkeypatch, [0.0] * len(uncs), eps_est=100e3)
    m = compute_stage_metrics(_run(uncs, dip_from=None))
    assert m["focus_found_step"] == 7  # held at 6..8
    assert m["refine_start_step"] == 8


def test_the_dip_marker_needs_the_dip_to_be_held(monkeypatch):
    uncs = [1e6] * 8
    _with_eta(monkeypatch, [0.0] * 8)
    run = _run(uncs, dip_from=0)
    for i in (1, 4):  # the dip disappears at indices 1 and 4
        run.snapshots[i].belief._has_dip = False
    m = compute_stage_metrics(run)
    assert m["dip_detected_step"] == 6  # first window of 3 consecutive dip steps starts at index 5
    assert m["focus_found_step"] is None  # sigma never fell, so there is no stage boundary


def test_the_dip_marker_uses_the_locators_own_false_alarm_level(monkeypatch):
    seen: list[float | None] = []
    _with_eta(monkeypatch, [0.0] * 4)
    run = _run([1e6] * 4, dip_from=0)
    for snap in run.snapshots:
        snap.belief.current_dips = lambda p_fa=None: seen.append(p_fa) or [object()]
    compute_stage_metrics(run)
    assert seen
    assert set(seen) == {None}  # None = the default NVISION_DIP_P_FA


def test_boundaries_ignore_the_measurement_efficiency(monkeypatch):
    # Every measurement at half the optimal information: refine is still reached, since the boundary is the precision.
    _with_eta(monkeypatch, [0.5] * 10)
    m = compute_stage_metrics(_run([10e3] + [4e3] * 9, dip_from=1))
    assert m["focus_found_step"] == 2
    assert m["refine_start_step"] == 3
    assert m["stage3_sens_eff"] == pytest.approx(0.5)
    assert m["stage3_wasted_share"] == pytest.approx(1.0)


def test_refine_is_not_reached_while_the_claimed_sigma_stays_above_the_tolerance(monkeypatch):
    _with_eta(monkeypatch, [1.0] * 10, eps_est=1e3)
    m = compute_stage_metrics(_run([10e3] + [4e3] * 9, dip_from=1))
    assert m["focus_found_step"] == 2
    assert m["refine_start_step"] is None
    assert m["stage3_steps"] is None


def test_refine_needs_the_tolerance_held_inside_the_run(monkeypatch):
    # Below the tolerance for only the last two steps: not held for 3 steps, so refine is not reached.
    uncs = [1e6] + [400e3] * 7 + [50e3] * 2
    _with_eta(monkeypatch, [0.0] * 10, eps_est=100e3)
    assert compute_stage_metrics(_run(uncs, dip_from=0))["refine_start_step"] is None
    # Held for the last three: refine is the last two measurements (indices 8, 9).
    uncs = [1e6] + [400e3] * 6 + [50e3] * 3
    m = compute_stage_metrics(_run(uncs, dip_from=0))
    assert m["refine_start_step"] == 9
    assert m["stage3_steps"] == 2


def test_drive_controlled_models_have_no_refine_boundary(monkeypatch):
    _with_eta(monkeypatch, [1.0] * 6)
    monkeypatch.setattr(stages, "_estimated_tolerance_phys", lambda run, primary: None)
    m = compute_stage_metrics(_run([10e3] + [4e3] * 5, dip_from=1))
    assert m["focus_found_step"] == 2
    assert m["refine_start_step"] is None


def test_never_finding_a_dip_leaves_only_the_search_stage(monkeypatch):
    _with_eta(monkeypatch, [1.0] * 3)
    m = compute_stage_metrics(_run([10e6, 9e6, 8e6], dip_from=None))
    assert m["focus_found_step"] is None
    assert m["refine_start_step"] is None
    assert m["stage1_steps"] == 3
    assert m["stage2_steps"] is None
    assert m["stage3_steps"] is None


def test_refine_starting_at_the_focus_step_leaves_empty_focus_stage(monkeypatch):
    uncs = [1e6, 1e6, 50e3, 50e3, 50e3, 50e3]
    _with_eta(monkeypatch, [0.0] * 6, eps_est=100e3)
    m = compute_stage_metrics(_run(uncs, dip_from=2))
    assert m["focus_found_step"] == 3
    assert m["refine_start_step"] == 4
    assert (m["stage1_steps"], m["stage2_steps"], m["stage3_steps"]) == (3, 0, 3)
    assert m["stage2_sens_eff"] is None


def test_empty_run_yields_no_stages():
    m = compute_stage_metrics(SimpleNamespace(snapshots=[], true_signal=None))
    assert all(v is None for v in m.values())


def test_model_without_primary_param_yields_no_stages():
    # Fixed center_freq, no Zeeman splitting: neither a split nor center_freq is a parameter.
    run = _run([1e6] * 3, dip_from=1)
    run.true_signal = SimpleNamespace(parameter_values=lambda: {"linewidth": 1.0, "c_total": 1.0})
    m = compute_stage_metrics(run)
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
    eta = sensitivity_efficiency_trace(run, "c")
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
    eta = sensitivity_efficiency_trace(SimpleNamespace(true_signal=true_signal, snapshots=snaps), "c")
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
    eta = sensitivity_efficiency_trace(SimpleNamespace(true_signal=true_signal, snapshots=[snap]), "c")
    assert 0.0 < eta[0] <= 1.0  # before the fix: ValueError "A measurement beat the grid optimum"


def test_refine_series_is_aligned_to_refine_start_with_fisher_reference(monkeypatch):
    from nvision.metrics.series import extract_refine_series

    class _B:
        def __init__(self, unc: float, est: float):
            self._unc, self._est = unc, est

        def uncertainty(self):
            return {"zeeman_shift": self._unc}

        def estimates(self):
            return {"zeeman_shift": self._est}

    uncs = [9.0, 4.0, 2.0, 1.0, 0.5]
    snaps = [SimpleNamespace(belief=_B(u, 10.0 + u)) for u in uncs]
    true_signal = SimpleNamespace(get_param_value=lambda name: 10.0)
    run = SimpleNamespace(snapshots=snaps, true_signal=true_signal)
    monkeypatch.setattr(stages, "fisher_information_trace", lambda r, p: np.array([0.0, 0.0, 3.0, 5.0, 7.0]))

    out = extract_refine_series(run, "zeeman_shift", refine_step=3)  # first refine measurement: index 2

    assert out["s0"] == 3
    assert out["u"] == [4.0, 2.0, 1.0, 0.5]  # belief before the first refine measurement, then after each
    assert out["e"] == [4.0, 2.0, 1.0, 0.5]
    # Fisher reference: u0 = 4 improved only by the info gained since: 1/16 + 3, + 5, + 7 (Hz^-2).
    expected = [(1 / 16 + c) ** -0.5 for c in (0.0, 3.0, 8.0, 15.0)]
    assert out["f"] == pytest.approx(expected, rel=1e-3)
    with pytest.raises(ValueError, match="outside the run"):
        extract_refine_series(run, "zeeman_shift", refine_step=99)


def test_optimum_tolerance_of_a_lorentzian_is_015_hwhm():
    # eta >= 0.9 holds within sqrt(0.1 / 4.5) = 0.149 HWHM of the steepest-slope point.
    model = _LorentzDipModel()
    true_signal = SimpleNamespace(
        parameter_names=("c",),
        get_param_bounds=lambda name: (-5.0, 5.0),
        model=model,
        typed_parameters=0.0,
        model_at=lambda drive: model,
    )
    belief = SimpleNamespace(drive_freq_bounds_phys=(-10.0, 10.0))
    snap = SimpleNamespace(belief=belief, obs=SimpleNamespace(drive=None, drive_freq_unit=0.5))
    sens = stages._primary_sensitivity(SimpleNamespace(true_signal=true_signal, snapshots=[snap]), "c")
    assert sens.eps_phys == pytest.approx(np.sqrt(0.1 / 4.5), rel=0.05)


def test_oracle_steps_and_efficiency(monkeypatch):
    # With best g^2 = 1, sigma = 1 and one shot per step, I* = 1 per step; eps = 0.1 Hz -> 1 / (eps^2 I*) = 100 steps.
    _with_eta(monkeypatch, [0.0] * 6 + [1.0] * 4, eps_est=500e3)
    # sigma <= 0.7 of 2e6 from index 2 (focus), <= eps_est from index 5 (refine starts with index 6)
    uncs = [2e6] * 2 + [900e3] * 3 + [100e3] * 5
    m = compute_stage_metrics(_run(uncs, dip_from=1))
    assert m["stage_fisher_star"] == pytest.approx(1.0)
    assert m["stage_eps_target_phys"] == pytest.approx(0.1)
    assert m["stage2_oracle_steps"] == pytest.approx(100.0)
    assert m["stage2_steps"] == 3
    assert m["stage2_efficiency"] == pytest.approx(100.0 / 3)
    assert m["stage2_wasted_share"] == pytest.approx(1.0)  # every focus measurement had eta 0
    assert m["stage3_wasted_share"] == pytest.approx(0.0)
    assert m["stage_fwhm_true_phys"] == pytest.approx(2.0)  # 2 * hwhm of the fake truth
    assert m["stage_window_phys"] == pytest.approx(100e6)
    assert m["stage_window_fwhm"] == pytest.approx(50e6)


def test_refine_oracle_is_the_fisher_steps_from_eps_to_tau(monkeypatch):
    _with_eta(monkeypatch, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], eps_est=20.0, eps_true=2.0)
    monkeypatch.setattr("nvision.metrics.series.effective_convergence_threshold", lambda run, param: 0.5)
    m = compute_stage_metrics(_run([10.0] + [4.0] * 5, dip_from=1))
    # Refine starts at precision eps = 2, tau = 0.5, I* = 1: (0.5^-2 - 2^-2) / 1 = 3.75 steps against 4 spent.
    assert m["stage3_oracle_steps"] == pytest.approx(3.75)
    assert m["stage3_efficiency"] == pytest.approx(3.75 / m["stage3_steps"])


def test_estimated_tolerance_is_the_tolerance_of_the_final_estimate():
    # A Lorentzian of HWHM 1 at estimate c = 2: the tolerance is sqrt(0.1 / 4.5) HWHM wherever c is.
    model = _LorentzDipModel()

    def sensitivity_sq(primary, xs):
        return ((model.compute_many_float64(xs, 2.0 + 1e-4) - model.compute_many_float64(xs, 2.0 - 1e-4)) / 2e-4) ** 2

    belief = SimpleNamespace(drive_freq_bounds_phys=(-10.0, 10.0), primary_sensitivity_sq_phys=sensitivity_sq)
    run = SimpleNamespace(
        true_signal=SimpleNamespace(model=SimpleNamespace(controls_drive=False)),
        snapshots=[SimpleNamespace(belief=belief)],
    )
    assert stages._estimated_tolerance_phys(run, "c") == pytest.approx(np.sqrt(0.1 / 4.5), rel=0.05)


def _estimated_run(g2: np.ndarray) -> SimpleNamespace:
    belief = SimpleNamespace(
        drive_freq_bounds_phys=(0.0, 1.0), primary_sensitivity_sq_phys=lambda primary, grid_phys: g2
    )
    return SimpleNamespace(
        true_signal=SimpleNamespace(model=SimpleNamespace(controls_drive=False)),
        snapshots=[SimpleNamespace(belief=belief)],
    )


def test_unresolvable_estimated_line_means_refine_is_not_reached(monkeypatch):
    n = stages._EFFICIENCY_GRID_POINTS
    one_cell = np.zeros(n)
    one_cell[n // 2] = 1.0  # a line narrower than one grid cell
    assert stages._estimated_tolerance_phys(_estimated_run(one_cell), "zeeman_shift") is None
    assert stages._estimated_tolerance_phys(_estimated_run(np.zeros(n)), "zeeman_shift") is None


def test_unresolvable_true_line_still_raises():
    g2 = np.zeros(stages._EFFICIENCY_GRID_POINTS)
    g2[10] = 1.0
    with pytest.raises(stages.UnresolvedLineError):
        stages._optimum_tolerance_phys(np.linspace(0.0, 1.0, g2.size), g2, 1.0)
