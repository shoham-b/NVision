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
        self.physical_param_bounds = {"zeeman_shift": (0.0, 60e6)}

    def uncertainty(self):
        return {"zeeman_shift": self._unc}

    def estimates(self):
        return {"zeeman_shift": self._estimate}

    def current_dips(self):
        return [object()] if self._has_dip else []


def _run(uncs: list[float], dip_from: int | None, shots: list[int] | None = None) -> SimpleNamespace:
    """A run whose step i has uncertainty uncs[i] and shows a dip from step index ``dip_from`` on."""
    shots = shots or [1] * len(uncs)
    snaps = [
        SimpleNamespace(belief=_Belief(u, dip_from is not None and i >= dip_from), obs=SimpleNamespace(n_shots=k))
        for i, (u, k) in enumerate(zip(uncs, shots, strict=True))
    ]
    true_signal = SimpleNamespace(
        parameter_values=lambda: {"zeeman_shift": 1.0, "hwhm": 1.0}, get_param_value=lambda name: 1.0
    )
    return SimpleNamespace(snapshots=snaps, true_signal=true_signal)


def _with_eta(monkeypatch, eta: list[float]) -> None:
    monkeypatch.setattr(stages, "sensitivity_efficiency_trace", lambda run, primary: np.array(eta))


def test_all_three_stages_reached(monkeypatch):
    uncs = [10e6, 8e6, 6e6, 1e6, 500e3, 200e3, 90e3, 80e3, 70e3, 60e3]
    # Dip revealed by measurement 3. Tail means from index 4: 0.67, from 5: 0.8, from 6: 0.95 >= 0.9 -> refine at 6.
    _with_eta(monkeypatch, [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 0.9, 1.0, 0.9])
    m = compute_stage_metrics(_run(uncs, dip_from=3))
    assert m["focus_found_step"] == 4
    assert m["max_info_step"] == 7
    assert (m["stage1_steps"], m["stage2_steps"], m["stage3_steps"]) == (4, 2, 4)
    assert m["stage1_unc_start"] == 10e6
    assert m["stage2_unc_start"] == 1e6  # belief after the focus measurement
    assert m["stage2_unc_end"] == 200e3
    assert m["stage3_unc_start"] == 200e3  # belief before the first refine measurement
    assert m["stage3_unc_end"] == 60e3
    assert m["stage2_unc_ratio"] == pytest.approx(0.2)
    assert m["stage1_sens_eff"] == pytest.approx(0.0)
    assert m["stage2_sens_eff"] == pytest.approx(0.1)
    assert m["stage3_sens_eff"] == pytest.approx(0.95)
    assert m["stage_primary_param"] == "zeeman_shift"


def test_error_at_the_end_of_each_stage(monkeypatch):
    uncs = [1e6] * 10
    _with_eta(monkeypatch, [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 0.9, 1.0, 0.9])
    run = _run(uncs, dip_from=3)
    # Estimate error (truth is 1.0) after measurement i: 10 - i, so stages end after indices 3, 5 and 9.
    for i, snap in enumerate(run.snapshots):
        snap.belief._estimate = 1.0 + (10 - i)
    m = compute_stage_metrics(run)
    assert (m["stage1_err_end"], m["stage2_err_end"], m["stage3_err_end"]) == (7.0, 5.0, 1.0)


def test_one_stray_measurement_does_not_end_refine(monkeypatch):
    # A single exploratory probe (eta 0) among ten optimal steps keeps the tail mean at 10/11 >= 0.9.
    eta = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    _with_eta(monkeypatch, eta)
    m = compute_stage_metrics(_run([1e6] * len(eta), dip_from=1))
    assert m["max_info_step"] == 3  # refine starts with the measurement after the focus one


def test_steady_but_suboptimal_sampling_is_not_refine(monkeypatch):
    # The same information every step, but only half the CRLB-optimal amount: not stage 3.
    _with_eta(monkeypatch, [0.5] * 10)
    m = compute_stage_metrics(_run([10e3] * 10, dip_from=1))
    assert m["focus_found_step"] == 2
    assert m["max_info_step"] is None
    assert m["stage3_steps"] is None


def test_shots_weight_the_information(monkeypatch):
    # From index 1: one 10-shot step at eta 0.5 and five 1-shot steps at 1.0 -> (5 + 5) / 15 < 0.9.
    _with_eta(monkeypatch, [0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
    m = compute_stage_metrics(_run([1e6] * 7, dip_from=0, shots=[1, 10, 1, 1, 1, 1, 1]))
    assert m["max_info_step"] == 3


def test_never_finding_a_dip_leaves_only_the_search_stage(monkeypatch):
    _with_eta(monkeypatch, [1.0] * 3)
    m = compute_stage_metrics(_run([10e6, 9e6, 8e6], dip_from=None))
    assert m["focus_found_step"] is None
    assert m["max_info_step"] is None
    assert m["stage1_steps"] == 3
    assert m["stage2_steps"] is None
    assert m["stage3_steps"] is None


def test_refine_starting_at_the_focus_step_leaves_empty_focus_stage(monkeypatch):
    _with_eta(monkeypatch, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    m = compute_stage_metrics(_run([1e6] * 6, dip_from=2))
    assert m["focus_found_step"] == 3
    assert m["max_info_step"] == 4
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
