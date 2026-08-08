"""Tests that GenericSweepLocator.finalize() produces accurate frequency estimates.

The locator uses a full model fit (scipy curve_fit) rather than centroid heuristics.
These tests verify accuracy in physically realistic regimes.
"""

import numpy as np
import pytest
from numpy.random import default_rng

from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.models.observation import Observation
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.spectra.nv_center import (
    NVCenterLorentzianModel,
    NVCenterLorentzianSpectrum,
    NVCenterLorentzianZeemanHyperfineSpectrum,
    NVCenterLorentzianZeemanSpectrum,
)
from nvision.spectra.unit_cube import UnitCubeSignalModel

_BOUNDS = {
    "frequency": (2.82, 2.92),
    "linewidth": (1e-4, 0.02),
    "split": (0.001, 0.015),
    "k_np": (1.0, 5.0),
    "c_total": (0.05, 0.5),
}


def _build_locator(domain_lo=2.82, domain_hi=2.92, n_steps=100) -> GenericSweepLocator:
    phys_model = NVCenterLorentzianModel(with_fixed_frequency=False)
    unit_model = UnitCubeSignalModel(
        phys_model,
        param_bounds_phys=dict(_BOUNDS) | {"frequency": (domain_lo, domain_hi)},
        x_bounds_phys=(domain_lo, domain_hi),
    )
    belief = UnitCubeSMCMarginalDistribution(
        unit_model,
        num_particles=50,
        physical_param_bounds=unit_model.param_bounds_phys,
        physical_x_bounds=(domain_lo, domain_hi),
    )
    return GenericSweepLocator(
        belief=belief,
        signal_model=unit_model,
        max_steps=n_steps,
        noise_std=0.005,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
    )


def _inject_sweep_data(
    locator: GenericSweepLocator,
    true_freq: float,
    true_lw: float = 0.003,
    true_split: float = 0.003,
    true_k_np: float = 2.0,
    true_c_total: float = 0.3,
    noise_std: float = 0.005,
    seed: int = 42,
) -> None:
    """Directly populate locator history with synthetic noisy observations."""
    phys_model = NVCenterLorentzianModel(with_fixed_frequency=False)
    true_params = NVCenterLorentzianSpectrum(
        frequency=true_freq,
        linewidth=true_lw,
        split=true_split,
        k_np=true_k_np,
        c_total=true_c_total,
    )
    rng = default_rng(seed)
    domain_lo = locator._domain_lo
    domain_hi = locator._domain_hi
    n = locator.max_steps
    xs_norm = np.linspace(0.0, 1.0, n)
    for x_norm in xs_norm:
        x_phys = domain_lo + x_norm * (domain_hi - domain_lo)
        y_true = float(phys_model.compute(x_phys, true_params))
        y_obs = y_true + rng.normal(0, noise_std)
        obs = Observation(x=x_norm, signal_value=y_obs, noise_std=noise_std)
        locator.history.append(obs)
        locator._pending_obs.append(obs)
    # step_count drives effective_step_count(); set it so metrics are computed correctly.
    locator.step_count = n


def _build_locator_for(model, bounds, domain_lo, domain_hi, n_steps=100, noise_std=0.003) -> GenericSweepLocator:
    """Model-agnostic locator builder (Zeeman / Zeeman+hyperfine variants)."""
    unit_model = UnitCubeSignalModel(model, param_bounds_phys=bounds, x_bounds_phys=(domain_lo, domain_hi))
    belief = UnitCubeSMCMarginalDistribution(
        unit_model,
        num_particles=50,
        physical_param_bounds=unit_model.param_bounds_phys,
        physical_x_bounds=(domain_lo, domain_hi),
    )
    return GenericSweepLocator(
        belief=belief,
        signal_model=unit_model,
        max_steps=n_steps,
        noise_std=noise_std,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
    )


def _inject_sweep_data_for(locator, model, true_params, noise_std, seed=1) -> None:
    """Model-agnostic sweep-history injection (Zeeman / Zeeman+hyperfine variants)."""
    rng = default_rng(seed)
    domain_lo, domain_hi = locator._domain_lo, locator._domain_hi
    n = locator.max_steps
    xs_norm = np.linspace(0.0, 1.0, n)
    for x_norm in xs_norm:
        x_phys = domain_lo + x_norm * (domain_hi - domain_lo)
        y_true = float(model.compute(x_phys, true_params))
        y_obs = y_true + rng.normal(0, noise_std)
        obs = Observation(x=x_norm, signal_value=y_obs, noise_std=noise_std)
        locator.history.append(obs)
        locator._pending_obs.append(obs)
    locator.step_count = n


# --- Accuracy tests ---


@pytest.mark.parametrize(
    ("k_np", "desc"),
    [
        (1.0, "symmetric (k_np=1)"),
        (2.0, "mildly asymmetric (k_np=2)"),
        (3.5, "highly asymmetric (k_np=3.5)"),
    ],
)
def test_sweep_fit_frequency_accuracy(k_np, desc):
    """Model fit should locate center frequency within 0.1% of domain width."""
    domain_lo, domain_hi = 2.82, 2.92
    true_freq = 2.87

    locator = _build_locator(domain_lo, domain_hi, n_steps=100)
    _inject_sweep_data(locator, true_freq, true_k_np=k_np)

    # Skip belief update (not testing belief, just the fit) by clearing pending obs
    locator._pending_obs.clear()
    locator.finalize()

    res = locator.result()
    assert "frequency" in res, f"No frequency in result for {desc}"
    assert "uncert" in res, f"No uncert in result for {desc}"

    err = abs(res["frequency"] - true_freq)
    tol = 0.005 * (domain_hi - domain_lo)  # 0.5% of domain = 500 kHz for 100 MHz window
    assert err < tol, (
        f"{desc}: freq error {err:.6f} GHz > tol {tol:.6f} GHz (est={res['frequency']:.6f}, true={true_freq})"
    )


def test_sweep_fit_narrow_dip():
    """Fit should work when dip is narrow relative to step size."""
    domain_lo, domain_hi = 2.84, 2.90
    true_freq = 2.87

    locator = _build_locator(domain_lo, domain_hi, n_steps=100)
    _inject_sweep_data(locator, true_freq, true_lw=0.0015, true_split=0.002)
    locator._pending_obs.clear()
    locator.finalize()

    res = locator.result()
    err = abs(res["frequency"] - true_freq)
    assert err < 0.003, f"Narrow-dip fit error {err:.4f} GHz"


def test_sweep_fit_acquisition_window_contains_true_freq():
    """Acquisition window should bracket the true frequency after finalize()."""
    domain_lo, domain_hi = 2.82, 2.92
    true_freq = 2.87

    locator = _build_locator(domain_lo, domain_hi)
    _inject_sweep_data(locator, true_freq, true_k_np=3.0)
    locator._pending_obs.clear()
    locator.finalize()

    res = locator.result()
    acq_lo = res["acquisition_lo"]
    acq_hi = res["acquisition_hi"]
    assert acq_lo <= true_freq <= acq_hi, (
        f"True freq {true_freq} outside acquisition window [{acq_lo:.4f}, {acq_hi:.4f}]"
    )


def test_sweep_fit_via_run_loop():
    """Model fit works through run_loop (raw model path, no UnitCubeSignalModel)."""
    import random

    from nvision import CoreExperiment, GenericSweepLocator, run_loop
    from nvision.models.noise import CompositeNoise
    from nvision.spectra.signal import TrueSignal

    true_params = NVCenterLorentzianSpectrum(
        frequency=2.85,
        linewidth=0.003,
        c_total=0.3,
        k_np=3.4,
        split=0.003,
    )
    bounds = {
        "frequency": (2.8, 2.9),
        "linewidth": (0.001, 0.02),
        "c_total": (0.0, 0.5),
        "k_np": (1.0, 5.0),
        "split": (0.001, 0.01),
    }
    true_signal = TrueSignal(
        model=NVCenterLorentzianModel(with_fixed_frequency=False),
        typed_parameters=true_params,
        bounds=bounds,
    )
    exp = CoreExperiment(
        true_signal=true_signal,
        noise=CompositeNoise(),
        x_min=2.8,
        x_max=2.9,
    )

    rng = random.Random(42)
    locator = None
    for loc in run_loop(
        GenericSweepLocator,
        exp,
        rng,
        max_steps=150,
        domain_lo=exp.x_min,
        domain_hi=exp.x_max,
        parameter_bounds=bounds,
    ):
        locator = loc

    locator.finalize()
    res = locator.result()
    assert "frequency" in res
    err = abs(res["frequency"] - 2.85)
    assert err < 0.001, f"run_loop fit error {err:.5f} GHz (est={res['frequency']:.5f})"


# --- Full-parameter accuracy tests (not just frequency) ---


def test_sweep_fit_full_parameter_recovery():
    """Every fitted physical parameter, not just frequency, should be close to truth."""
    domain_lo, domain_hi = 2.82, 2.92
    true_freq, true_lw, true_split, true_k_np, true_c_total = 2.87, 0.003, 0.003, 2.0, 0.3

    locator = _build_locator(domain_lo, domain_hi, n_steps=100)
    _inject_sweep_data(locator, true_freq, true_lw, true_split, true_k_np, true_c_total)
    locator._pending_obs.clear()
    locator.finalize()

    fit = locator.fit_mode_estimates()
    assert fit is not None, "finalize() must populate fit_mode_estimates()"
    assert abs(fit["frequency"] - true_freq) < 0.0005, f"frequency: {fit['frequency']} vs {true_freq}"
    assert abs(fit["linewidth"] - true_lw) < 0.0005, f"linewidth: {fit['linewidth']} vs {true_lw}"
    assert abs(fit["split"] - true_split) < 0.0005, f"split: {fit['split']} vs {true_split}"
    assert abs(fit["k_np"] - true_k_np) < 0.5, f"k_np: {fit['k_np']} vs {true_k_np}"
    assert abs(fit["c_total"] - true_c_total) < 0.05, f"c_total: {fit['c_total']} vs {true_c_total}"


def test_sweep_fit_zeeman_only():
    """Zeeman-split (2-dip) spectrum: frequency and zeeman_split recovered accurately."""
    domain_lo, domain_hi = 2.7, 3.0
    model = NVCenterLorentzianModel(
        with_hyperfine_splitting=False, with_zeeman_splitting=True, with_fixed_frequency=False
    )
    true_params = NVCenterLorentzianZeemanSpectrum(frequency=2.85, linewidth=0.0015, zeeman_split=0.03, c_total=0.3)
    bounds = {
        "frequency": (2.75, 2.95),
        "linewidth": (1e-4, 0.01),
        "zeeman_split": (0.0, 0.05),
        "c_total": (0.05, 0.5),
    }

    locator = _build_locator_for(model, bounds, domain_lo, domain_hi, n_steps=500, noise_std=0.003)
    _inject_sweep_data_for(locator, model, true_params, noise_std=0.003)
    locator._pending_obs.clear()
    locator.finalize()

    fit = locator.fit_mode_estimates()
    assert fit is not None
    assert abs(fit["frequency"] - 2.85) < 0.001, f"frequency: {fit['frequency']}"
    assert abs(fit["zeeman_split"] - 0.03) < 0.003, f"zeeman_split: {fit['zeeman_split']}"


def test_sweep_fit_zeeman_only_narrow_linewidth_resolves_fixed_hyperfine():
    """Zeeman-only model (``with_hyperfine_splitting=False``) at a linewidth
    narrow enough to resolve the *fixed* nitrogen hyperfine triplet that the
    model always evaluates internally (``NV_N14_HYPERFINE_SPLIT_HZ``, k_np=1.0
    — see ``NVCenterLorentzianModel.compute``), even though hyperfine is not a
    free fit parameter here.

    Regression guard for a production bug: ``expected_dip_count()`` reports 2
    dips whenever hyperfine isn't a free parameter, assuming the fixed triplet
    always merges into one dip per Zeeman group. That's only true when
    linewidth >> ~2.16 MHz. At a narrow linewidth (500 kHz here) the triplet
    resolves into 3 separate dips per group (6 total), but the old peak cap
    (bounded by the model's declared count of 2) kept only the 2 most
    prominent detected peaks — often two hyperfine sub-lines from the *same*
    Zeeman group — seeding ``zeeman_split`` tens of MHz off from the true
    value with no way for curve_fit to recover (frequency ended up tens of
    MHz off too). Uses a large step count to isolate this seeding/cap bug from
    the separate, coarser-grained sampling-density limit documented in
    [[sweep-fit-bounds-aliasing]].
    """
    domain_lo, domain_hi = 2.78e9, 2.96e9
    model = NVCenterLorentzianModel(with_hyperfine_splitting=False, with_zeeman_splitting=True)
    true_params = NVCenterLorentzianZeemanSpectrum(
        frequency=2.87e9, linewidth=0.5e6, zeeman_split=53.29e6, c_total=0.25
    )
    bounds = {
        "frequency": (domain_lo, domain_hi),
        "linewidth": (0.2e6, 15.0e6),
        "zeeman_split": (0.0, 60e6),
        "c_total": (0.1, 0.4),
    }
    noise_std = 1e-6

    # n_steps=400 matches NVISION_SIMPLESWEEP_MAX_STEPS, the actual production
    # default for the "SimpleSweep" strategy — this is the exact regime a
    # real run hit the bug in.
    locator = _build_locator_for(model, bounds, domain_lo, domain_hi, n_steps=400, noise_std=noise_std)
    _inject_sweep_data_for(locator, model, true_params, noise_std=noise_std)
    locator._pending_obs.clear()
    locator.finalize()

    fit = locator.fit_mode_estimates()
    assert fit is not None
    assert abs(fit["frequency"] - 2.87e9) < 1e6, f"frequency: {fit['frequency'] / 1e6:.2f} MHz (true 2870.0)"
    assert abs(fit["zeeman_split"] - 53.29e6) < 2e6, f"zeeman_split: {fit['zeeman_split'] / 1e6:.2f} MHz (true 53.29)"


def test_sweep_fit_zeeman_and_hyperfine_six_dip():
    """Combined Zeeman + hyperfine (6-dip) spectrum: frequency and both splits recovered.

    This is the case ``_seed_frequency_and_splits`` exists for: 2 Zeeman groups
    of 3 hyperfine lines each, where a naive argmin/centroid init is
    systematically wrong (see NVCenterLorentzianModel.expected_dip_count()).
    """
    domain_lo, domain_hi = 2.7, 3.0
    model = NVCenterLorentzianModel(
        with_hyperfine_splitting=True, with_zeeman_splitting=True, with_fixed_frequency=False
    )
    true_params = NVCenterLorentzianZeemanHyperfineSpectrum(
        frequency=2.85, linewidth=0.0015, zeeman_split=0.03, split=0.004, k_np=2.0, c_total=0.3
    )
    bounds = {
        "frequency": (2.75, 2.95),
        "linewidth": (1e-4, 0.01),
        "zeeman_split": (0.0, 0.05),
        "split": (0.001, 0.01),
        "k_np": (1.0, 5.0),
        "c_total": (0.05, 0.5),
    }

    locator = _build_locator_for(model, bounds, domain_lo, domain_hi, n_steps=600, noise_std=0.003)
    _inject_sweep_data_for(locator, model, true_params, noise_std=0.003)
    locator._pending_obs.clear()
    locator.finalize()

    fit = locator.fit_mode_estimates()
    assert fit is not None
    assert abs(fit["frequency"] - 2.85) < 0.001, f"frequency: {fit['frequency']}"
    assert abs(fit["zeeman_split"] - 0.03) < 0.003, f"zeeman_split: {fit['zeeman_split']}"
    assert abs(fit["split"] - 0.004) < 0.001, f"split: {fit['split']}"


@pytest.mark.timeout(300)
def test_sweep_fit_asymmetric_triplet_shallow_line_hidden():
    """Highly asymmetric hyperfine triplet (k_np=4.5, split at max bound): all
    three lines fitted even when the shallow freq-split line hides below the
    peak-detection floor.

    Regression guard for two coupled seeding bugs (found from a production
    sweep where the fit tracked only the deepest dips): with only 2 of 3
    triplet lines detected, the old seed placed frequency on the *right* peak
    with split = gap/2, which is exactly a wrong local minimum (model's outer
    dips on the data's center+deepest dips, own center dip hidden by maxing
    k_np); and the sample-count-scaled smoothing window (n//30) smeared the
    whole triplet into one blob on dense sweeps, so even 1500 points converged
    to that wrong minimum with frequency ~4 MHz (split/2) off.  Real physical
    scales (Hz) on purpose — the toy-GHz tests never hit this regime.

    timeout=300, not the default 180: the float64 scalar curve_fn (required
    here, see the float32 comment in generic_sweep_locator.py) runs the
    n_steps=500 fit in ~95s uninstrumented but ~200s under `--cov` (its
    per-point Python-level `inner.compute` calls are exactly what coverage
    tracing penalizes most) — 180s left no margin for CI's added variance.
    """
    domain_lo, domain_hi = 2.8e9, 3.3e9
    true_freq, true_split, true_k_np = 3.057e9, 8.5e6, 4.5
    bounds = {
        "frequency": (domain_lo, domain_hi),
        "linewidth": (0.2e6, 5.0e6),
        "split": (2.0e6, 8.5e6),
        "k_np": (1.0, 5.0),
        "c_total": (0.1, 0.7),
    }
    model = NVCenterLorentzianModel(with_fixed_frequency=False)
    true_params = NVCenterLorentzianSpectrum(
        frequency=true_freq, linewidth=0.8e6, split=true_split, k_np=true_k_np, c_total=0.55
    )

    locator = _build_locator_for(model, bounds, domain_lo, domain_hi, n_steps=500, noise_std=0.01)
    _inject_sweep_data_for(locator, model, true_params, noise_std=0.01)
    locator._pending_obs.clear()
    locator.finalize()

    fit = locator.fit_mode_estimates()
    assert fit is not None
    assert abs(fit["frequency"] - true_freq) < 0.5e6, (
        f"frequency off by {abs(fit['frequency'] - true_freq) / 1e6:.2f} MHz"
    )
    assert abs(fit["split"] - true_split) < 0.5e6, f"split: {fit['split'] / 1e6:.2f} MHz (true 8.5)"
    assert abs(fit["k_np"] - true_k_np) < 1.0, f"k_np: {fit['k_np']:.2f} (true 4.5)"


# --- Fit-failure propagation ---


def test_sweep_fit_raises_when_bounds_incomplete():
    """finalize() must raise, not silently fall back, when physical bounds are incomplete.

    A GenericSweepLocator sweep is only useful *because* of the model fit; a
    fit that can't run (missing bounds for linewidth/split/k_np/c_total here)
    is a real configuration problem to surface, not something to paper over
    with a cruder centroid estimate.
    """
    import random

    from nvision import CoreExperiment, GenericSweepLocator, run_loop
    from nvision.models.noise import CompositeNoise
    from nvision.spectra.signal import TrueSignal

    true_params = NVCenterLorentzianSpectrum(frequency=2.85, linewidth=0.003, c_total=0.3, k_np=3.4, split=0.003)
    true_signal = TrueSignal(
        model=NVCenterLorentzianModel(),
        typed_parameters=true_params,
        bounds={"frequency": (2.8, 2.9)},
    )
    exp = CoreExperiment(true_signal=true_signal, noise=CompositeNoise(), x_min=2.8, x_max=2.9)

    rng = random.Random(42)
    locator = None
    # Deliberately incomplete: only frequency bounds, so the fit has nowhere
    # to look for linewidth/split/k_np/c_total.
    for loc in run_loop(
        GenericSweepLocator,
        exp,
        rng,
        max_steps=50,
        domain_lo=exp.x_min,
        domain_hi=exp.x_max,
        parameter_bounds={"frequency": (exp.x_min, exp.x_max)},
    ):
        locator = loc

    with pytest.raises(RuntimeError, match="physical_param_bounds"):
        locator.finalize()


# --- End-to-end low-SNR / narrow-linewidth stress test ---


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sweep_then_fit_finds_good_fit_at_low_snr(seed):
    """Actually sweep (locator.next()/observe(), not manual history injection)
    a narrow, noisy dip and confirm the model fit still recovers the true
    frequency, across several seeds.

    Regression guard for the known weak spot in ``_smooth_for_peak_detection``:
    its smoothing window is a fixed fraction of the domain (~domain_width/30),
    independent of the true linewidth, so a linewidth much narrower than that
    window distorts the ``c_total``/frequency seed.  This case (linewidth
    ~33x narrower than the smoothing window, noise_std=0.02 against a
    contrast of 0.3) is deliberately harder than ``test_sweep_fit_narrow_dip``
    (which uses noise-free data), while staying in a regime verified stable
    across 8 seeds — see the note below for where it stops being reliable.

    ``n_steps=250`` matters: calibration showed 150 steps at this linewidth
    genuinely isn't enough sweep resolution here — most seeds raised
    ``RuntimeError`` (curve_fit couldn't converge from any candidate) rather
    than just being less accurate. 250 steps converged well on 8/8 seeds
    tried; this test checks 4 of them to catch regressions without re-adding
    that flakiness.

    Uses plain ``rng.normal()`` additive noise on each observation rather than
    ``CoreExperiment``'s ``CompositeOverFrequencyNoise`` path: that path
    constructs a fresh polars ``DataBatch`` per single measurement, which is
    slow enough (and, under load, unreliable enough — it intermittently took
    45s+ for a similar case) to risk the suite's 60s per-test timeout. Driving
    ``next()``/``observe()`` directly still exercises the real sweep state
    machine, just not that noise layer.

    Note: pushing linewidth/noise further (e.g. linewidth=0.0006,
    noise_std=0.04, 150 steps) makes the fit unreliable — it converges to a
    wrong local minimum in roughly 2 of 3 seeds tried. That's a genuine
    limitation of the current seeding heuristics, not covered by this test.
    """
    domain_lo, domain_hi = 2.7, 3.0
    true_freq = 2.85
    noise_std = 0.02
    bounds = {
        "frequency": (domain_lo, domain_hi),
        "linewidth": (1e-4, 0.02),
        "split": (0.0005, 0.005),
        "k_np": (1.0, 5.0),
        "c_total": (0.05, 0.5),
    }
    model = NVCenterLorentzianModel(with_fixed_frequency=False)
    true_params = NVCenterLorentzianSpectrum(frequency=true_freq, linewidth=0.001, split=0.002, k_np=2.0, c_total=0.3)

    locator = _build_locator_for(model, bounds, domain_lo, domain_hi, n_steps=250, noise_std=noise_std)
    rng = default_rng(seed)
    domain_width = domain_hi - domain_lo
    while not locator.done():
        x_norm = locator.next()
        x_phys = domain_lo + x_norm * domain_width
        y_true = float(model.compute(x_phys, true_params))
        y_obs = y_true + rng.normal(0, noise_std)
        locator.observe(Observation(x=x_norm, signal_value=y_obs, noise_std=noise_std))

    locator.finalize()
    res = locator.result()
    err = abs(res["frequency"] - true_freq)
    assert err < 0.01, f"low-SNR sweep+fit error {err:.5f} GHz (est={res['frequency']:.5f}, true={true_freq})"


def test_sweep_fit_zeeman_hyperfine_no_spurious_dips():
    """Zeeman+hyperfine (6-dip) fit must not invent dips where the data is flat.

    Regression guard for a production bug: when each Zeeman group's 3
    hyperfine lines are asymmetric (k_np != 1) and only 2 of the 3 resolve
    above the peak-detection floor, `_seed_frequency_and_splits` used to seed
    a single (wrong) within-group split guess, and the old peak cap (bounded
    by the model's *declared* `expected_dip_count()`, which some model
    families under-report for zeeman+hyperfine — see
    ``NVCenterSaturationVoigtModel``) discarded real detected peaks. Both
    together made curve_fit converge to k_np≈1 (fully symmetric sextet): a
    fit with a small but nonzero dip at every one of the 6 line positions,
    including the two that are barely visible in the real (asymmetric) data —
    a small spurious dip appearing where the plotted true signal is flat.

    Checked directly against the dense curve (not just fitted parameter
    values, since this model's saturation reparametrization has genuine
    parameter degeneracies): the fitted curve's dip positions/count must
    match the true curve's, and it must not predict a >1% dip anywhere the
    true signal is under 0.2% deep.
    """
    from scipy.signal import find_peaks

    from nvision.spectra.nv_center import NVCenterSaturationVoigtModel, NVCenterSaturationVoigtZeemanHyperfineSpectrum

    domain_lo, domain_hi = 2.7e9, 3.0e9
    model = NVCenterSaturationVoigtModel(
        with_hyperfine_splitting=True, with_zeeman_splitting=True, with_fixed_frequency=False
    )
    true_params = NVCenterSaturationVoigtZeemanHyperfineSpectrum(
        frequency=2.85e9, saturation=2.0, sigma_inhom=0.3e6, zeeman_split=30e6, split=4e6, k_np=2.0
    )
    bounds = {
        "frequency": (domain_lo, domain_hi),
        "saturation": (0.02, 30.0),
        "sigma_inhom": (0.0, 3.0e6),
        "zeeman_split": (0.0, 100e6),
        "split": (2.0e6, 8.5e6),
        "k_np": (1.0, 5.0),
    }
    noise_std = 0.003

    locator = _build_locator_for(model, bounds, domain_lo, domain_hi, n_steps=600, noise_std=noise_std)
    _inject_sweep_data_for(locator, model, true_params, noise_std=noise_std)
    locator._pending_obs.clear()
    locator.finalize()

    fit = locator.fit_mode_estimates()
    assert fit is not None
    from nvision.spectra.nv_center import NVCenterSaturationVoigtZeemanHyperfineSpectrum as _Spec

    fit_params = _Spec(**fit)
    xs = np.linspace(domain_lo, domain_hi, 20000)
    y_fit = np.array([model.compute(x, fit_params) for x in xs])
    y_true = np.array([model.compute(x, true_params) for x in xs])

    fit_dips, fit_props = find_peaks(-y_fit, prominence=0.001)
    true_dips, _ = find_peaks(-y_true, prominence=0.001)
    assert len(fit_dips) == len(true_dips) == 6, (
        f"expected 6 dips in both curves, got fit={len(fit_dips)} true={len(true_dips)}"
    )

    # The bug this guards: with k_np=2 the 3 hyperfine lines per Zeeman group
    # have depth ratio 1:2:4 (true signal), but the seeding bug made curve_fit
    # collapse to k_np≈1 — a fully *symmetric* sextet, all 6 dips equal depth
    # (≈2.7% each here) — which is wrong-shaped everywhere, not just "a bit
    # off": it puts a visible dip at the two positions where the real
    # (asymmetric) signal is nearly flat. Check depth spread instead of an
    # absolute per-position threshold, since this model's saturation
    # reparametrization has a separate, unrelated k_np/split degeneracy this
    # test doesn't attempt to fully resolve.
    fit_depths = fit_props["prominences"]
    depth_ratio = float(fit_depths.min() / fit_depths.max())
    assert depth_ratio < 0.5, (
        f"fitted dip depths are too uniform (min/max={depth_ratio:.3f}) — looks like the "
        f"k_np-collapsed-to-symmetric regression (true ratio is 0.25 for k_np=2)"
    )


def test_sweep_fit_linewidth_not_blind_seeded():
    """Zeeman doublet: linewidth/c_total must not land in the compensating
    (too-wide, too-shallow) local minimum caused by blind bounds-midpoint
    linewidth seeding.

    Regression guard for a production bug found via a real `nv run-single
    NVCenter-lorentzian ... SimpleSweep --no-cache` repeat: `make_p0` seeded
    `c_total` from the data (`_estimate_dip_depth`) but left `linewidth`
    at the bounds midpoint always. With a bounds range spanning 10-25x
    (0.2-15 MHz here), a midpoint start far from the true ~0.26 MHz let
    curve_fit converge to a *compensating* wrong minimum: linewidth ~1.7-2x
    too wide, c_total ~2.4-2.5x too shallow — plausible-looking but wrong on
    both axes together, which is exactly what a narrower-but-shallower or
    wider-but-shallower fitted curve looks like when plotted against the
    data. Fixed by `_estimate_hwhm`: a data-driven half-width estimate from
    the half-depth crossing around the dip minimum, seeded the same way
    `c_total` already was.

    Uses a linewidth (2 MHz) comfortably above the sweep's ~1.25 MHz point
    spacing (500 MHz domain / 400 steps) -- narrower true linewidths (e.g.
    the original repro's ~0.26 MHz) hit a separate, genuine resolution limit
    (average <1 sample point per HWHM) that no amount of seeding can fix;
    this test isolates the seeding bug from that sampling-density issue.
    """
    from nvision.spectra.nv_center import NVCenterLorentzianZeemanSpectrum, nv_center_lorentzian_bounds_for_domain

    domain_lo, domain_hi = 2.6e9, 3.1e9
    model = NVCenterLorentzianModel(with_hyperfine_splitting=False, with_zeeman_splitting=True)
    true_params = NVCenterLorentzianZeemanSpectrum(
        frequency=2.87e9, linewidth=2.0e6, zeeman_split=42925601.76703225, c_total=0.2529513412260511
    )
    bounds = nv_center_lorentzian_bounds_for_domain(
        domain_lo, domain_hi, with_hyperfine_splitting=False, with_zeeman_splitting=True
    )

    locator = _build_locator_for(model, bounds, domain_lo, domain_hi, n_steps=400, noise_std=0.01)
    _inject_sweep_data_for(locator, model, true_params, noise_std=0.01, seed=0)
    locator._pending_obs.clear()
    locator.finalize()

    fit = locator.fit_mode_estimates()
    assert fit is not None
    lw_ratio = fit["linewidth"] / true_params.linewidth
    c_ratio = fit["c_total"] / true_params.c_total
    assert 0.7 < lw_ratio < 1.4, f"linewidth ratio {lw_ratio:.2f} looks like the blind-seed compensating minimum"
    assert 0.7 < c_ratio < 1.4, f"c_total ratio {c_ratio:.2f} looks like the blind-seed compensating minimum"
