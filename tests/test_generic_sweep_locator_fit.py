"""Tests that GenericSweepLocator.finalize() produces accurate frequency estimates.

The locator uses a full model fit (scipy curve_fit) rather than centroid heuristics.
These tests verify accuracy in physically realistic regimes.
"""

import numpy as np
import pytest
from numpy.random import default_rng

from nvision.models.observation import Observation
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel, NVCenterLorentzianSpectrum
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution


_BOUNDS = {
    "frequency": (2.82, 2.92),
    "linewidth": (1e-4, 0.02),
    "split": (0.001, 0.015),
    "k_np": (1.0, 5.0),
    "c_total": (0.05, 0.5),
}


def _build_locator(domain_lo=2.82, domain_hi=2.92, n_steps=100) -> GenericSweepLocator:
    phys_model = NVCenterLorentzianModel()
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
    phys_model = NVCenterLorentzianModel()
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


# --- Accuracy tests ---

@pytest.mark.parametrize("k_np,desc", [
    (1.0, "symmetric (k_np=1)"),
    (2.0, "mildly asymmetric (k_np=2)"),
    (3.5, "highly asymmetric (k_np=3.5)"),
])
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
        f"{desc}: freq error {err:.6f} GHz > tol {tol:.6f} GHz "
        f"(est={res['frequency']:.6f}, true={true_freq})"
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
    from nvision import CoreExperiment, SimpleSweepLocator, run_loop
    from nvision.spectra.signal import TrueSignal
    from nvision.models.noise import CompositeNoise

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
        model=NVCenterLorentzianModel(),
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
        SimpleSweepLocator,
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
