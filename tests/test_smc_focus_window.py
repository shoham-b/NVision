import numpy as np

from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.spectra.gaussian import GaussianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


def _make_smc(freq_lo: float = 2.7e9, freq_hi: float = 2.8e9) -> UnitCubeSMCMarginalDistribution:
    """Helper: return a MockSMC with no-op candidate generation."""

    class MockSMC(UnitCubeSMCMarginalDistribution):
        def _generate_epoch_candidates(self) -> None:
            self._current_candidates = np.linspace(0.0, 1.0, 10).astype(np.float32)

    bounds = {
        "frequency": (freq_lo, freq_hi),
        "dip_depth": (0.0, 1.0),
        "sigma": (1e6, 10e6),
        "background": (0.0, 1.0),
    }
    base_model = GaussianModel()
    base_model.signal_min_span = lambda w: 1e5
    model = UnitCubeSignalModel(base_model, param_bounds_phys=bounds, x_bounds_phys=(freq_lo, freq_hi))
    smc = MockSMC(model=model, num_particles=1000, physical_param_bounds=bounds, physical_x_bounds=(freq_lo, freq_hi))
    smc._param_names = ["frequency", "sigma", "dip_depth", "background"]
    smc._weights = np.ones(1000, dtype=np.float32) / 1000.0
    return smc


def test_focus_window_prevents_covariance_underflow():
    """
    Test that when particles are tightly focused, narrowing the physical bounds
    restores the internal float32 covariance to a stable, non-underflowing scale.
    """
    smc = _make_smc()

    # Simulate particles collapsing to a tiny 10 KHz window around 2.75 GHz.
    # 2.75 GHz is at 0.5 in unit space; 10 KHz in 100 MHz is 1e-4 in unit space.
    f_idx = 0
    smc._particles[:, f_idx] = np.random.normal(loc=0.5, scale=1e-4, size=1000)
    smc._particles[:, f_idx] = np.clip(smc._particles[:, f_idx], 0.0, 1.0)

    internal_var_before = np.var(smc._particles[:, f_idx])
    assert internal_var_before < 1e-7, f"Variance {internal_var_before} is not small enough"

    # Narrow the physical bounds to a 50 KHz window around 2.75 GHz.
    smc.narrow_scan_parameter_physical_bounds("frequency", 2.75e9 - 25e3, 2.75e9 + 25e3)

    # Particles were spanning ~10 KHz; new bounds are 50 KHz.
    # They should occupy ~20% of [0, 1], giving variance ~O(1e-2).
    internal_var_after = np.var(smc._particles[:, f_idx])
    assert internal_var_after > 1e-3, f"Variance {internal_var_after} did not recover!"

    print(f"Internal variance recovered from {internal_var_before:.2e} to {internal_var_after:.2e}")


def test_focus_window_automatic_narrowing_during_resampling():
    """
    Test that when particles are tightly focused, calling _resample()
    automatically triggers narrowing of the physical bounds and remapping of the particles.
    """
    smc = _make_smc()

    f_idx = 0
    smc._particles[:, f_idx] = np.random.normal(loc=0.5, scale=1e-5, size=1000)
    smc._particles[:, f_idx] = np.clip(smc._particles[:, f_idx], 0.0, 1.0)

    # Narrowing is delayed until NVISION_MIN_STEPS_BEFORE_NARROWING (default 8)
    # steps have elapsed, to resolve multi-modal hyperfine ambiguity first.
    smc._step_count = 8

    old_lo, old_hi = smc.physical_param_bounds["frequency"]
    smc._resample()

    new_lo, new_hi = smc.physical_param_bounds["frequency"]
    assert new_lo > old_lo, f"Expected lo bound to increase, but stayed {new_lo}"
    assert new_hi < old_hi, f"Expected hi bound to decrease, but stayed {new_hi}"
    assert (new_hi - new_lo) < (old_hi - old_lo), "Expected window to be narrowed"

    internal_var_after = np.var(smc._particles[:, f_idx])
    assert internal_var_after > 5e-4, f"Internal variance after resampling did not recover: {internal_var_after}"

    print(f"Natively narrowed bounds from {(old_lo, old_hi)} to {(new_lo, new_hi)}")


def test_shoulder_based_narrowing_single_dip():
    """Window should narrow to the region the posterior particles occupy around a single dip.

    Narrowing (the "Unified Active-Range Union Focusing" in
    UnitCubeSMCMarginalDistribution._resample) derives the new window from the
    particle distribution itself, not from raw observations — so here we seed
    particles as if the posterior had already converged on a dip near
    ``dip_center`` (e.g. after several real ``update()`` calls) rather than
    setting ``_observations`` directly.
    """
    freq_lo, freq_hi = 2.7e9, 2.8e9
    smc = _make_smc(freq_lo, freq_hi)

    dip_center = 2.75e9
    dip_sigma = 2e6

    dip_center_unit = (dip_center - freq_lo) / (freq_hi - freq_lo)
    dip_sigma_unit = dip_sigma / (freq_hi - freq_lo)
    smc._particles[:, 0] = np.clip(np.random.normal(loc=dip_center_unit, scale=dip_sigma_unit, size=1000), 0.0, 1.0)

    # Narrowing is delayed until NVISION_MIN_STEPS_BEFORE_NARROWING (default 8)
    # steps have elapsed, to resolve multi-modal hyperfine ambiguity first.
    smc._step_count = 8

    smc._resample()

    new_lo, new_hi = smc.physical_param_bounds["frequency"]
    new_width = new_hi - new_lo

    # Window must have narrowed from the full 100 MHz span. Particle rejuvenation
    # (a small uniformly-redrawn fraction, see _resample step 8.5) keeps some
    # width even for a tightly clustered posterior, so we only assert the
    # direction of narrowing rather than a tight percentage bound.
    assert new_width < (freq_hi - freq_lo), "Window should have narrowed"
    # Dip centre must still be inside the window.
    assert new_lo <= dip_center <= new_hi, "Dip centre must remain inside window"

    print(f"Single-dip narrowing: width={new_width / 1e6:.1f} MHz, window=[{new_lo / 1e9:.4f}, {new_hi / 1e9:.4f}] GHz")


def test_shoulder_based_narrowing_connected_dips():
    """Two adjacent dip clusters are unioned into a single narrowed window covering both.

    As in ``test_shoulder_based_narrowing_single_dip``, the posterior is seeded
    directly (bimodal particle cloud) to represent what the belief would look
    like after real observations near each dip, since narrowing reads the
    particle distribution rather than raw observations.
    """
    freq_lo, freq_hi = 2.7e9, 2.8e9
    smc = _make_smc(freq_lo, freq_hi)

    dip_a = 2.74e9
    dip_b = 2.76e9  # Only 20 MHz apart — no baseline recovery between them.
    dip_sigma = 3e6

    dip_a_unit = (dip_a - freq_lo) / (freq_hi - freq_lo)
    dip_b_unit = (dip_b - freq_lo) / (freq_hi - freq_lo)
    dip_sigma_unit = dip_sigma / (freq_hi - freq_lo)

    half = 500
    particles = np.concatenate(
        [
            np.random.normal(loc=dip_a_unit, scale=dip_sigma_unit, size=half),
            np.random.normal(loc=dip_b_unit, scale=dip_sigma_unit, size=1000 - half),
        ]
    )
    smc._particles[:, 0] = np.clip(particles, 0.0, 1.0)

    # Narrowing is delayed until NVISION_MIN_STEPS_BEFORE_NARROWING (default 8)
    # steps have elapsed, to resolve multi-modal hyperfine ambiguity first.
    smc._step_count = 8

    smc._resample()

    new_lo, new_hi = smc.physical_param_bounds["frequency"]

    # Both dip centres must be inside the window.
    assert new_lo <= dip_a <= new_hi, "Left dip centre must remain inside window"
    assert new_lo <= dip_b <= new_hi, "Right dip centre must remain inside window"
    # Window must have narrowed from the 100 MHz full span.
    assert (new_hi - new_lo) < (freq_hi - freq_lo), "Window should have narrowed"

    print(
        f"Connected-dip narrowing: width={(new_hi - new_lo) / 1e6:.1f} MHz, "
        f"window=[{new_lo / 1e9:.4f}, {new_hi / 1e9:.4f}] GHz"
    )
