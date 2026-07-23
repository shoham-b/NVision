import numpy as np
import pytest
from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.models.observation import Observation

def test_generic_sweep_classic_fit():
    np.random.seed(42)

    # Physical k_np must be >= 1 (minimum physical value for NV center).
    # Using k_np=1.5: freq+split dip is 1.5x deeper than the other two,
    # which matches the actual physical constraint (k_np in [1, 5]).
    true_freq = 2871.23
    phys_model = NVCenterLorentzianModel(with_fixed_frequency=False)

    true_params = {
        "frequency": true_freq,
        "linewidth": 2.0,
        "split": 1.0,
        "k_np": 1.5,
        "c_total": 0.05,
    }

    unit_model = UnitCubeSignalModel(
        phys_model,
        param_bounds_phys={
            "frequency": (2860.0, 2880.0),
            "linewidth": (1.0, 5.0),
            "split": (0.0, 5.0),
            "k_np": (1.0, 5.0),
            "c_total": (0.01, 0.1),
        },
        x_bounds_phys=(2860.0, 2880.0),
    )

    belief = UnitCubeSMCMarginalDistribution(
        unit_model,
        num_particles=100,
        physical_param_bounds=unit_model.param_bounds_phys,
        physical_x_bounds=unit_model.x_bounds_phys,
    )

    locator = GenericSweepLocator(
        belief=belief,
        signal_model=unit_model,
        max_steps=20,
        noise_std=0.005,
        domain_lo=2860.0,
        domain_hi=2880.0,
    )

    # Simulate a sweep
    while not locator.done():
        x = locator.next()

        # Evaluate model to get y
        u_arrs = []
        for name in unit_model.parameter_names():
            lo, hi = unit_model.param_bounds_phys[name]
            u = (true_params[name] - lo) / (hi - lo)
            u_arrs.append(np.array([u]))

        y_true = unit_model.compute_vectorized_many(np.array([x]), u_arrs)[0, 0]
        y_obs = y_true + np.random.normal(0, 0.005)
        locator.observe(Observation(x, y_obs))

    locator.finalize()
    res = locator.result()

    est_freq = res["frequency"]
    err = abs(est_freq - true_freq)
    print(f"True freq: {true_freq}, Est: {est_freq}, Err: {err}")
    # Tolerance: 1 MHz (= one sweep step for a 20-step / 20 MHz grid).
    assert err < 1.0, f"Error is {err} MHz, which is too large for a 20 MHz grid!"
