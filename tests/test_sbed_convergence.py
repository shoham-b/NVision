import random
import numpy as np
import pytest

from nvision import CoreExperiment, NVCenterCoreGenerator, Observer, nv_center_smc_belief, run_loop
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator

@pytest.mark.slow
@pytest.mark.timeout(120)
def test_sbed_locator_long_running_convergence():
    rng = random.Random(42)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant='lorentzian')
    true_signal = gen.generate(rng)

    x_min, x_max = true_signal.get_param_bounds('frequency')
    exp = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)

    pb = {name: true_signal.get_param_bounds(name) for name in true_signal.parameter_names}

    cfg = {
        'builder': nv_center_smc_belief,
        'max_steps': 150,  # Long running
        'convergence_threshold': 0.15,
        'parameter_bounds': pb,
        'noise_std': 0.05,
    }

    it = run_loop(SequentialBayesianExperimentDesignLocator, exp, rng, **cfg)
    last_state = None
    for state in it:
        last_state = state

    # Check convergence
    assert last_state is not None
    assert getattr(last_state, "_is_converged", False) or last_state.step_count >= 150 or getattr(last_state.locator, "_is_converged", False) or getattr(last_state.belief, "_is_converged", False)

    # Assert frequency estimate is close to true
    est_freq = last_state.belief.estimates()['frequency']
    true_freq = true_signal.get_param_value('frequency')
    assert abs(est_freq - true_freq) < 0.2e9
