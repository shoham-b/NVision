import random
import numpy as np
from nvision import CoreExperiment, NVCenterCoreGenerator, Observer, nv_center_smc_belief, run_loop
from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator

rng = random.Random(11)
gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant='lorentzian')
true_signal = gen.generate(rng)
x_min, x_max = true_signal.get_param_bounds('frequency')
exp = CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)
pb = {name: true_signal.get_param_bounds(name) for name in true_signal.parameter_names}
cfg = {
    'builder': nv_center_smc_belief,
    'max_steps': 20,
    'convergence_threshold': 0.15,
    'parameter_bounds': pb,
    'noise_std': 0.05,
}

locator = SequentialBayesianExperimentDesignLocator.create(**cfg)
candidates = locator._generate_candidates()
print('n_candidates:', len(candidates))
print('candidates from _generate_candidates()', candidates[:5])
