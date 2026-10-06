"""The belief draws only from its own seeded stream, so equal seeds reproduce a run exactly."""

import numpy as np

from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from tests.noise import gaussian_noise


def _run(seed: int | None, *, global_seed: int) -> np.ndarray:
    np.random.seed(global_seed)  # must not matter
    belief = nv_center_smc_belief(num_particles=300, noise_model=gaussian_noise(), seed=seed)
    for i in range(12):
        lo_phys, hi_phys = belief.drive_freq_bounds_phys
        if i:
            drive_freq_phys = float(belief.select_max_information_gain(belief.get_candidate_drive_freq_phys(), 1)[0])
            drive_freq_unit = (drive_freq_phys - lo_phys) / (hi_phys - lo_phys)
        else:
            drive_freq_unit = 0.3
        belief.update(Observation(drive_freq_unit=drive_freq_unit, signal_value=0.9 + 0.01 * (i % 3), noise_std=0.02))
    belief._resample()
    return belief._particles.copy()


def test_same_seed_reproduces_particles_regardless_of_global_state():
    assert np.array_equal(_run(5, global_seed=1), _run(5, global_seed=2))


def test_different_seeds_differ():
    assert not np.array_equal(_run(5, global_seed=1), _run(6, global_seed=1))
