"""The belief draws only from its own seeded stream, so equal seeds reproduce a run exactly."""

import numpy as np

from nvision.models.observation import Observation
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from tests.noise import gaussian_noise


def _run(seed: int | None, *, global_seed: int) -> np.ndarray:
    np.random.seed(global_seed)  # must not matter
    belief = nv_center_smc_belief(num_particles=300, noise_model=gaussian_noise(), seed=seed)
    for i in range(12):
        x = float(belief.select_max_information_gain(belief.get_candidate_drive_freq_phys(), 1)[0]) if i else 0.3
        unit_x = (x - belief.drive_freq_bounds_phys[0]) / (
            belief.drive_freq_bounds_phys[1] - belief.drive_freq_bounds_phys[0]
        )
        belief.update(Observation(drive_freq_unit=unit_x, signal_value=0.9 + 0.01 * (i % 3), noise_std=0.02))
    belief._resample()
    return belief._particles.copy()


def test_same_seed_reproduces_particles_regardless_of_global_state():
    assert np.array_equal(_run(5, global_seed=1), _run(5, global_seed=2))


def test_different_seeds_differ():
    assert not np.array_equal(_run(5, global_seed=1), _run(6, global_seed=1))
