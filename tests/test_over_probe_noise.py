from __future__ import annotations

import random

from nvision.sim.noises import OverProbeDriftNoise, OverProbeRandomWalkNoise


def test_over_probe_drift_noise_does_not_shift_baseline() -> None:
    rng = random.Random(0)
    noise = OverProbeDriftNoise(drift_per_unit=0.5, stateful=False)
    y = noise.apply(1.0, rng, locator=None)
    assert y == 1.0


def test_over_probe_random_walk_noise_does_not_shift_baseline() -> None:
    rng = random.Random(0)
    noise = OverProbeRandomWalkNoise(step_sigma=0.5, initial_offset=0.0, stateful=False)
    y = noise.apply(1.0, rng, locator=None)
    assert y == 1.0
