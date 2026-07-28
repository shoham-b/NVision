from __future__ import annotations

import numpy as np

from nvision import likelihood_from_observation_model


def test_unknown_model_falls_back_to_gaussian() -> None:
    obs_y = 0.2
    predicted = np.array([0.0, 0.2, 0.4], dtype=np.float64)
    likelihood = likelihood_from_observation_model(
        obs_y=obs_y,
        predicted=predicted,
        noise_std=0.1,
        frequency_noise_model=({"type": "unknown", "name": "CustomNoise"},),
    )
    assert int(np.argmax(likelihood)) == 1
