from __future__ import annotations

import numpy as np

from nvision.spectra.likelihood import likelihood_from_observation_model


def test_poisson_likelihood_prefers_matching_rate() -> None:
    scale = 100.0
    obs_y = 0.5  # corresponds to k=50 counts
    predicted = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    likelihood = likelihood_from_observation_model(
        obs_y=obs_y,
        predicted=predicted,
        noise_std=0.05,
        frequency_noise_model=({"type": "poisson", "scale": scale},),
    )
    assert int(np.argmax(likelihood)) == 1


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
