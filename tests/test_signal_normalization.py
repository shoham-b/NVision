from __future__ import annotations

import random

import numpy as np

from nvision import LorentzianModel
from nvision import VoigtZeemanModel


def test_signal_normalization() -> None:
    for model in [LorentzianModel(), VoigtZeemanModel()]:
        rng = random.Random(42)
        for _ in range(100):
            params = model.sample_params(rng)
            xs = np.linspace(0.0, 1.0, 1000)
            ys = [model.compute_from_params(float(x), params) for x in xs]
            assert min(ys) >= -0.05, "signal below 0 by more than tolerance"
            assert max(ys) <= 1.05, "signal above 1 by more than tolerance"
