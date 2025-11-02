from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PriorModel(Protocol):
    def get_probabilities(self, min_x: float, max_x: float, num_x_bins: int) -> list[float]: ...


@dataclass
class UniformPrior(PriorModel):
    def get_probabilities(self, min_x: float, max_x: float, num_x_bins: int) -> list[float]:
        return [1.0 / num_x_bins] * num_x_bins


@dataclass
class GaussianPrior(PriorModel):
    center: float
    sigma: float

    def get_probabilities(self, min_x: float, max_x: float, num_x_bins: int) -> list[float]:
        xs = np.linspace(min_x, max_x, num_x_bins)
        probabilities = np.exp(-0.5 * ((xs - self.center) / self.sigma) ** 2)
        probabilities /= np.sum(probabilities)  # Normalize
        return probabilities.tolist()
