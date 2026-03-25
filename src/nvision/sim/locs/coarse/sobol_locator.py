"""Sobol-based coarse search locator.

Uses a low-discrepancy Sobol sequence over [0, 1] for needle-in-a-haystack style
global search on mostly-zero signals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from nvision.models.locator import Locator
from nvision.signal.abstract_belief import AbstractBeliefDistribution
from nvision.signal.grid_belief import GridBeliefDistribution, GridParameter
from nvision.signal.signal import SignalModel


class BlackBoxSignalModel(SignalModel):
    """Signal model placeholder for Sobol search (we only measure)."""

    @staticmethod
    def eval_black_box_signal_model(x: float, peak_x: float) -> float:
        return 0.0

    def compute(self, x: float, params: list) -> float:  # pragma: no cover - not used
        v = self._param_floats_canonical(params)
        return self.eval_black_box_signal_model(x, v[0])

    def parameter_names(self) -> list[str]:
        return ["peak_x"]


class SobolLocator(Locator):
    """Coarse locator using a 1D Sobol sequence over [0, 1]."""

    def __init__(self, belief: AbstractBeliefDistribution, max_steps: int = 64):
        super().__init__(belief)
        self.max_steps = max_steps
        self.step_count = 0
        self._points = self._sobol_1d(max_steps)
        self.best_signal = -np.inf
        self.best_x = 0.5

    @classmethod
    def create(cls, max_steps: int = 64, n_grid: int = 128, **kwargs) -> SobolLocator:
        model = BlackBoxSignalModel()
        belief = GridBeliefDistribution(
            model=model,
            parameters=[
                GridParameter(
                    name="peak_x",
                    bounds=(0.0, 1.0),
                    grid=np.linspace(0.0, 1.0, n_grid),
                    posterior=np.ones(n_grid) / n_grid,
                ),
            ],
        )
        return cls(belief, max_steps)

    def _sobol_1d(self, n: int) -> NDArray[np.float64]:
        """Minimal 1D Sobol sequence generator over [0, 1]."""

        # Use a simple van der Corput base-2 sequence as a stand-in.
        def vdc(k: int, base: int = 2) -> float:
            v = 0.0
            denom = 1.0
            while k:
                k, remainder = divmod(k, base)
                denom *= base
                v += remainder / denom
            return v

        return np.array([vdc(i + 1) for i in range(n)], dtype=float)

    def next(self) -> float:
        x = self._points[self.step_count]
        self.step_count += 1
        return float(x)

    def done(self) -> bool:
        return self.step_count >= self.max_steps

    def result(self) -> dict[str, float]:
        return {
            "peak_x": self.best_x,
            "peak_signal": self.best_signal,
        }

    def observe(self, obs) -> None:
        if obs.signal_value > self.best_signal:
            self.best_signal = obs.signal_value
            self.best_x = obs.x
        peak_param = self.belief.get_param("peak_x")
        likelihoods = np.exp(-0.5 * ((peak_param.grid - obs.x) / 0.1) ** 2) * (obs.signal_value + 1.0)
        unnormalized = peak_param.posterior * (likelihoods + 1e-10)
        total = unnormalized.sum()
        if total > 1e-10:
            peak_param.posterior = unnormalized / total
            peak_param.value = peak_param.mean()
        self.belief.last_obs = obs
