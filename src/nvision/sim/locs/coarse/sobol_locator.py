"""Sobol-based coarse search locator.

Uses a low-discrepancy Sobol sequence over [0, 1] for needle-in-a-haystack style
global search on mostly-zero signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_belief import AbstractBeliefDistribution
from nvision.belief.grid_belief import GridBeliefDistribution, GridParameter
from nvision.models.locator import Locator
from nvision.sim.locs.coarse.numba_kernels import gaussian_peak_posterior_update
from nvision.spectra.dtypes import FLOAT_DTYPE
from nvision.spectra.signal import ParamSpec, SignalModel


def sobol_1d_sequence(n: int) -> NDArray[np.float64]:
    """Minimal deterministic 1D low-discrepancy sequence over [0, 1]."""

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


@dataclass(frozen=True)
class _BlackBoxParams:
    peak_x: float


@dataclass(frozen=True)
class _BlackBoxSampleParams:
    peak_x: np.ndarray


@dataclass(frozen=True)
class _BlackBoxUncertaintyParams:
    peak_x: float


class _BlackBoxSpec(ParamSpec[_BlackBoxParams, _BlackBoxSampleParams, _BlackBoxUncertaintyParams]):
    @property
    def names(self) -> tuple[str, ...]:
        return ("peak_x",)

    @property
    def dim(self) -> int:
        return 1

    def unpack_params(self, values) -> _BlackBoxParams:
        (v,) = values
        return _BlackBoxParams(float(v))

    def pack_params(self, params: _BlackBoxParams) -> tuple[float, ...]:
        return (float(params.peak_x),)

    def unpack_uncertainty(self, values) -> _BlackBoxUncertaintyParams:
        (v,) = values
        return _BlackBoxUncertaintyParams(float(v))

    def pack_uncertainty(self, u: _BlackBoxUncertaintyParams) -> tuple[float, ...]:
        return (float(u.peak_x),)

    def unpack_samples(self, arrays_in_order) -> _BlackBoxSampleParams:
        (px,) = arrays_in_order
        return _BlackBoxSampleParams(peak_x=np.asarray(px, dtype=FLOAT_DTYPE))

    def pack_samples(self, samples: _BlackBoxSampleParams) -> tuple[np.ndarray, ...]:
        return (np.asarray(samples.peak_x, dtype=FLOAT_DTYPE),)


class BlackBoxSignalModel(SignalModel[_BlackBoxParams, _BlackBoxSampleParams, _BlackBoxUncertaintyParams]):
    """Signal model placeholder for Sobol search (we only measure)."""

    _SPEC = _BlackBoxSpec()

    @property
    def spec(self) -> _BlackBoxSpec:
        return self._SPEC

    def compute(self, x: float, params: _BlackBoxParams) -> float:
        return 0.0

    def compute_vectorized_samples(self, x: float, samples: _BlackBoxSampleParams) -> np.ndarray:
        # Not used by coarse locators' observe() path; exists for compatibility.
        return np.zeros_like(samples.peak_x, dtype=FLOAT_DTYPE)


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
        return sobol_1d_sequence(n)

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
        updated_posterior, total = gaussian_peak_posterior_update(
            peak_param.grid,
            peak_param.posterior,
            float(obs.x),
            float(obs.signal_value),
        )
        if total > 1e-10:
            peak_param.posterior = updated_posterior
            peak_param.value = peak_param.mean()
        self.belief.last_obs = obs
