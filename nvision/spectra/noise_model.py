"""Abstractions for noise as a signal model.

NoiseSignalModel mirrors SignalModel but operates on residuals (y_obs - μ)
rather than probe positions x.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence

import numpy as np
from numba import njit

from nvision.spectra.spec import BasicParamSpec, NoiseSignalModel, ParamSpec


class NoiseSignalModel(NoiseSignalModel, abc.ABC):
    """Abstract base for noise models that can be jointly inferred.

    A NoiseSignalModel provides a ParamSpec for its latent parameters
    and a composite log-likelihood kernel that combines aleatoric
    (physical noise) and epistemic (parameter uncertainty) spread.
    """

    @property
    @abc.abstractmethod
    def spec(self) -> ParamSpec:
        """The parameter specification for this noise model."""

    @abc.abstractmethod
    def composite_log_likelihood(
        self,
        predicted: np.ndarray,
        residuals: np.ndarray,
        noise_param_arrays: Sequence[np.ndarray],
        sigma_epistemic: float,
    ) -> np.ndarray:
        """Compute per-particle log-likelihoods.

        Parameters
        ----------
        predicted : np.ndarray
            μ_i for each particle i. Shape (N,).
        residuals : np.ndarray
            y_obs - μ_i for each particle i. Shape (N,).
        noise_param_arrays : Sequence[np.ndarray]
            Arrays for each latent noise parameter. Each shape (N,).
        sigma_epistemic : float
            Standard deviation of signal predictions across all particles at x.

        Returns
        -------
        np.ndarray
            Log-likelihoods for each particle. Shape (N,).
        """


@njit(cache=True)
def _gaussian_composite_ll_jit(
    residuals: np.ndarray,
    sigma_aleatoric_arr: np.ndarray,
    sigma_epistemic: float,
) -> np.ndarray:
    n = residuals.shape[0]
    out = np.empty(n, dtype=np.float64)
    eps_sq = sigma_epistemic * sigma_epistemic
    for i in range(n):
        s_a = max(sigma_aleatoric_arr[i], 1e-9)
        # Combine aleatoric and epistemic variances
        s_eff = math.sqrt(s_a * s_a + eps_sq)
        out[i] = -0.5 * (residuals[i] / s_eff) ** 2 - math.log(s_eff)
    return out


class GaussianNoiseSignalModel(NoiseSignalModel):
    """Gaussian noise with an uncertain standard deviation."""

    def __init__(self, prior_bounds: dict[str, tuple[float, float]]):
        self._spec = BasicParamSpec(["noise_sigma"], prior_bounds)

    @property
    def spec(self) -> ParamSpec:
        return self._spec

    def composite_log_likelihood(
        self,
        predicted: np.ndarray,
        residuals: np.ndarray,
        noise_param_arrays: Sequence[np.ndarray],
        sigma_epistemic: float,
    ) -> np.ndarray:
        return _gaussian_composite_ll_jit(
            residuals,
            noise_param_arrays[0],
            sigma_epistemic,
        )


@njit(cache=True)
def _poisson_composite_ll_jit(
    obs_y: float,
    predicted: np.ndarray,
    scale_arr: np.ndarray,
    sigma_epistemic: float,
) -> np.ndarray:
    n = predicted.shape[0]
    out = np.empty(n, dtype=np.float64)
    eps_sq = sigma_epistemic * sigma_epistemic
    for i in range(n):
        scale = max(scale_arr[i], 1e-12)
        lam = max(predicted[i] * scale, 1e-12)
        k = max(round(obs_y * scale), 0)

        # Poisson log-likelihood: k*log(lam) - lam - log(k!)
        log_p = k * math.log(lam) - lam - math.lgamma(k + 1.0)

        if eps_sq > 1e-12:
            # Broaden Poisson by adding epistemic uncertainty in quadrature
            # sigma_total^2 = lam + (sigma_epistemic * scale)^2
            # We approximate the broadened Poisson as Gaussian-like tempering
            sigma_p_sq = lam
            sigma_eff_sq = sigma_p_sq + (sigma_epistemic * scale) ** 2
            tempering = sigma_p_sq / sigma_eff_sq
            out[i] = log_p * tempering
        else:
            out[i] = log_p
    return out


class PoissonNoiseSignalModel(NoiseSignalModel):
    """Poisson noise with an uncertain scale (counts per unit signal)."""

    def __init__(self, prior_bounds: dict[str, tuple[float, float]]):
        self._spec = BasicParamSpec(["poisson_scale"], prior_bounds)

    @property
    def spec(self) -> ParamSpec:
        return self._spec

    def composite_log_likelihood(
        self,
        predicted: np.ndarray,
        residuals: np.ndarray,
        noise_param_arrays: Sequence[np.ndarray],
        sigma_epistemic: float,
    ) -> np.ndarray:
        # obs_y = predicted + residuals
        # We only need the first element of obs_y if it's a scalar measurement
        obs_y = float(predicted[0] + residuals[0])
        return _poisson_composite_ll_jit(
            obs_y,
            predicted,
            noise_param_arrays[0],
            sigma_epistemic,
        )


class DriftNoiseSignalModel(NoiseSignalModel):
    """Adds a slow linear drift across sequential probes."""

    def __init__(self, prior_bounds: dict[str, tuple[float, float]]):
        self._spec = BasicParamSpec(["drift_rate"], prior_bounds)

    @property
    def spec(self) -> ParamSpec:
        return self._spec

    def composite_log_likelihood(
        self,
        predicted: np.ndarray,
        residuals: np.ndarray,
        noise_param_arrays: Sequence[np.ndarray],
        sigma_epistemic: float,
    ) -> np.ndarray:
        # Drift rate acts as a scale on the signal deviation from baseline (1.0).
        # We approximate it as an additive Gaussian noise with sigma proportional
        # to the drift rate and the current signal depth.
        drift_rate = np.maximum(noise_param_arrays[0], 1e-12)
        # depth = |signal - baseline|
        depth = np.abs(predicted - 1.0)
        # Effective sigma from drift
        sigma_aleatoric = drift_rate * depth / np.sqrt(12.0)  # Uniform variance
        return _gaussian_composite_ll_jit(residuals, sigma_aleatoric, sigma_epistemic)


class CompositeNoiseSignalModel(NoiseSignalModel):
    """Combines multiple noise models by summing their log-likelihoods."""

    def __init__(self, models: list[NoiseSignalModel]):
        self.models = models
        # Concatenate specs
        all_names = []
        all_bounds = {}
        for m in models:
            all_names.extend(m.spec.names)
            all_bounds.update(m.spec.bounds)
        self._spec = BasicParamSpec(all_names, all_bounds)

    @property
    def spec(self) -> ParamSpec:
        return self._spec

    def composite_log_likelihood(
        self,
        predicted: np.ndarray,
        residuals: np.ndarray,
        noise_param_arrays: Sequence[np.ndarray],
        sigma_epistemic: float,
    ) -> np.ndarray:
        log_lik = np.zeros(len(residuals))
        offset = 0
        for model in self.models:
            n_params = len(model.spec.names)
            subset = noise_param_arrays[offset : offset + n_params]
            log_lik += model.composite_log_likelihood(predicted, residuals, subset, sigma_epistemic)
            offset += n_params
        return log_lik
