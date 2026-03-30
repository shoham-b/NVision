"""Single-observation Fisher information (Gaussian or Poisson), aligned with ``likelihood.py``."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvision.models.observation import Observation, gaussian_likelihood_std
from nvision.parameter import Parameter
from nvision.spectra.signal import SignalModel


def _is_poisson_frequency_model(frequency_noise_model: tuple[dict[str, Any], ...] | None) -> bool:
    if not frequency_noise_model or len(frequency_noise_model) != 1:
        return False
    spec = frequency_noise_model[0]
    return spec.get("type") == "poisson" and float(spec.get("scale", 0.0) or 0.0) > 0.0


def poisson_expected_fisher_matrix(grad_vec: np.ndarray, f_pred: float, scale: float) -> np.ndarray:
    """Expected Fisher for Poisson counts with mean ``lambda = max(f_pred * scale, eps)``.

    Matches the single-component Poisson branch in
    :func:`nvision.spectra.likelihood.likelihood_from_observation_model`: rate
    ``lambda = f_pred * scale`` and ``grad(lambda) = scale * grad(f)``.

    Uses ``I = (1/lambda) grad(lambda) grad(lambda)^T`` (standard Poisson rate Fisher).
    """
    g = np.ascontiguousarray(grad_vec, dtype=np.float64)
    scale_f = float(scale)
    min_lam = 1e-12
    lam = max(float(f_pred) * scale_f, min_lam)
    v = scale_f * g
    return np.outer(v, v) / lam


def fisher_information_matrix(
    *,
    x: float,
    model: SignalModel,
    parameters: list[Parameter],
    last_obs: Observation | None,
) -> np.ndarray | None:
    """Single-observation Fisher information at ``x`` (Gaussian or Poisson).

    Uses :attr:`Observation.frequency_noise_model` the same way as
    :func:`nvision.spectra.likelihood.likelihood_from_observation_model`:

    - **Poisson** (one component, ``type=="poisson"``, ``scale > 0``): expected Fisher
      for Poisson rate with ``lambda = f(theta|x) * scale``.
    - **Otherwise**: additive Gaussian noise with ``sigma`` from
      :func:`~nvision.models.observation.gaussian_likelihood_std`.

    Returns ``None`` if :meth:`~nvision.spectra.signal.SignalModel.gradient` is unavailable.
    """
    grads = model.gradient(x, parameters)
    if grads is None:
        return None
    grad_vec = np.array([grads[p.name] for p in parameters], dtype=np.float64)

    freq = last_obs.frequency_noise_model if last_obs is not None else None
    if _is_poisson_frequency_model(freq):
        scale = float(freq[0].get("scale", 0.0))
        f_pred = float(model.compute_from_params(x, parameters))
        return poisson_expected_fisher_matrix(grad_vec, f_pred, scale)

    sigma = gaussian_likelihood_std(last_obs)
    return gaussian_fisher_matrix(grad_vec, sigma)


def gaussian_fisher_information_matrix(
    *,
    x: float,
    model: SignalModel,
    parameters: list[Parameter],
    last_obs: Observation | None,
) -> np.ndarray | None:
    """Gaussian FIM only (ignores Poisson metadata). Prefer :func:`fisher_information_matrix`."""
    grads = model.gradient(x, parameters)
    if grads is None:
        return None
    grad_vec = np.array([grads[p.name] for p in parameters], dtype=np.float64)
    sigma = gaussian_likelihood_std(last_obs)
    return gaussian_fisher_matrix(grad_vec, sigma)


def gaussian_fisher_matrix(grad_vec: np.ndarray, sigma: float) -> np.ndarray:
    """Scalar Gaussian likelihood Fisher matrix: ``(1/sigma^2) g g^T``."""
    g = np.ascontiguousarray(grad_vec, dtype=np.float64)
    s = float(sigma)
    return np.outer(g, g) / (s * s)


def single_shot_marginal_stds_from_fim(
    fim: np.ndarray | None,
    n_params: int,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    """``sqrt(diag(pinv(FIM + ridge*I)))`` as a length-``n_params`` vector; NaNs if invalid."""
    out = np.full(n_params, np.nan, dtype=np.float64)
    if fim is None or fim.size == 0 or fim.shape != (n_params, n_params):
        return out
    cov = np.linalg.pinv(fim + np.eye(n_params, dtype=np.float64) * ridge)
    for i in range(n_params):
        out[i] = float(np.sqrt(max(0.0, cov[i, i])))
    return out
