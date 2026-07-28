"""Single-observation Fisher information, aligned with ``likelihood.py``."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvision.models.observation import Observation, gaussian_likelihood_std
from nvision.spectra.signal import SignalModel


def fisher_information_matrix(
    *,
    x: float,
    model: SignalModel,
    parameters: Any,
    last_obs: Observation | None,
) -> np.ndarray | None:
    """Single-observation Fisher information at ``x``.

    Uses additive Gaussian noise with ``sigma`` from
    :func:`~nvision.models.observation.gaussian_likelihood_std`.

    Returns ``None`` if :meth:`~nvision.spectra.signal.SignalModel.gradient` is unavailable.
    """
    if not hasattr(model, "gradient") or not callable(getattr(model, "gradient", None)):
        return None
    try:
        grads = model.gradient(x, parameters)
    except AttributeError:
        return None
    if grads is None:
        return None
    grad_vec = np.array([grads[name] for name in model.parameter_names()], dtype=np.float64)

    sigma = gaussian_likelihood_std(last_obs)
    return gaussian_fisher_matrix(grad_vec, sigma)


def gaussian_fisher_matrix(grad_vec: np.ndarray, sigma: float) -> np.ndarray:
    """Scalar Gaussian likelihood Fisher matrix: ``(1/sigma^2) g g^T``."""
    g = np.ascontiguousarray(grad_vec, dtype=np.float64)
    s = float(sigma)
    return np.outer(g, g) / (s * s)


def marginal_crlbs_at_budget(
    model: SignalModel,
    true_typed_params: Any,
    x_lo: float,
    x_hi: float,
    noise_std: float,
    n_steps: int,
    n_grid: int = 512,
) -> dict[str, float]:
    """Per-parameter marginal CRLB achievable with ``n_steps`` uniform measurements.

    Computes the expected Fisher information for a uniform grid of ``n_grid``
    probe positions over ``[x_lo, x_hi]``, averages across them, scales by
    ``n_steps``, and returns per-parameter marginal CRLBs as
    ``sqrt(diag(pinv(n_steps * mean_FIM)))``.

    Uses Gaussian noise with ``sigma = noise_std`` throughout (Gaussian branch only).
    Returns an empty dict if the model has no analytical ``gradient`` method.
    """
    if not hasattr(model, "gradient") or not callable(getattr(model, "gradient", None)):
        return {}

    names = list(model.parameter_names())
    n_params = len(names)
    xs = np.linspace(x_lo, x_hi, n_grid)
    cum_fim = np.zeros((n_params, n_params), dtype=np.float64)
    valid = 0
    for xi in xs:
        try:
            grads = model.gradient(float(xi), true_typed_params)
        except Exception:
            continue
        if grads is None:
            continue
        grad_vec = np.array([grads[name] for name in names], dtype=np.float64)
        cum_fim += gaussian_fisher_matrix(grad_vec, noise_std)
        valid += 1

    if valid == 0:
        return {}

    mean_fim = cum_fim / valid
    total_fim = mean_fim * n_steps
    stds = single_shot_marginal_stds_from_fim(total_fim, n_params)
    return {names[i]: float(stds[i]) for i in range(n_params)}


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
