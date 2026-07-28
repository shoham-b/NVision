"""Single-observation Fisher information, aligned with ``likelihood.py``."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvision.models.observation import Observation, gaussian_likelihood_std
from nvision.spectra.signal import SignalModel


def numerical_gradient_vector(
    x: float,
    model: SignalModel,
    parameters: Any,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    rel_step: float = 1e-4,
) -> np.ndarray | None:
    """Central-difference d(signal)/d(parameter) at ``x``, in ``parameter_names()`` order.

    Fallback for models with no analytical ``gradient`` — which is *every*
    NV-center model (``NVCenterVoigtModel``, ``NVCenterLorentzianModel``,
    ``NVCenterSaturationVoigtModel``). Without this the cumulative FIM is never
    built, so ``crlb_per_param()`` returns ``{}`` and every consumer of the
    per-parameter CRLB silently degrades to "no information available".

    Step size is taken from the parameter's own range (``rel_step * (hi - lo)``)
    rather than from its value: these parameters differ by ~7 orders of magnitude
    (Hz-scale widths vs a dimensionless contrast ~0.25), and a value-relative
    step degenerates near zero. Steps are clamped into ``param_bounds`` and the
    realized (possibly one-sided) denominator is used, so a parameter sitting on
    a bound still yields a valid derivative.
    """
    names = list(model.parameter_names())
    try:
        base = [float(v) for v in model.spec.pack_params(parameters)]
    except Exception:
        return None
    if len(base) != len(names):
        return None

    grads = np.zeros(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        lo, hi = (param_bounds or {}).get(name, (-np.inf, np.inf))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            h = rel_step * (hi - lo)
        else:
            h = rel_step * max(abs(base[i]), 1.0)
        if h <= 0:
            continue

        up, dn = list(base), list(base)
        up[i] = min(base[i] + h, hi)
        dn[i] = max(base[i] - h, lo)
        denom = up[i] - dn[i]
        if denom <= 0:
            continue
        try:
            y_up = float(model.compute(x, model.spec.unpack_params(up)))
            y_dn = float(model.compute(x, model.spec.unpack_params(dn)))
        except Exception:
            return None
        grads[i] = (y_up - y_dn) / denom
    return grads


def fisher_information_matrix(
    *,
    x: float,
    model: SignalModel,
    parameters: Any,
    last_obs: Observation | None,
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray | None:
    """Single-observation Fisher information at ``x``.

    Uses additive Gaussian noise with ``sigma`` from
    :func:`~nvision.models.observation.gaussian_likelihood_std`.

    Prefers the model's analytical :meth:`~nvision.spectra.signal.SignalModel.gradient`
    and falls back to :func:`numerical_gradient_vector` when the model has none.
    Returns ``None`` only if the gradient cannot be obtained either way.
    """
    grads = None
    if hasattr(model, "gradient") and callable(getattr(model, "gradient", None)):
        try:
            grads = model.gradient(x, parameters)
        except AttributeError:
            grads = None

    if grads is None:
        grad_vec = numerical_gradient_vector(x, model, parameters, param_bounds)
        if grad_vec is None:
            return None
    else:
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
    param_bounds: dict[str, tuple[float, float]] | None = None,
    n_grid: int = 512,
) -> dict[str, float]:
    """Per-parameter marginal CRLB achievable with ``n_steps`` uniform measurements.

    Computes the expected Fisher information for a uniform grid of ``n_grid``
    probe positions over ``[x_lo, x_hi]``, averages across them, scales by
    ``n_steps``, and returns per-parameter marginal CRLBs (in each parameter's
    own physical units) as ``sqrt(diag(pinv(n_steps * mean_FIM)))``.

    Uses Gaussian noise with ``sigma = noise_std`` throughout (Gaussian branch only).
    Returns an empty dict if the model has no analytical ``gradient`` method
    (e.g. Voigt/Saturation-Voigt NV-center models, which don't have one yet;
    NVCenterLorentzianModel and NVCenterOnePeakLorentzianModel do).

    ``param_bounds`` normalizes each parameter's gradient by its own range
    before building the FIM (and un-normalizes the resulting stds back to
    physical units afterward) -- **pass this whenever available.**
    :func:`single_shot_marginal_stds_from_fim`'s ridge is absolute and only
    meaningful when every parameter has comparable scale; built directly from
    physical gradients, Hz-scale widths (~1e6) and a dimensionless contrast
    (~0.1) differ by ~7 orders of magnitude, so the ridge dominates every
    Hz-scale direction and its CRLB silently saturates at ``sqrt(1/ridge) ==
    1000`` regardless of the data -- the caller-visible symptom this function
    almost shipped with in that state before the ``param_bounds`` normalization
    fed by ``runner/executor.py``'s CRLB feasibility gate was added. Omitting
    ``param_bounds`` reproduces that unnormalized (unsafe) behavior; only skip
    it for callers that have already normalized the gradient themselves.
    """
    if not hasattr(model, "gradient") or not callable(getattr(model, "gradient", None)):
        return {}

    names = list(model.parameter_names())
    n_params = len(names)
    ranges = np.ones(n_params, dtype=np.float64)
    if param_bounds is not None:
        for i, name in enumerate(names):
            lo, hi = param_bounds.get(name, (0.0, 0.0))
            if hi > lo:
                ranges[i] = hi - lo

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
        grad_vec = np.array([grads[name] for name in names], dtype=np.float64) * ranges
        cum_fim += gaussian_fisher_matrix(grad_vec, noise_std)
        valid += 1

    if valid == 0:
        return {}

    mean_fim = cum_fim / valid
    total_fim = mean_fim * n_steps
    stds_normalized = single_shot_marginal_stds_from_fim(total_fim, n_params)
    return {names[i]: float(stds_normalized[i] * ranges[i]) for i in range(n_params)}


def single_shot_marginal_stds_from_fim(
    fim: np.ndarray | None,
    n_params: int,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    """``sqrt(diag(pinv(FIM + ridge*I)))`` as a length-``n_params`` vector; NaNs if invalid.

    **The ridge is absolute, so this is only valid for a FIM in unit-normalized
    coordinates.** It caps the CRLB of a fully degenerate direction at
    ``sqrt(1/ridge)`` *in whatever units the FIM is in*. With unit-cube parameters
    the diagonal runs ~1e4-1e7, the ridge is negligible, and a degenerate direction
    correctly reports a huge CRLB -- which is how :meth:`crlb_per_param` uses it.
    Hand it a FIM built from *physical* parameters instead (Hz-scale widths give
    entries ~1e-10) and the ridge dominates completely: every parameter comes back
    at exactly ``sqrt(1/1e-6)`` = 1000, independent of the data. Normalize the
    gradients by each parameter's range before calling this.
    """
    out = np.full(n_params, np.nan, dtype=np.float64)
    if fim is None or fim.size == 0 or fim.shape != (n_params, n_params):
        return out
    cov = np.linalg.pinv(fim + np.eye(n_params, dtype=np.float64) * ridge)
    for i in range(n_params):
        out[i] = float(np.sqrt(max(0.0, cov[i, i])))
    return out
