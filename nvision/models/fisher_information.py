"""Fisher information and Cramer-Rao bounds: the single home for everything CRLB-related.

Single-observation Fisher information (aligned with ``likelihood.py``), cumulative FIMs in
unit-normalized coordinates (:class:`CumulativeFisher`), the per-run history and oracle curves the
plots read, the feasibility-gate budget CRLBs, and the closed-form NV zero_field_freq CRLB.
"""

from __future__ import annotations

import math
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


# ---------------------------------------------------------------------------
# Parameter scaling. The ridge in single_shot_marginal_stds_from_fim is absolute, so every
# FIM that reaches it must be in unit-normalized coordinates: parameter i scaled to [0, 1]
# of its own physical range. (Hz-scale widths ~1e6 next to a dimensionless contrast ~0.1
# differ by ~7 orders of magnitude; unnormalized, the ridge dominates every Hz direction
# and its CRLB saturates at a constant regardless of the data.)
# ---------------------------------------------------------------------------

# A direction whose unit-normalized cumulative FIM diagonal sits at or below this has
# accumulated no real information yet -- the ridge alone sets its "CRLB" -- so it is
# reported as NaN rather than as a fake number.
DEGENERATE_FIM_DIAG_FLOOR = 1e-3


def ranges_from_bounds(names: list[str], bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    """Physical width of each named parameter's bounds (1.0 when unbounded). shape: (n_params,)"""
    ranges = np.array([(bounds[n][1] - bounds[n][0]) if n in bounds else 1.0 for n in names], dtype=np.float64)
    ranges[ranges <= 0] = 1.0
    return ranges


def typed_parameters(model: SignalModel, values: dict[str, float]) -> Any:
    """The model's typed parameter object from a ``{name: physical value}`` mapping."""
    return model.spec.unpack_params([values[name] for name in model.parameter_names()])


def unit_normalized_fim(
    model: SignalModel,
    parameters: Any,
    x: float,
    obs: Observation | None,
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray | None:
    """Single-observation FIM for ``model`` (physical) w.r.t. parameters rescaled to ``[0, 1]`` of ``bounds``.

    Returns shape (n_params, n_params), or None when no gradient is obtainable. Scaling each gradient
    component by ``ranges[k]`` scales entry ``(i, j)`` by ``ranges[i] * ranges[j]``.
    """
    fim = fisher_information_matrix(x=x, model=model, parameters=parameters, last_obs=obs, param_bounds=bounds)
    if fim is None:
        return None
    ranges = ranges_from_bounds(list(model.parameter_names()), bounds)
    return fim * np.outer(ranges, ranges)


class CumulativeFisher:
    """Running Fisher information of a physical model, held in unit-normalized coordinates.

    This is the one place a cumulative FIM is accumulated and turned into marginal CRLBs; the belief
    (per step), the plot history and the oracle curve all go through it.
    """

    def __init__(self, model: SignalModel, bounds: dict[str, tuple[float, float]]) -> None:
        self.model = model
        self.bounds = dict(bounds)
        self.names: list[str] = list(model.parameter_names())
        self.ranges = ranges_from_bounds(self.names, self.bounds)
        self.unit_fim = np.zeros((len(self.names), len(self.names)), dtype=np.float64)

    def add(self, x: float, parameters: Any, obs: Observation | None) -> bool:
        """Add the FIM of one observation at ``x``, evaluated at ``parameters``; False if no gradient exists.

        A drive-controlled model is evaluated at the observation's own drive.
        """
        model = self.model
        if getattr(model, "controls_drive", False):  # duck-typed test models have no drive attribute
            if obs is None or obs.drive is None:
                raise ValueError("CumulativeFisher.add: a drive-controlled model needs the observation's drive.")
            model = model.at_drive(obs.drive)
        fim = unit_normalized_fim(model, parameters, x, obs, self.bounds)
        if fim is None:
            return False
        self.unit_fim += fim
        return True

    def is_empty(self) -> bool:
        return not np.any(self.unit_fim != 0)

    def matrix_phys(self) -> np.ndarray:
        """Cumulative FIM in physical units. shape: (n_params, n_params)"""
        return self.unit_fim / np.outer(self.ranges, self.ranges)

    def marginal_crlbs(self, *, nan_below_floor: bool = False) -> dict[str, float]:
        """``sqrt(diag(pinv(FIM)))`` per parameter in physical units; ``{}`` before any information.

        With ``nan_below_floor`` a direction with no real information is NaN instead of ridge-limited.
        """
        if self.is_empty():
            return {}
        stds = single_shot_marginal_stds_from_fim(self.unit_fim, len(self.names)) * self.ranges
        diag = np.diag(self.unit_fim)
        return {
            name: (float("nan") if nan_below_floor and diag[j] <= DEGENERATE_FIM_DIAG_FLOOR else float(stds[j]))
            for j, name in enumerate(self.names)
        }

    def copy(self) -> CumulativeFisher:
        other = CumulativeFisher(self.model, self.bounds)
        other.unit_fim = self.unit_fim.copy()
        return other


def fisher_history(
    snapshots: list[Any],
    estimates_hist: list[dict[str, float]],
    param_names: list[str],
    physical_bounds: dict[str, tuple[float, float]],
    drive_freq_bounds_phys: tuple[float, float],
    *,
    true_params: dict[str, float] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, float]], bool]:
    """Per-step cumulative Fisher info of a run: ``(fisher_hist, fisher_bounds_hist, fim_is_degenerate)``.

    ``fisher_hist[i]`` is the cumulative FIM after snapshot ``i`` in physical units;
    ``fisher_bounds_hist[i]`` maps each parameter to its marginal CRLB (physical; NaN while the direction
    has no information). When ``true_params`` is provided and covers all model parameters, each observation
    is evaluated at the true parameter values (measuring the true Fisher information of the sampled points);
    otherwise it falls back to that step's own posterior estimate (``estimates_hist[i]``, physical), on the
    belief's *inner* physical model -- the unit-cube wrapper would re-interpret the physical values as
    ``[0, 1]`` fractions. Each snapshot's ``obs.drive_freq_unit`` is mapped to Hz over
    ``drive_freq_bounds_phys`` (the range it is a unit coordinate of) before evaluation.
    """
    inner_model = snapshots[0].belief.model
    inner_model = getattr(inner_model, "inner", inner_model)
    fisher = CumulativeFisher(inner_model, physical_bounds)
    if list(fisher.names) != list(param_names):
        raise ValueError(f"fisher_history: model parameters {fisher.names} != requested {param_names}")

    fixed_typed_params = (
        typed_parameters(inner_model, true_params)
        if true_params is not None and all(name in true_params for name in fisher.names)
        else None
    )

    fisher_hist: list[np.ndarray] = []
    fisher_bounds_hist: list[dict[str, float]] = []
    lo_phys, hi_phys = drive_freq_bounds_phys
    for s, est in zip(snapshots, estimates_hist, strict=True):
        drive_freq_phys = lo_phys + s.obs.drive_freq_unit * (hi_phys - lo_phys)
        eval_params = fixed_typed_params if fixed_typed_params is not None else typed_parameters(inner_model, est)
        fisher.add(drive_freq_phys, eval_params, s.obs)
        fisher_hist.append(fisher.matrix_phys())
        bounds_now = fisher.marginal_crlbs(nan_below_floor=True)
        fisher_bounds_hist.append(bounds_now or {name: float("nan") for name in param_names})
    return fisher_hist, fisher_bounds_hist, fisher.is_empty()


def oracle_crlb_history(
    n_steps: int,
    model: SignalModel,
    true_typed_params: Any,
    x_lo: float,
    x_hi: float,
    noise_std: float,
    bounds: dict[str, tuple[float, float]],
    n_grid: int = 256,
) -> list[dict[str, float]]:
    """Best-achievable marginal CRLB per step: ``step + 1`` optimal dip measurements at the *true* parameters.

    For each parameter ``j``, an oracle that knows ``true_typed_params`` places its measurements at the
    most informative frequencies ``argmax_x I_jj(x)`` (for symmetric dip flanks, pairing the left and right
    inflection points also cancels off-diagonal nuisance couplings). By the Schur complement inequality
    ``[(sum_n I(x_n))^-1]_jj >= 1 / (N * max_x I_jj(x))``, this is a strict lower bound that no acquisition
    strategy's achieved CRLB can beat. Because CRLB scales as ``1 / sqrt(N)``, the peak single-measurement
    Fisher information is computed once over the grid and scaled by ``step + 1``.
    """
    names = list(model.parameter_names())
    ranges = ranges_from_bounds(names, bounds)
    noise_only_obs = Observation(drive_freq_unit=0.0, signal_value=0.0, noise_std=noise_std)  # only noise_std is read
    fims: list[np.ndarray] = []
    for xi in np.linspace(x_lo, x_hi, n_grid):
        fim = unit_normalized_fim(model, true_typed_params, float(xi), noise_only_obs, bounds)
        if fim is not None:
            fims.append(fim)
    if not fims:
        return [{} for _ in range(n_steps)]

    fim_stack = np.asarray(fims, dtype=np.float64)  # shape: (n_valid, n_params, n_params)
    max_diag = np.diagonal(fim_stack, axis1=1, axis2=2).max(axis=0)  # shape: (n_params,)
    oracle_fim = np.diag(max_diag)

    history: list[dict[str, float]] = []
    for step in range(n_steps):
        fim_at_step = oracle_fim * (step + 1)
        stds = single_shot_marginal_stds_from_fim(fim_at_step, len(names))
        diag_step = np.diag(fim_at_step)
        history.append(
            {
                name: float(stds[j] * ranges[j]) if diag_step[j] > DEGENERATE_FIM_DIAG_FLOOR else float("nan")
                for j, name in enumerate(names)
            }
        )
    return history


# ---------------------------------------------------------------------------
# Closed-form zero_field_freq CRLB for uniform sampling of NV dips.
#
# For uniformly spaced probes (rho = n_obs / bandwidth measurements per Hz) of a single
# population-normalized dip of amplitude a and height-normalized shape V (peak 1):
#     I(f) = (rho / sigma^2) * a^2 * J,    CRLB_f = sqrt(sigma^2 / (rho * a^2 * J)),  J = int (V')^2 dx.
# For a Lorentzian of HWHM Omega, J = pi / (4 Omega) (verified by numerical integration of the coded
# lineshape), i.e. CRLB_f^2 = 4 sigma^2 Omega / (pi a^2 rho).
# ---------------------------------------------------------------------------


def lorentzian_zero_field_freq_crlb(
    hwhm: float, contrast_total: float, noise_std: float, n_obs: int, bandwidth: float
) -> float:
    """Closed-form zero_field_freq CRLB (Hz) for ``n_obs`` uniform probes of a Lorentzian dip; ``inf`` if degenerate."""
    if hwhm <= 0 or contrast_total <= 0 or noise_std <= 0 or bandwidth <= 0 or n_obs <= 0:
        return math.inf
    rho = n_obs / bandwidth
    variance = (4.0 * noise_std**2 * hwhm) / (math.pi * contrast_total**2 * rho)
    return math.sqrt(max(variance, 0.0))


def uniform_steps_for_zero_field_freq_crlb(
    hwhm: float, contrast_total: float, noise_std: float, bandwidth: float, target_std: float
) -> float:
    """Uniform probes a Lorentzian dip needs for :func:`lorentzian_zero_field_freq_crlb` to reach ``target_std``."""
    return (4.0 * noise_std**2 * hwhm * bandwidth) / (math.pi * contrast_total**2 * target_std**2)


def zero_field_freq_crlb(
    inner_model: Any, estimates: dict[str, float], noise_std: float, n_obs: int, bandwidth: float
) -> float:
    """Closed-form zero_field_freq CRLB (Hz) at the physical ``estimates`` for the NV Lorentzian/Voigt models.

    Voigt-type lineshapes use the single-dip pseudo-Voigt ``J = int (V')^2 dx`` without the
    Lorentzian x Gaussian cross-term -- a conservative (larger) CRLB, exact as ``sigma_inhom -> 0``.
    Returns ``inf`` for unsupported models or degenerate inputs.
    """
    from nvision.spectra.numba_kernels import _pv_factors
    from nvision.spectra.nv_center import (
        NV_SATURATION_C_MAX,
        NVCenterLorentzianModel,
        NVCenterSaturationVoigtModel,
        NVCenterVoigtModel,
        _saturation_voigt_reparam_scalar,
        _voigt_reparam_scalar,
    )

    if n_obs <= 0 or noise_std <= 0 or bandwidth <= 0:
        return math.inf

    if isinstance(inner_model, NVCenterLorentzianModel):
        return lorentzian_zero_field_freq_crlb(
            estimates.get("hwhm", 0.0), estimates.get("contrast_total", 0.0), noise_std, n_obs, bandwidth
        )

    if isinstance(inner_model, NVCenterSaturationVoigtModel):
        if "saturation" not in estimates or "sigma_inhom" not in estimates:
            return math.inf
        fwhm_total, lorentz_frac, contrast_total = _saturation_voigt_reparam_scalar(
            estimates["saturation"], estimates["sigma_inhom"], NV_SATURATION_C_MAX
        )
    elif isinstance(inner_model, NVCenterVoigtModel):
        if not {"hwhm_lorentz", "sigma_inhom", "contrast_total"} <= estimates.keys():
            return math.inf
        if estimates["hwhm_lorentz"] <= 0:
            return math.inf
        fwhm_total, lorentz_frac = _voigt_reparam_scalar(estimates["hwhm_lorentz"], estimates["sigma_inhom"])
        contrast_total = estimates["contrast_total"]
    else:
        return math.inf

    if fwhm_total <= 0 or contrast_total <= 0:
        return math.inf
    elf, egf, nhs, gamma2, has_gamma, has_sigma = _pv_factors(fwhm_total, lorentz_frac)
    j_lorentz = (elf**2) * math.pi / (4.0 * gamma2**2.5) if has_gamma else 0.0
    j_gauss = (egf**2) * math.sqrt(-math.pi * nhs / 2.0) if has_sigma and nhs < 0 else 0.0
    j_total = j_lorentz + j_gauss
    if j_total <= 0:
        return math.inf
    rho = n_obs / bandwidth
    return math.sqrt(max(noise_std**2 / (rho * contrast_total**2 * j_total), 0.0))
