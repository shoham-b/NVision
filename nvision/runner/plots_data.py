"""JSON data writers for time-varying Bayesian visualizations.

Each function replaces a Plotly HTML generator with a compact JSON file.
The frontend fetches the JSON and renders client-side, avoiding the
Python Plotly property-validation overhead (~11-18s per run).

All files are written as gzip-compressed JSON with Float32 typed-array
encoding (via ``nvision.viz._f32_json.dump_gz``).  Paths use the
``.json.gz`` extension.  The JS frontend decompresses on the fly with
the ``DecompressionStream`` API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvision.viz._f32_json import dump_gz

_PARAM_SCALES: dict[str, float] = {
    "frequency": 1e9,
    "linewidth": 1e6,
    "homogeneous_linewidth": 1e6,
    "sigma_inhom": 1e6,
    "split": 1e6,
    "fwhm_total": 1e6,
    "fwhm_lorentz": 1e6,
    "fwhm_gauss": 1e6,
    "zeeman_split": 1e6,
}

_PARAM_UNITS: dict[str, str] = {
    "frequency": "GHz",
    "linewidth": "MHz",
    "homogeneous_linewidth": "MHz",
    "sigma_inhom": "MHz",
    "split": "MHz",
    "fwhm_total": "MHz",
    "fwhm_lorentz": "MHz",
    "fwhm_gauss": "MHz",
    "zeeman_split": "MHz",
}


def _scale_param_dict(d: dict[str, float]) -> dict[str, float]:
    return {k: float(v) / _PARAM_SCALES.get(k, 1.0) for k, v in d.items()}


def _subsample_particles(
    particles: np.ndarray,
    weights: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a weighted random subsample of particles capped at n."""
    total = len(particles)
    if total <= n:
        return particles, weights
    weights_norm = weights / (weights.sum() + 1e-30)
    idx = np.random.choice(total, size=n, replace=False, p=weights_norm)
    sub_w = weights_norm[idx]
    sub_w = sub_w / sub_w.sum()
    return particles[idx], sub_w


def _weighted_robust_sigma(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted IQR/1.349 — a spread estimate that ignores rejuvenation particles.

    Every resample injects ``int(N * min_exploration_frac * exp(-step/25))``
    particles drawn from the prior (``_resample`` steps 8.5/8.6 in
    ``smc_marginal.py``). They are a vanishing fraction of the cloud and the next
    likelihood update kills them, but the standard deviation is quadratic in
    distance, so a couple of prior-drawn particles sitting half a bound-range away
    inflate it several-fold for exactly one step. The result is a sawtooth in the
    uncertainty trace that reads as the belief repeatedly widening and re-narrowing
    when nothing of the sort happened. The interquartile range is insensitive to
    them; /1.349 rescales it to a Gaussian-equivalent sigma so the two curves are
    directly comparable.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 0 or len(values) < 4:
        return float("nan")
    order = np.argsort(np.asarray(values, dtype=np.float64))
    v = np.asarray(values, dtype=np.float64)[order]
    # Midpoint CDF: the quantile of a particle is the mass strictly below it plus
    # half its own, which keeps the estimate unbiased for small clouds.
    cw = np.cumsum(w[order]) / total
    cw = cw - (w[order] / total) * 0.5
    q1, q3 = np.interp([0.25, 0.75], cw, v)
    return float((q3 - q1) / 1.349)


def write_posterior_data(
    anim_all: dict[str, tuple[list[np.ndarray], np.ndarray]],
    out_path: Path | None = None,
    *,
    true_params: dict[str, float] | None = None,
    resampled_steps: list[int] | None = None,
    physical_bounds: dict[str, tuple[float, float]] | None = None,
    n_particles: int = 60,
    ess_threshold: float | None = None,
    ess_history: list[float | None] | None = None,
    num_particles: int | None = None,
    param_hist: list[dict[str, float]] | None = None,
    convergence_threshold: float | None = None,
    absolute_thresholds: dict[str, float] | None = None,
) -> bytes | None:
    """Serialise particle posterior history to gzip-compressed Float32 JSON.

    Returns the compressed bytes (and writes to *out_path* if given).
    Returns ``None`` if there is nothing to serialise.
    """
    if not anim_all:
        return None

    param_names = list(anim_all.keys())
    first_param = param_names[0]
    n_steps = len(anim_all[first_param][0])

    steps: list[dict[str, Any]] = []
    for i in range(n_steps):
        step: dict[str, Any] = {}
        for param in param_names:
            history, grid = anim_all[param]
            arr = history[i]
            scale = _PARAM_SCALES.get(param, 1.0)

            unc_val = None
            if param_hist is not None and i < len(param_hist):
                val = param_hist[i].get(param)
                if val is not None:
                    unc_val = float(val) / scale

            if arr.ndim == 2 and arr.shape[1] == 2:
                raw_particles = arr[:, 0]
                raw_weights = arr[:, 1]
                robust_val = _weighted_robust_sigma(raw_particles, raw_weights) / scale
                particles, weights = _subsample_particles(raw_particles, raw_weights, n_particles)
                # ndarrays are passed through; dump_gz encodes them directly
                # to Float32 without materializing Python lists.
                step[param] = {
                    "type": "particles",
                    "values": particles / scale,
                    "weights": weights,
                }
                if np.isfinite(robust_val):
                    step[param]["uncertainty_robust"] = float(robust_val)
            else:
                step[param] = {
                    "type": "grid",
                    "axis": grid / scale,
                    "posterior": arr,
                }
            if unc_val is not None:
                step[param]["uncertainty"] = unc_val
        steps.append(step)

    bounds_out: dict[str, list[float]] = {}
    if physical_bounds:
        for p in param_names:
            if p in physical_bounds:
                lo, hi = physical_bounds[p]
                scale = _PARAM_SCALES.get(p, 1.0)
                bounds_out[p] = [float(lo) / scale, float(hi) / scale]

    abs_thresh_out = {}
    if absolute_thresholds:
        for p, val in absolute_thresholds.items():
            if p in param_names:
                scale = _PARAM_SCALES.get(p, 1.0)
                abs_thresh_out[p] = float(val) / scale

    payload = {
        "schema": "posterior_v1",
        "param_names": param_names,
        "param_units": {p: _PARAM_UNITS.get(p, "") for p in param_names},
        "physical_bounds": bounds_out,
        "true_params": _scale_param_dict({k: v for k, v in (true_params or {}).items() if k in param_names})
        if true_params
        else None,
        "resampled_steps": resampled_steps or [],
        # Pre-resample ESS over the *full* particle cloud, recorded by the belief.
        # The weights stored per step are a 60-particle subsample of that cloud and
        # are uniform on any step that resampled, so an ESS derived from them is
        # neither the filter's ESS nor comparable to ess_threshold * num_particles.
        "ess_history": ess_history or [],
        "num_particles": num_particles,
        "ess_threshold": ess_threshold,
        "convergence_threshold": convergence_threshold,
        "absolute_thresholds": abs_thresh_out,
        "steps": steps,
    }

    return dump_gz(payload, out_path)


def write_covariance_data(
    cov_hist: list[np.ndarray],
    param_names: list[str],
    pairs: list[tuple[int, int]],
    estimates_hist: list[dict[str, float]],
    out_path: Path | None = None,
    *,
    true_params: dict[str, float] | None = None,
    physical_bounds: dict[str, tuple[float, float]] | None = None,
) -> bytes | None:
    """Serialise covariance matrix history to gzip-compressed Float32 JSON."""
    if not cov_hist or not pairs:
        return None

    scales = np.array([_PARAM_SCALES.get(p, 1.0) for p in param_names])
    scale_outer = np.outer(scales, scales)

    steps = []
    for cov, means in zip(cov_hist, estimates_hist, strict=False):
        # Scale covariance to display units (ndarray encoded directly by dump_gz)
        steps.append(
            {
                "covariance": cov / scale_outer,
                "means": _scale_param_dict(means),
            }
        )

    bounds_out: dict[str, list[float]] = {}
    if physical_bounds:
        for p in param_names:
            if p in physical_bounds:
                lo, hi = physical_bounds[p]
                scale = _PARAM_SCALES.get(p, 1.0)
                bounds_out[p] = [float(lo) / scale, float(hi) / scale]

    payload = {
        "schema": "covariance_ellipses_v1",
        "param_names": param_names,
        "param_units": {p: _PARAM_UNITS.get(p, "") for p in param_names},
        "pairs": [list(pair) for pair in pairs],
        "physical_bounds": bounds_out,
        "true_params": _scale_param_dict({k: v for k, v in (true_params or {}).items() if k in param_names})
        if true_params
        else None,
        "steps": steps,
    }

    return dump_gz(payload, out_path)


def write_parameter_convergence_data(
    param_hist: list[dict[str, float]],
    estimates_hist: list[dict[str, float]],
    out_path: Path | None = None,
    *,
    true_params: dict[str, float] | None = None,
    convergence_threshold: float | None = None,
    absolute_thresholds: dict[str, float] | None = None,
) -> bytes | None:
    """Write per-step uncertainty and estimate history to JSON."""
    if not param_hist:
        return None

    param_names = list(param_hist[0].keys()) if param_hist else []

    steps = [
        {
            "uncertainties": _scale_param_dict(unc),
            "estimates": _scale_param_dict(est),
        }
        for unc, est in zip(param_hist, estimates_hist, strict=False)
    ]

    payload = {
        "schema": "parameter_convergence_v1",
        "param_names": param_names,
        "param_units": {p: _PARAM_UNITS.get(p, "") for p in param_names},
        "true_params": _scale_param_dict({k: v for k, v in (true_params or {}).items() if k in param_names})
        if true_params
        else None,
        # Convergence-limit reference line data, mirroring write_convergence_metrics_data.
        # Optional: absent on data written before this field existed.
        "convergence_threshold": convergence_threshold,
        "absolute_thresholds": absolute_thresholds or {},
        "steps": steps,
    }

    return dump_gz(payload, out_path)


def write_convergence_metrics_data(
    conv_metrics: list[dict[str, Any]],
    param_names: list[str],
    convergence_threshold: float,
    convergence_patience: int,
    out_path: Path | None = None,
    *,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    absolute_thresholds: dict[str, float] | None = None,
) -> bytes | None:
    """Write per-step convergence metric history to JSON."""
    if not conv_metrics:
        return None

    bounds_out: dict[str, list[float]] = {}
    if param_bounds:
        for p in param_names:
            if p in param_bounds:
                lo, hi = param_bounds[p]
                scale = _PARAM_SCALES.get(p, 1.0)
                bounds_out[p] = [float(lo) / scale, float(hi) / scale]

    payload = {
        "schema": "convergence_metrics_v1",
        "param_names": param_names,
        "param_units": {p: _PARAM_UNITS.get(p, "") for p in param_names},
        "convergence_threshold": float(convergence_threshold),
        "convergence_patience": int(convergence_patience),
        "param_bounds": bounds_out,
        "absolute_thresholds": absolute_thresholds or {},
        "steps": conv_metrics,
    }

    return dump_gz(payload, out_path)


def write_fisher_data(
    fisher_bounds_hist: list[dict[str, float]],
    actual_uncertainty_hist: list[dict[str, float]],
    fisher_hist: list[np.ndarray],
    param_names: list[str],
    out_path: Path | None = None,
    *,
    true_params: dict[str, float] | None = None,
    oracle_crlb_hist: list[dict[str, float]] | None = None,
) -> bytes | None:
    """Write Fisher information history (bounds + full FIM) to JSON.

    ``oracle_crlb_hist`` (optional, same length as the other histories) is the
    hard information limit for an ideal, uniformly-sampled design at the *true*
    parameters -- distinct from ``fisher_bounds_hist``, which is derived from
    this run's own actual measurements and estimates. Absent for older callers
    / when it couldn't be computed (e.g. no analytical or numerical gradient).
    """
    if not fisher_hist:
        return None

    scales = np.array([_PARAM_SCALES.get(p, 1.0) for p in param_names])
    # FIM scales as 1/variance, so scale FIM by 1/scale² per param pair
    fim_scale = np.outer(scales, scales)

    if oracle_crlb_hist is not None and len(oracle_crlb_hist) != len(fisher_hist):
        oracle_crlb_hist = None

    steps = []
    for i, (bounds, actuals, fim) in enumerate(
        zip(fisher_bounds_hist, actual_uncertainty_hist, fisher_hist, strict=False)
    ):
        step_entry = {
            "fisher_bounds": _scale_param_dict(bounds),
            "actual_uncertainty": _scale_param_dict(actuals),
            # Scale FIM to display units (ndarray encoded directly by dump_gz)
            "fisher_matrix": fim * fim_scale,
        }
        if oracle_crlb_hist is not None:
            step_entry["oracle_crlb"] = _scale_param_dict(oracle_crlb_hist[i])
        steps.append(step_entry)

    payload = {
        "schema": "fisher_v1",
        "param_names": param_names,
        "param_units": {p: _PARAM_UNITS.get(p, "") for p in param_names},
        "true_params": _scale_param_dict({k: v for k, v in (true_params or {}).items() if k in param_names})
        if true_params
        else None,
        "steps": steps,
    }

    return dump_gz(payload, out_path)


def write_matlab_freq_stats_data(
    freq_hz: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    min_vals: np.ndarray | None = None,
    max_vals: np.ndarray | None = None,
    out_path: Path | None = None,
) -> bytes | None:
    """Write per-frequency shot mean/std/min/max for a MATLAB run's "actual
    averages per frequency" view — an alternative to the sampled-measurements
    scatter, showing every recorded shot's per-bin average, spread, and extremes
    rather than just the subset the locator happened to visit. min_vals/max_vals
    are optional so older callers (and cached files) without them still decode.
    """
    if freq_hz is None or len(freq_hz) == 0:
        return None

    payload = {
        "schema": "matlab_freq_stats_v1",
        "freq_hz": np.asarray(freq_hz, dtype=np.float64),
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
    }
    if min_vals is not None and max_vals is not None:
        payload["min"] = np.asarray(min_vals, dtype=np.float64)
        payload["max"] = np.asarray(max_vals, dtype=np.float64)

    return dump_gz(payload, out_path)
