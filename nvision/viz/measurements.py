from __future__ import annotations

import contextlib
import json
import logging
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

from nvision.models.noise import CompositeOverFrequencyNoise
from nvision.noises.over_frequency.gaussian_noise import OverFrequencyGaussianNoise
from nvision.sim.batch import DataBatch
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.tools.paths import ensure_out_dir

# Match ``nvision.spectra.numba_kernels.nv_center_lorentzian_eval`` / Voigt no-split branch.
_NV_SPLIT_ZERO_TOL = 1e-10


def _align_mode_split_with_ground_truth(
    scan: Any,
    mode_estimates: Mapping[str, float],
) -> Mapping[str, float]:
    """If the true spectrum is zero-field NV (split≈0), evaluate the overlay at split=0.

    ``belief_mode_estimates`` uses a **marginal** argmax per grid parameter. The marginal
    mode for ``split`` can sit above zero even when the generated signal is a single
    combined dip, which would draw three dips in the dashed \"locator most likely\" curve.
    """
    if "split" not in mode_estimates:
        return mode_estimates
    ts = getattr(scan, "true_signal", None)
    if ts is None:
        return mode_estimates
    getv = getattr(ts, "get_param_value", None)
    if not callable(getv):
        return mode_estimates
    try:
        truth_split = float(getv("split"))
    except (KeyError, TypeError, ValueError):
        return mode_estimates
    if abs(truth_split) >= _NV_SPLIT_ZERO_TOL:
        return mode_estimates
    out = dict(mode_estimates)
    out["split"] = 0.0
    return out


def _noise_scale_for_scan(scan: Any, over_frequency_noise: CompositeOverFrequencyNoise | None) -> float:
    """Return noise scale (fixed to 1.0 to match the non-scaling CoreExperiment.measure)."""
    return 1.0


def _dense_model_curve(model: Any, xs: np.ndarray, values_in_order: Sequence[float]) -> np.ndarray | None:
    """Evaluate one parameter set over many ``xs`` in a single vectorized call.

    Uses the model's ``compute_vectorized_many`` kernel with length-1 parameter
    arrays — one batched kernel call instead of thousands of scalar
    ``compute`` calls. Returns ``None`` when the model has no compatible
    vectorized interface; callers fall back to the per-point loop.
    """
    fn = getattr(model, "compute_vectorized_many", None)
    if fn is None:
        return None
    try:
        arrays = [np.full(1, float(v), dtype=np.float32) for v in values_in_order]
        out = np.asarray(fn(np.asarray(xs, dtype=np.float32), arrays))
    except Exception:
        return None
    if out.ndim == 2 and out.shape[0] == len(xs) and out.shape[1] == 1:
        return out[:, 0].astype(float)
    return None


def _true_signal_label(scan: Any) -> str:
    """Legend/hover label for the dense ``y_dense`` curve.

    Simulated experiments generate it from a known parametric model
    (``scan.true_signal.model``) — "true signal" is accurate there. Real MATLAB
    runs have no such model (``_MatlabSignalProxy.model`` is always ``None``): the
    curve is just the per-bin mean over every recorded shot, not an independent
    reference the run is being checked against, so calling it "true" or "real"
    overclaims what it is.
    """
    model = getattr(getattr(scan, "true_signal", None), "model", None)
    return "true signal" if model is not None else "recorded mean signal"


def _true_signal_dense_y(scan: Any, xs: np.ndarray) -> np.ndarray:
    """Dense true-signal curve, vectorized when the model supports it."""
    true_signal = scan.true_signal
    model = getattr(true_signal, "model", None)
    if model is not None and hasattr(true_signal, "typed_parameters"):
        try:
            values = model.spec.pack_params(true_signal.typed_parameters)
        except Exception:
            values = None
        if values is not None:
            curve = _dense_model_curve(model, xs, values)
            if curve is not None:
                return curve
    return np.asarray([float(scan.signal(x)) for x in xs], dtype=float)


def _mode_dense_y_unit_cube(
    model: UnitCubeSignalModel,
    xs: np.ndarray,
    mode_estimates: Mapping[str, float],
) -> np.ndarray | list[float] | None:
    """MAP curve using a unit-cube forward model (physical ``xs`` and physical marginal modes)."""
    names = list(model.parameter_names())
    if not names or not all(name in mode_estimates for name in names):
        return None
    x_lo, x_hi = model.x_bounds_phys
    w = float(x_hi - x_lo)
    if w <= 0:
        return None
    xs_u = (np.asarray(xs, dtype=float) - x_lo) / w
    u_values: list[float] = []
    for name in names:
        lo, hi = model.param_bounds_phys[name]
        hw = float(hi - lo)
        v = float(mode_estimates[name])
        # Treat small split values as zero to show correct number of dips
        if name == "split" and v < _NV_SPLIT_ZERO_TOL:
            v = 0.0
        u = (v - lo) / hw if hw > 0 else 0.5
        u_values.append(min(max(u, 0.0), 1.0))
    curve = _dense_model_curve(model, xs_u, u_values)
    if curve is not None:
        return curve
    typed = model.spec.unpack_params(u_values)
    return [float(model.compute_from_params(float(xu), typed)) for xu in xs_u]


def _mode_belief_dense_y(
    scan: Any,
    xs: np.ndarray,
    mode_estimates: Mapping[str, float],
    *,
    belief_unit_cube: UnitCubeSignalModel | None = None,
) -> np.ndarray | list[float] | None:
    """Evaluate the forward model at ``mode_estimates`` along ``xs`` (physical domain).

    For :class:`~nvision.spectra.unit_cube.UnitCubeSignalModel`, ``mode_estimates`` are
    physical parameters and ``xs`` are physical probe positions (same as the true-signal
    plot); internally normalized coordinates are applied for evaluation.

    When ``belief_unit_cube`` is set (Bayesian runs), it is used instead of
    ``scan.true_signal.model`` so the dashed curve matches the inference model — e.g. NV
    Voigt ground truth with Lorentzian belief still gets a consistent MAP overlay.
    """
    mode_estimates = _align_mode_split_with_ground_truth(scan, mode_estimates)
    if belief_unit_cube is not None:
        return _mode_dense_y_unit_cube(belief_unit_cube, xs, mode_estimates)

    model = getattr(scan.true_signal, "model", None)
    bounds = getattr(scan.true_signal, "bounds", None)
    if model is None or bounds is None:
        return None
    names = list(model.parameter_names())
    if not names or not all(name in mode_estimates for name in names):
        return None

    if isinstance(model, UnitCubeSignalModel):
        return _mode_dense_y_unit_cube(model, xs, mode_estimates)

    values = [float(mode_estimates[name]) for name in names]
    # Treat small split values as zero to show correct number of dips
    if "split" in mode_estimates and float(mode_estimates["split"]) < _NV_SPLIT_ZERO_TOL:
        split_idx = names.index("split")
        values[split_idx] = 0.0
    curve = _dense_model_curve(model, xs, values)
    if curve is not None:
        return curve
    typed = model.spec.unpack_params(values)
    return [float(model.compute_from_params(float(x), typed)) for x in xs]


def _detect_dip_segments(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    max_dips: int | None = None,
) -> list[tuple[float, float]]:
    """Detect dip segments from dense signal evaluation using percentile thresholding."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if len(xs) < 3 or len(ys) < 3:
        return []
    order = np.argsort(xs)
    xs_s = xs[order]
    ys_s = ys[order]
    # Find significant local maxima (near baseline) and minima (deep dips),
    # then pair each minimum with its nearest flanking maxima.
    ymax = float(np.max(ys))
    ymin = float(np.min(ys))
    # Significant maxima must be near the baseline (top 90%)
    max_threshold = ymax - 0.1 * (ymax - ymin)
    # Significant minima must be well into the dip (below 40%)
    min_threshold = ymax - 0.4 * (ymax - ymin)

    maxima_indices = []
    minima_indices = []
    for i in range(1, len(ys_s) - 1):
        if ys_s[i] > ys_s[i - 1] and ys_s[i] > ys_s[i + 1] and float(ys_s[i]) > max_threshold:
            maxima_indices.append(i)
        elif ys_s[i] < ys_s[i - 1] and ys_s[i] < ys_s[i + 1] and float(ys_s[i]) < min_threshold:
            minima_indices.append(i)

    if not minima_indices:
        return []

    if max_dips is not None and len(minima_indices) > max_dips:
        minima_indices = sorted(minima_indices, key=lambda i: float(ys_s[i]))[:max_dips]
        minima_indices.sort()

    segments: list[tuple[float, float]] = []
    for min_idx in minima_indices:
        left_max = None
        for m in reversed(maxima_indices):
            if m < min_idx:
                left_max = m
                break
        right_max = None
        for m in maxima_indices:
            if m > min_idx:
                right_max = m
                break
        if left_max is not None and right_max is not None:
            segments.append((float(xs_s[left_max]), float(xs_s[right_max])))

    # Merge overlapping or adjacent segments
    if len(segments) <= 1:
        return segments
    segments.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = [segments[0]]
    for lo, hi in segments[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _compute_meas_dist_data(
    xs: np.ndarray,
    ys: np.ndarray | list[float],
    history: pl.DataFrame,
) -> dict[str, Any] | None:
    """Compute measurement distribution curve data for the lean data format."""
    if history.height == 0 or "x" not in history.columns:
        return None
    x_vals = history.get_column("x").to_list()
    x_meas = np.asarray([float(x) for x in x_vals if x is not None], dtype=float)
    if x_meas.size < 2:
        return None
    n_bins = max(60, min(400, int(np.sqrt(x_meas.size) * 20)))
    counts, edges = np.histogram(x_meas, bins=n_bins, range=(float(xs.min()), float(xs.max())))
    if counts.sum() <= 0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts.astype(float) / max(1.0, float(counts.max()))
    y_min = float(min(ys))
    y_max = float(max(ys))
    y_span = max(1e-9, y_max - y_min)
    y_band_base = y_min + 0.02 * y_span
    y_band_height = 0.25 * y_span
    y_curve = y_band_base + density * y_band_height
    return {
        "x": centers,
        "y_baseline": float(y_band_base),
        "y_curve": y_curve,
        "customdata": density * 100.0,
    }


def _json_safe_float(v: Any) -> float | None:
    """Finite floats only; NaN/inf become None for strict JSON manifests."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def _extract_history_xy(history: pl.DataFrame) -> tuple[list[Any], list[Any]]:
    xs_s = history.get_column("x").to_list() if "x" in history.columns else []
    ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
    return xs_s, ys_s


def _dense_xs_with_measurements(scan: Any, history_xs: list[Any], *, n_dense: int = 5000) -> np.ndarray:
    xs_base = np.linspace(scan.x_min, scan.x_max, n_dense)
    if not history_xs:
        return xs_base
    history_xs_arr = np.asarray([float(x) for x in history_xs if x is not None], dtype=float)
    if history_xs_arr.size == 0:
        return xs_base
    # Only include measurement xs that fall within the physical scan domain.
    # Old cached runs may have stored normalized [0, 1] x values; exclude them.
    in_range = (history_xs_arr >= scan.x_min) & (history_xs_arr <= scan.x_max)
    history_xs_arr = history_xs_arr[in_range]
    if history_xs_arr.size == 0:
        return xs_base
    return np.unique(np.concatenate([xs_base, history_xs_arr]))


def _compute_noisy_dense_values(
    xs: np.ndarray,
    ys: np.ndarray | list[float],
    over_frequency_noise: CompositeOverFrequencyNoise,
    noise_scale: float = 1.0,
    rng: random.Random | None = None,
) -> np.ndarray:
    dense_batch = DataBatch.from_arrays(xs, ys, meta={})
    noisy_batch = over_frequency_noise.apply(dense_batch, rng if rng is not None else random.Random(0))
    vals = np.asarray(noisy_batch.signal_values, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    bad = ~np.isfinite(vals)
    if noise_scale != 1.0:
        vals = ys_arr + (vals - ys_arr) * noise_scale
    vals[bad] = ys_arr[bad]
    return vals


def _compute_noisy_dense_band(
    xs: np.ndarray,
    ys: np.ndarray | list[float],
    over_frequency_noise: CompositeOverFrequencyNoise,
    noise_scale: float = 1.0,
    n_draws: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo envelope of where a re-measurement could plausibly land.

    Draws ``n_draws`` independent noise realizations over the dense true-signal
    curve and returns two nested percentile bands per x -- an inner ~1 sigma
    band (15.87/84.13 pct) and an outer ~2 sigma band (2.28/97.72 pct), so a
    single outlier draw doesn't dominate the width.
    """
    seed_rng = random.Random(seed)
    gaussian_parts = _gaussian_only_parts(over_frequency_noise)
    if gaussian_parts is not None and len(xs) > 1:
        draws = _gaussian_noisy_dense_draws(ys, gaussian_parts, noise_scale, n_draws, seed_rng)
    else:
        draws = np.empty((n_draws, len(xs)), dtype=float)
        for i in range(n_draws):
            draws[i] = _compute_noisy_dense_values(
                xs, ys, over_frequency_noise, noise_scale, rng=random.Random(seed_rng.randint(0, 2**31 - 1))
            )
    # One call: np.percentile partitions the (n_draws, n_x) array once for all four quantiles.
    lo1, hi1, lo2, hi2 = np.percentile(draws, [15.87, 84.13, 2.28, 97.72], axis=0)
    return lo1, hi1, lo2, hi2


def _gaussian_only_parts(noise: CompositeOverFrequencyNoise) -> list[OverFrequencyGaussianNoise] | None:
    """The noise's parts if it is a non-empty composite of plain Gaussian components, else ``None``."""
    parts = getattr(noise, "_parts", None)
    if parts and all(type(p) is OverFrequencyGaussianNoise for p in parts):
        return list(parts)
    return None


def _gaussian_noisy_dense_draws(
    ys: np.ndarray | list[float],
    parts: Sequence[OverFrequencyGaussianNoise],
    noise_scale: float,
    n_draws: int,
    seed_rng: random.Random,
) -> np.ndarray:
    """Bit-identical, Polars-free equivalent of ``n_draws`` calls to :func:`_compute_noisy_dense_values`.

    Reproduces the generic path's RNG stream exactly: one ``random.Random`` per draw, seeded from
    ``seed_rng``, from which each Gaussian component takes ``getrandbits(64)`` to seed its numpy
    generator (see ``OverFrequencyGaussianNoise.apply``'s bulk branch), then adds and clips in the
    same order. Skips the per-draw DataBatch/Polars round-trip, which dominated the cost.

    Returns shape ``(n_draws, len(ys))``: noisy signal values per draw over the dense x grid.
    """
    ys_arr = np.asarray(ys, dtype=float)
    n = ys_arr.size
    draws = np.empty((n_draws, n), dtype=float)
    for i in range(n_draws):
        rng = random.Random(seed_rng.randint(0, 2**31 - 1))
        vals = ys_arr
        for part in parts:
            noise = np.random.default_rng(rng.getrandbits(64)).normal(0.0, max(part.std, 0.0), size=n)
            vals = vals + noise
            if part.clip_min is not None or part.clip_max is not None:
                vals = np.clip(vals, part.clip_min, part.clip_max)
        draws[i] = vals
    bad = ~np.isfinite(draws)
    if noise_scale != 1.0:
        draws = ys_arr + (draws - ys_arr) * noise_scale
    return np.where(bad, ys_arr, draws)


def _splice_noisy_dense_at_measurements(
    xs: np.ndarray,
    noisy_vals: list[float],
    history_xs: list[Any],
    history_ys: list[Any],
) -> None:
    """In-place splice noisy dense curve so it passes through measurement points."""
    # Bypassed splicing to prevent wrong noise / spurious spikes artifact
    # in physical experiments with drift and coordinate/time dependencies.
    return


def _measurements_from_history(history: pl.DataFrame) -> dict[str, Any]:
    if history.height == 0:
        return {"mode": "empty"}

    if "phase" in history.columns:
        phases = history.get_column("phase").to_list()
        xs_s = history.get_column("x").to_list() if "x" in history.columns else []
        ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
        coarse_x: list[float] = []
        coarse_y: list[float | None] = []
        secondary_x: list[float] = []
        secondary_y: list[float | None] = []
        tertiary_x: list[float] = []
        tertiary_y: list[float | None] = []
        fine_x: list[float] = []
        fine_y: list[float | None] = []
        fine_step: list[int] = []
        for x, y, p in zip(xs_s, ys_s, phases, strict=False):
            if p == "coarse":
                coarse_x.append(float(x))
                coarse_y.append(_json_safe_float(y))
            elif p == "secondary":
                secondary_x.append(float(x))
                secondary_y.append(_json_safe_float(y))
            elif p == "tertiary":
                tertiary_x.append(float(x))
                tertiary_y.append(_json_safe_float(y))
            elif p == "fine":
                fine_x.append(float(x))
                fine_y.append(_json_safe_float(y))
                fine_step.append(len(fine_step))
        return {
            "mode": "phases",
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "secondary_x": secondary_x,
            "secondary_y": secondary_y,
            "tertiary_x": tertiary_x,
            "tertiary_y": tertiary_y,
            "fine_x": fine_x,
            "fine_y": fine_y,
            "fine_step": fine_step,
        }

    xs_s = history.get_column("x").to_list() if "x" in history.columns else []
    ys_s = history.get_column("signal_values").to_list() if "signal_values" in history.columns else []
    steps = list(range(history.height))
    result: dict[str, Any] = {
        "mode": "steps",
        "x": [float(x) for x in xs_s],
        "y": [_json_safe_float(y) for y in ys_s],
        "step": steps,
    }
    # Real acquisitions (e.g. MATLAB replay) scan every frequency once per sweep, then
    # scan them all again — so which *sweep* a shot came from is the real time axis, and a
    # better color choice than the locator's own adaptive visit order (`step` above), which
    # can revisit a bin many sweeps apart. Only include it when every point has one: a
    # locator can mix real and synthetic observations within one run (e.g. a warm start),
    # and a partial column would silently mis-color those without a real sweep index.
    if "sweep_index" in history.columns:
        sweep_idx_s = history.get_column("sweep_index").to_list()
        if sweep_idx_s and all(v is not None for v in sweep_idx_s):
            result["sweep_index"] = [int(v) for v in sweep_idx_s]
    return result


def compute_scan_plot_data(
    scan: Any,
    history: pl.DataFrame,
    over_frequency_noise: CompositeOverFrequencyNoise | None,
    focus_window: tuple[float, float] | None = None,
    mode_estimates: Mapping[str, float] | None = None,
    belief_unit_cube: UnitCubeSignalModel | None = None,
    narrowed_param_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Dense curve + measurement points for static UI head-to-head (matches ``plot_scan_measurements``).

    Unlike the internal gz-writer path, this public helper keeps a plain-JSON
    contract: all arrays are converted to Python lists before returning.
    """
    history_xs_s, history_ys_s = _extract_history_xy(history)
    xs = _dense_xs_with_measurements(scan, history_xs_s)
    ys = _true_signal_dense_y(scan, xs)
    out: dict[str, Any] = {
        "x_dense": xs.tolist(),
        "y_dense": ys.tolist(),
        "true_signal_label": _true_signal_label(scan),
    }
    if mode_estimates:
        y_mode = _mode_belief_dense_y(scan, xs, mode_estimates, belief_unit_cube=belief_unit_cube)
        if y_mode is not None and len(y_mode) == len(xs):
            out["y_dense_mode"] = np.asarray(y_mode).tolist()
    if over_frequency_noise is not None:
        noise_scale = _noise_scale_for_scan(scan, over_frequency_noise)
        noisy_vals = _compute_noisy_dense_values(xs, ys, over_frequency_noise, noise_scale)
        _splice_noisy_dense_at_measurements(xs, noisy_vals, history_xs_s, history_ys_s)
        out["y_dense_noisy"] = noisy_vals.tolist()

    has_metrics = history.height > 0 and any(col in history.columns for col in ("entropy", "max_prob", "uncertainty"))
    out["has_metrics"] = has_metrics
    out["measurements"] = _measurements_from_history(history)
    if focus_window is not None:
        lo, hi = float(focus_window[0]), float(focus_window[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out["focus_window"] = [lo, hi]
    if narrowed_param_bounds:
        safe_bounds: dict[str, list[float]] = {}
        for name, (lo, hi) in narrowed_param_bounds.items():
            flo, fhi = _json_safe_float(lo), _json_safe_float(hi)
            if flo is not None and fhi is not None and fhi > flo:
                safe_bounds[name] = [flo, fhi]
        if safe_bounds:
            out["narrowed_param_bounds"] = safe_bounds
    return out


def _parse_figure_from_scan_html(html: str) -> go.Figure | None:
    """Rebuild a :class:`plotly.graph_objects.Figure` from ``Plotly.newPlot`` JSON in saved HTML."""
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', html)
    if not m:
        return None
    pos = m.end()
    decoder = json.JSONDecoder()
    try:
        data, pos = decoder.raw_decode(html, pos)
    except json.JSONDecodeError:
        return None
    while pos < len(html) and html[pos] in " \t\n\r,":
        pos += 1
    try:
        layout, _pos = decoder.raw_decode(html, pos)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not isinstance(layout, dict):
        return None
    return go.Figure(data=data, layout=layout)


def _decode_plotly_array(v: Any) -> list[Any] | None:
    if v is None:
        return None
    if hasattr(v, "to_plotly_json"):
        v = v.to_plotly_json()
    if isinstance(v, dict) and "bdata" in v and "dtype" in v:
        import base64

        import numpy as np

        try:
            raw = base64.b64decode(v["bdata"])
            arr = np.frombuffer(raw, dtype=v["dtype"])
            return arr.tolist()
        except Exception:
            pass
    return v


def _trace_xy_lists(tr: Any) -> tuple[list[float], list[float | None]]:
    xs: list[float] = []
    ys: list[float | None] = []
    tx = getattr(tr, "x", None)
    ty = getattr(tr, "y", None)
    tx = _decode_plotly_array(tx)
    ty = _decode_plotly_array(ty)
    if tx is not None:
        for v in tx:
            xs.append(float(v))
    if ty is not None:
        for v in ty:
            ys.append(_json_safe_float(v))
    n = min(len(xs), len(ys))
    return xs[:n], ys[:n]


def plot_data_from_scan_figure(fig: go.Figure) -> dict[str, Any] | None:
    """Rebuild ``plot_data`` from a scan figure (must match ``plot_scan_measurements``)."""
    x_dense: list[float] | None = None
    y_dense: list[float] | None = None
    y_dense_noisy: list[float] | None = None
    coarse_x: list[float] = []
    coarse_y: list[float | None] = []
    fine_x: list[float] = []
    fine_y: list[float | None] = []
    fine_step: list[float] = []
    step_x: list[float] = []
    step_y: list[float | None] = []
    step_idx: list[float] = []

    has_metrics = any(getattr(t, "name", None) in ("Entropy", "Uncertainty") for t in fig.data)

    y_dense_mode: list[float] | None = None
    y_dense_sobol_mode: list[float] | None = None
    sobol_x: list[float] = []
    sobol_y: list[float | None] = []
    for tr in fig.data:
        name = getattr(tr, "name", None) or ""
        mode = getattr(tr, "mode", "") or ""
        if name in ("locator most likely signal", "locator mode belief signal") and "lines" in mode:
            _, ym = _trace_xy_lists(tr)
            if ym and all(v is not None for v in ym):
                y_dense_mode = [float(v) for v in ym if v is not None]
            continue
        if name == "sobol most likely signal" and "lines" in mode:
            _, ym = _trace_xy_lists(tr)
            if ym and all(v is not None for v in ym):
                y_dense_sobol_mode = [float(v) for v in ym if v is not None]
            continue
        if name == "true signal" and "lines" in mode:
            x_dense, y_dense = _trace_xy_lists(tr)
            continue
        if name == "simulated noisy signal (over-frequency)" and "lines" in mode:
            _, yn = _trace_xy_lists(tr)
            y_dense_noisy = []
            for i, v in enumerate(yn):
                if v is not None:
                    y_dense_noisy.append(float(v))
                elif y_dense is not None and i < len(y_dense) and y_dense[i] is not None:
                    y_dense_noisy.append(float(y_dense[i]))
                else:
                    y_dense_noisy.append(0.0)
            continue
        if name == "measurements (coarse)":
            coarse_x, coarse_y = _trace_xy_lists(tr)
            continue
        if name == "measurements (inference)":
            fine_x, fine_y = _trace_xy_lists(tr)
            mk = getattr(tr, "marker", None)
            if mk is not None:
                c = getattr(mk, "color", None)
                c = _decode_plotly_array(c)
                if c is not None and hasattr(c, "__iter__") and not isinstance(c, str | bytes):
                    fine_step = [float(v) for v in c]
            continue
        if name == "measurements (noisy)":
            step_x, step_y = _trace_xy_lists(tr)
            mk = getattr(tr, "marker", None)
            if mk is not None:
                c = getattr(mk, "color", None)
                c = _decode_plotly_array(c)
                if c is not None and hasattr(c, "__iter__") and not isinstance(c, str | bytes):
                    step_idx = [float(v) for v in c]
            continue
        if name == "sobol measurements (noisy)":
            sobol_x, sobol_y = _trace_xy_lists(tr)
            continue

    if x_dense is None or y_dense is None:
        return None

    out: dict[str, Any] = {
        "x_dense": [float(x) for x in x_dense],
        "y_dense": [float(y) for y in y_dense if y is not None],
        "has_metrics": has_metrics,
    }
    if y_dense_mode is not None and len(y_dense_mode) == len(out["x_dense"]):
        out["y_dense_mode"] = y_dense_mode
    if y_dense_sobol_mode is not None and len(y_dense_sobol_mode) == len(out["x_dense"]):
        out["y_dense_sobol_mode"] = y_dense_sobol_mode
    if y_dense_noisy is not None and len(y_dense_noisy) == len(out["x_dense"]):
        out["y_dense_noisy"] = y_dense_noisy

    if sobol_x:
        out["sobol_measurements"] = {
            "x": sobol_x,
            "y": [float(y) for y in sobol_y if y is not None],
        }

    if coarse_x or fine_x:
        if len(fine_step) != len(fine_x):
            fine_step = [float(i) for i in range(len(fine_x))]
        out["measurements"] = {
            "mode": "phases",
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "fine_x": fine_x,
            "fine_y": fine_y,
            "fine_step": [int(s) for s in fine_step],
        }
    elif step_x:
        if len(step_idx) != len(step_x):
            step_idx = [float(i) for i in range(len(step_x))]
        out["measurements"] = {
            "mode": "steps",
            "x": step_x,
            "y": step_y,
            "step": [int(s) for s in step_idx],
        }
    else:
        out["measurements"] = {"mode": "empty"}

    return out


def backfill_scan_plot_data_if_missing(entry: dict[str, Any], out_dir: Path) -> None:
    """If a scan manifest entry has no ``plot_data``, rebuild it from the saved scan file on disk."""
    if entry.get("type") != "scan" or entry.get("plot_data"):
        return
    rel = entry.get("path")
    if not isinstance(rel, str) or not rel.strip():
        return
    path = out_dir / rel
    if not path.exists():
        return
    try:
        if rel.endswith(".json.gz"):
            from nvision.viz._f32_json import from_gz_bytes

            raw = from_gz_bytes(path.read_bytes())
            if isinstance(raw, dict) and raw.get("_graph_type") == "scan":
                # New lean format — plot_data is a subset of the lean data
                plot_data: dict[str, Any] = {
                    k: raw[k] for k in ("x_dense", "y_dense", "has_metrics", "measurements") if k in raw
                }
                for k in (
                    "focus_window",
                    "narrowed_param_bounds",
                    "y_dense_noisy",
                    "y_dense_noisy_lo",
                    "y_dense_noisy_hi",
                    "y_dense_mode",
                ):
                    if k in raw:
                        plot_data[k] = raw[k]
                entry["plot_data"] = plot_data
                return
            # Legacy full Plotly figure JSON
            from nvision.viz._f32_json import figure_from_gz_bytes

            fig = figure_from_gz_bytes(path.read_bytes())
        else:
            html = path.read_text(encoding="utf-8")
            fig = _parse_figure_from_scan_html(html)
            if fig is None:
                return
        plot_data = plot_data_from_scan_figure(fig)
    except Exception as exc:
        logging.warning("Failed to backfill scan plot data for %s: %s", rel, exc)
        return
    if plot_data:
        entry["plot_data"] = plot_data


def _compute_scan_data_dict(
    scan: Any,
    history: pl.DataFrame,
    over_frequency_noise: CompositeOverFrequencyNoise | None,
    focus_window: tuple[float, float] | None = None,
    mode_estimates: Mapping[str, float] | None = None,
    belief_unit_cube: UnitCubeSignalModel | None = None,
    narrowed_param_bounds: dict[str, tuple[float, float]] | None = None,
    per_dip_windows: list[tuple[float, float]] | None = None,
    sobol_xs: list[float] | None = None,
    sobol_ys: list[float] | None = None,
    sobol_mode_estimates: Mapping[str, float] | None = None,
    sweep_xs: list[float] | None = None,
    sweep_ys: list[float] | None = None,
    sweep_mode_estimates: Mapping[str, float] | None = None,
    true_params: dict | None = None,
    found_params: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the lean scan data dict written to disk (replaces the full Plotly figure)."""
    history_xs_raw, _history_ys_raw = _extract_history_xy(history)
    xs = _dense_xs_with_measurements(scan, history_xs_raw, n_dense=5000)
    ys = _true_signal_dense_y(scan, xs)

    out: dict[str, Any] = {
        "_graph_type": "scan",
        "x_dense": xs,
        "y_dense": ys,
        "true_signal_label": _true_signal_label(scan),
    }

    if mode_estimates:
        y_mode = _mode_belief_dense_y(scan, xs, mode_estimates, belief_unit_cube=belief_unit_cube)
        if y_mode is not None and len(y_mode) == len(xs):
            out["y_dense_mode"] = y_mode

    # The locator's reported estimates (the same values as the final_est_* metrics), so the
    # UI's flip view can fold about the *found* center and Zeeman split. Real measurements
    # have no ground truth to fold about, and their center sits off the simulated grid's
    # fixed 2870 MHz anyway.
    if found_params:
        safe_found = {name: f for name, v in found_params.items() if (f := _json_safe_float(v)) is not None}
        if safe_found:
            out["found_params"] = safe_found

    if over_frequency_noise is not None:
        noise_scale = _noise_scale_for_scan(scan, over_frequency_noise)
        noisy_lo1, noisy_hi1, noisy_lo2, noisy_hi2 = _compute_noisy_dense_band(
            xs, ys, over_frequency_noise, noise_scale
        )
        out["y_dense_noisy_lo"] = noisy_lo1
        out["y_dense_noisy_hi"] = noisy_hi1
        out["y_dense_noisy_lo2"] = noisy_lo2
        out["y_dense_noisy_hi2"] = noisy_hi2

    has_metrics = history.height > 0 and any(col in history.columns for col in ("entropy", "max_prob", "uncertainty"))
    out["has_metrics"] = has_metrics
    out["measurements"] = _measurements_from_history(history)

    meas_dist = _compute_meas_dist_data(xs, ys, history)
    if meas_dist is not None:
        out["meas_dist"] = meas_dist

    if focus_window is not None:
        lo, hi = float(focus_window[0]), float(focus_window[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out["focus_window"] = [lo, hi]

    # Filter per-dip windows to expected dip count (same guard as old plotting code)
    if per_dip_windows:
        expected_dips: int | None = None
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            expected_dips = int(scan.true_signal.model.expected_dip_count())
        filtered = per_dip_windows
        if expected_dips is not None and len(per_dip_windows) > expected_dips:
            filtered = None
        if filtered:
            safe_windows: list[list[float]] = []
            for lo, hi in filtered:
                flo, fhi = _json_safe_float(lo), _json_safe_float(hi)
                if flo is not None and fhi is not None and fhi > flo:
                    safe_windows.append([flo, fhi])
            if safe_windows:
                out["per_dip_windows"] = safe_windows

    if narrowed_param_bounds:
        safe_bounds: dict[str, list[float]] = {}
        for name, (lo, hi) in narrowed_param_bounds.items():
            flo, fhi = _json_safe_float(lo), _json_safe_float(hi)
            if flo is not None and fhi is not None and fhi > flo:
                safe_bounds[name] = [flo, fhi]
        if safe_bounds:
            out["narrowed_param_bounds"] = safe_bounds

    if true_params and isinstance(true_params, dict):
        out["true_params"] = true_params

    if sobol_xs and sobol_ys:
        width = float(scan.x_max - scan.x_min)
        out["sobol_measurements"] = {
            "x": float(scan.x_min) + np.asarray(sobol_xs, dtype=float) * width,
            "y": np.asarray(sobol_ys, dtype=float),
        }
    if sobol_mode_estimates:
        y_sobol_mode = _mode_belief_dense_y(scan, xs, sobol_mode_estimates, belief_unit_cube=belief_unit_cube)
        if y_sobol_mode is not None and len(y_sobol_mode) > 0:
            out["sobol_mode_y"] = y_sobol_mode

    if sweep_xs and sweep_ys:
        width = float(scan.x_max - scan.x_min)
        out["sweep_measurements"] = {
            "x": float(scan.x_min) + np.asarray(sweep_xs, dtype=float) * width,
            "y": np.asarray(sweep_ys, dtype=float),
        }
    if sweep_mode_estimates:
        y_sweep_mode = _mode_belief_dense_y(scan, xs, sweep_mode_estimates, belief_unit_cube=belief_unit_cube)
        if y_sweep_mode is not None and len(y_sweep_mode) > 0:
            out["sweep_mode_y"] = y_sweep_mode

    if has_metrics:
        if "entropy" in history.columns:
            entropy = history.get_column("entropy").to_list()
            if any(x is not None and not np.isnan(float(x)) for x in entropy if x is not None):
                out["entropy"] = [_json_safe_float(x) for x in entropy]
        if "uncertainty" in history.columns:
            uncert = history.get_column("uncertainty").to_list()
            if any(x is not None and not np.isnan(float(x)) for x in uncert if x is not None):
                out["uncertainty"] = [_json_safe_float(x) for x in uncert]

    return out


class MeasurementsMixin:
    """Mixin for scan measurement plotting."""

    # Typing for mixin dependency (self.out_dir from VizBase)
    out_dir: Path

    def plot_scan_measurements(
        self,
        scan,
        history: pl.DataFrame,
        out_path: Path | None = None,
        over_frequency_noise: CompositeOverFrequencyNoise | None = None,
        mode_estimates: Mapping[str, float] | None = None,
        focus_window: tuple[float, float] | None = None,
        per_dip_windows: list[tuple[float, float]] | None = None,
        belief_unit_cube: UnitCubeSignalModel | None = None,
        narrowed_param_bounds: dict[str, tuple[float, float]] | None = None,
        sobol_xs: list[float] | None = None,
        sobol_ys: list[float] | None = None,
        sobol_mode_estimates: Mapping[str, float] | None = None,
        sweep_xs: list[float] | None = None,
        sweep_ys: list[float] | None = None,
        sweep_mode_estimates: Mapping[str, float] | None = None,
        true_params: dict | None = None,
        found_params: Mapping[str, float] | None = None,
    ) -> bytes:
        """Serialize scan data as a lean JSON.gz (definition lives in static/graphs/scan.json)."""
        if out_path is not None:
            ensure_out_dir(out_path.parent)

        data = _compute_scan_data_dict(
            scan,
            history,
            over_frequency_noise,
            focus_window=focus_window,
            mode_estimates=mode_estimates,
            belief_unit_cube=belief_unit_cube,
            narrowed_param_bounds=narrowed_param_bounds,
            per_dip_windows=per_dip_windows,
            sobol_xs=sobol_xs,
            sobol_ys=sobol_ys,
            sobol_mode_estimates=sobol_mode_estimates,
            sweep_xs=sweep_xs,
            sweep_ys=sweep_ys,
            sweep_mode_estimates=sweep_mode_estimates,
            true_params=true_params,
            found_params=found_params,
        )

        from nvision.viz._f32_json import dump_gz

        return dump_gz(data, out_path)
