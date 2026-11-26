"""Refine rates of cached runs: rebuild what a run measured, then summarise a use case.

The cache keeps no belief snapshots, so the oracle efficiency ``eta_t`` of a cached run is rebuilt from three things it
does keep:

* the **true signal**: ``generator.generate(Random(repeat_seed_int(signal_repeat_key(seed, generator, repeat))),
  repeat_idx=repeat, n_repeats=N)`` with ``N`` the study's total repeat count, the same draw every locator of the
  repeat saw. It is compared with the stored ``true_params`` and a mismatch raises (the
  generator code changed since the run);
* **where each measurement was taken**: Bayesian-SBED's from the scan figure (``measurements.x``, float16 rescaled
  into the drive window, so about 1e-3 of the window, far below the HWHM); a plain sweep's from its design (the grid
  ``SimpleSweep-{k}perFWHM`` repeats pass after pass, :func:`~nvision.metrics.refine_rates.sweep_positions_phys`),
  because its figure is not stored;
* the **dip detection step** and the run's scalars from the stored row.

Every locator takes one shot per measurement (no strategy config sets ``n_shots``), so a measurement count is also a
shot count and ``I_max = i_max`` is per measurement.
"""

from __future__ import annotations

import base64
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from nvision.metrics import refine_rates as rr
from nvision.metrics.run_axes import primary_sensitivity_at, primary_sensitivity_curve
from nvision.runner.repeat_keys import repeat_seed_int, signal_repeat_key
from nvision.sim.grid_enums import optimal_sweep_multiplier

# Gains sigma_0 / tau the cost curves are evaluated at, sigma_0 = W / sqrt(12) the prior's spread (Eq. prior).
COST_GAINS = np.geomspace(1.5, 80.0, 24)
# Offset histogram: edges in units of the true HWHM gamma.
HIST_EDGES = np.arange(-8.0, 8.0 + 1e-9, 0.2)
# A dip of the clean signal is a local minimum at least this deep, as a share of the signal's own range.
_DIP_DEPTH_SHARE = 0.3
SBED = "Bayesian-SBED"
# SBED with the line shape known (the lab mirror and tracking); its positions come from the scan figure like SBED's.
SBED_KNOWN_SHAPE = "Bayesian-SBED-KnownShape"
SBED_STRATEGIES = (SBED, SBED_KNOWN_SHAPE)
_SWEEP_RE = re.compile(r"^SimpleSweep-(\d+(?:\.\d+)?)perFWHM$")


def is_sweep(strategy: str) -> bool:
    """True for the plain sweeps: ``SimpleSweep-{k}perFWHM`` and ``SimpleSweep-optimal[-x<m>]``."""
    return optimal_sweep_multiplier(strategy) is not None or _SWEEP_RE.match(strategy) is not None


def sweep_divisor(strategy: str, row: dict[str, Any] | None = None) -> float | None:
    """Grid points per prior FWHM of a sweep run, ``None`` for any other strategy.

    ``k`` of ``SimpleSweep-{k}perFWHM``; for ``SimpleSweep-optimal[-x<m>]`` the grid the run realised (``sweep_k`` of
    its results ``row``, i.e. ``m * k*``, which must then be given and carry it).
    """
    if optimal_sweep_multiplier(strategy) is not None:
        if row is None or row.get("sweep_k") is None:
            raise ValueError(f"{strategy} needs the run's realised sweep_k.")
        return float(row["sweep_k"])
    m = _SWEEP_RE.match(strategy)
    return float(m.group(1)) if m else None


@dataclass
class RunTrace:
    """One repeat of one locator, with what its efficiency needs. Arrays are over the run's measurements."""

    strategy: str
    repeat: int
    # The three arrays below, and the three theory fields, are None for an SBED repeat whose scan figure was never built
    # (``NVISION_GRAPH_REPEATS``: only the first repeats of a combo keep one); its series and scalars still count.
    x_phys: np.ndarray | None  # shape (n_steps,), drive frequency (Hz) of every measurement
    eta: np.ndarray | None  # shape (n_steps,), Fisher efficiency of every measurement, in [0, 1]
    offsets_gamma: np.ndarray | None  # shape (n_steps,), signed offset from the nearest true line centre, units of HWHM
    steps: np.ndarray  # shape (m,), downsampled per-step series: measurement counts
    errs: np.ndarray  # shape (m,), true error |estimate - truth| of the primary parameter (Hz)
    end: int  # measurements of the run
    dip_step: int | None  # 1-indexed ``detection_step``
    capped: bool  # hit the shared measurement budget without the stop rule firing
    fisher_star: float  # I*, Hz^-2 per measurement
    sigma0_phys: float  # standard deviation of the primary parameter's prior (Hz), W / sqrt(12) when uniform
    fwhm_phys: float  # single-line FWHM of the true signal (Hz)
    window_phys: float
    q_sweep_exact: float | None  # mean(g^2) / max(g^2) over the window
    peak_offset_gamma: float | None  # |offset| of the steepest point from the nearest line centre, units of the HWHM
    n_lines: int | None  # dips of the true signal
    j_tau: float | None = None  # posterior Fisher information at the detection (Hz^-2)
    j_tau_2h: float | None = None  # the same at twice the KDE bandwidth (the other end of its factor-2 bracket)

    @property
    def placement_efficiency(self) -> float | None:
        """Mean ``eta`` after the dip detection; ``None`` if no dip was detected or the run ended there."""
        return None if self.dip_step is None or self.eta is None else rr.placement_efficiency(self.eta, self.dip_step)


def line_centres_phys(true_signal: Any, x_lo_phys: float, x_hi_phys: float) -> np.ndarray:
    """Drive frequencies (Hz) of the dips of the noise-free ``true_signal`` inside the window.

    Local minima at least ``_DIP_DEPTH_SHARE`` of the signal's range deep, found on a fine grid, so they are the lines
    as they are (a hyperfine triplet gives three, an unresolved one a single merged line). Shape (n_lines,).
    """
    spec = true_signal.model.spec
    params = spec.unpack_params([float(v) for v in spec.pack_params(true_signal.typed_parameters)])
    xs = np.linspace(x_lo_phys, x_hi_phys, 20001)
    y = true_signal.baseline_model.compute_many_float64(xs, params)
    deep = y.max() - y >= _DIP_DEPTH_SHARE * (y.max() - y.min())
    inner = (y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]) & deep[1:-1]
    centres = xs[1:-1][inner]
    if centres.size == 0:
        raise ValueError("The true signal has no dip inside the drive window.")
    return centres


def _check_true_params(true_signal: Any, stored: dict[str, float]) -> None:
    """Raise unless the rebuilt true signal has the stored parameters (the generator is as it was at the run)."""
    for name, value in stored.items():
        rebuilt = true_signal.get_param_value(name)
        if not np.isclose(rebuilt, value, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"Rebuilt true signal differs from the stored one in {name}: {rebuilt!r} vs {value!r}. "
                "The generator or the seed derivation changed since this run; re-run it."
            )


def rebuild_true_signal(
    generator: Any, generator_name: str, repeat_idx: int, row: dict[str, Any], scan: dict[str, Any]
) -> Any:
    """The true signal repeat ``repeat_idx`` of the run saw, checked against the stored ``true_params``.

    Drawn as the executor draws it (``RunTaskExecutor._build_signal_experiment``): the repeat's own seed, and
    ``repeat_idx`` / ``n_repeats`` because the generator stratifies ``zeeman_shift`` across the repeats.
    ``n_repeats`` is the study's total repeat count (``task.repeat_total or repeats``), which the result row stores
    as ``repeats``; the combo's achieved count is not used, because it is smaller while a run is in progress.
    Raises ``ValueError`` when the row lacks it or the rebuilt parameters differ from the stored ones.
    """
    n_repeats = row.get("repeats")
    if n_repeats is None:
        raise ValueError("The result row has no total repeat count ('repeats'), so the true signal cannot be rebuilt.")
    true_signal = generator.generate(
        random.Random(repeat_seed_int(signal_repeat_key(row["seed"], generator_name, repeat_idx))),
        repeat_idx=repeat_idx,
        n_repeats=int(n_repeats),
    )
    _check_true_params(true_signal, (scan.get("true_params") or {}).get("params") or {})
    return true_signal


def _sbed_positions_phys(scan_entry: dict[str, Any], n_steps: int, x_lo_phys: float, x_hi_phys: float) -> np.ndarray:
    """Drive frequencies (Hz) of a run; shape (n_steps,), one per measurement, in order.

    From ``meas_drive_freq_unit`` (float32 unit coordinates, stored for every repeat of every run since the
    full-budget change) or else, for SBED, from the scan figure (float16 rescaled into the window; only the first
    ``NVISION_GRAPH_REPEATS`` repeats).
    """
    from nvision.viz._f32_json import from_gz_bytes

    if "meas_drive_freq_unit" in scan_entry:
        unit = np.frombuffer(base64.b64decode(scan_entry["meas_drive_freq_unit"]), dtype="<f4").astype(np.float64)
        if unit.size != n_steps:
            raise ValueError(f"The run stored {unit.size} measurement positions, it made {n_steps}.")
        return x_lo_phys + unit * (x_hi_phys - x_lo_phys)

    if "content_bin" not in scan_entry:
        raise ValueError("The scan figure of this run is not stored, so where SBED measured is unknown.")
    figure = from_gz_bytes(base64.b85decode(scan_entry["content_bin"]))
    meas = figure.get("measurements") or {}
    if meas.get("mode") != "steps":
        raise ValueError(f"Unexpected measurements mode {meas.get('mode')!r} in the scan figure (need 'steps').")
    # The step index is stored as float16, which cannot hold integers above 2048 exactly (error up to 2^-10 of the
    # value); the measurements themselves are in step order, so only the count and the rough index are checked.
    steps = np.asarray(meas["step"], dtype=np.float64)
    expected = np.arange(n_steps, dtype=np.float64)
    if steps.size != n_steps or np.any(np.abs(steps - expected) > 1.0 + 2.0**-10 * expected):
        raise ValueError(f"The scan figure holds {steps.size} measurements, the run made {n_steps}, or not in order.")
    return np.asarray(meas["x"], dtype=np.float64)


def _sweep_prior_fwhm_phys(true_signal: Any) -> float:
    """FWHM (Hz) the sweep's belief prior implies: twice the effective HWHM of the prior-mean parameters.

    Mirrors :func:`nvision.sim.locs.search_grid.prior_fwhm_phys` from the generator's own prior (``bounds['_priors']``,
    physical mean and std), so no belief has to be built. A prior mean outside its bounds is clipped to them, as the
    belief does on its unit cube; a parameter without a prior sits at the midpoint of its bounds.
    """
    from nvision.spectra.nv_center import effective_hwhm

    priors = true_signal.bounds.get("_priors") or {}
    means: dict[str, float] = {}
    for name in ("hwhm", "hwhm_lorentz", "sigma_inhom", "saturation"):
        if name not in true_signal.bounds:
            continue
        lo, hi = true_signal.bounds[name]
        prior = priors.get(name)
        means[name] = min(max(float(prior[0]), lo), hi) if prior is not None else 0.5 * (lo + hi)
    return 2.0 * float(effective_hwhm(means))


def build_run_trace(
    strategy: str,
    generator: Any,
    generator_name: str,
    repeat_idx: int,
    entries: list[dict[str, Any]],
    row: dict[str, Any],
) -> RunTrace:
    """Rebuild one cached repeat of ``strategy`` (:data:`SBED` or a ``SimpleSweep-{k}perFWHM``) as a :class:`RunTrace`.

    ``generator``: the generator object of ``generator_name``; ``repeat_idx``: the repeat's index in the combo (the
    signal seed). ``entries``, ``row``: the cached repeat (the scan entry, with its figure for SBED, and the result
    row). Raises ``ValueError`` when the run cannot be rebuilt or does not reproduce its own stored stage
    sensitivities; nothing is guessed.
    """
    scan = next((e for e in entries if e.get("type") == "scan"), None)
    if scan is None:
        raise ValueError("The repeat has no scan entry.")
    primary = row.get("primary_param")
    if not primary or row.get("i_max") is None:
        raise ValueError("The run has no run axes (primary parameter / I_max); re-run it.")
    series = scan.get("series") or {}
    if not series.get("step"):
        raise ValueError("The run has no per-step series.")
    n_steps = int(row["measurements"])
    errs = np.array([np.nan if v is None else v for v in series["error"]], dtype=np.float64)
    steps = np.asarray(series["step"], dtype=np.float64)
    keep = np.isfinite(errs)
    common = {
        "strategy": strategy,
        "repeat": repeat_idx,
        "steps": steps[keep],
        "errs": errs[keep],
        "end": n_steps,
        "dip_step": None if row.get("detection_step") is None else int(row["detection_step"]),
        "capped": row.get("failure_reason") == "max_steps",
        "fisher_star": float(row["i_max"]),
        "sigma0_phys": float(row["sigma0_phys"]),
        "fwhm_phys": float(row["fwhm_phys"]),
        "window_phys": float(row["window_phys"]),
        "j_tau": _float_or_none(row.get("j_tau_phys")),
        "j_tau_2h": _float_or_none(row.get("j_tau_2h_phys")),
    }
    is_sbed = strategy in SBED_STRATEGIES
    if is_sbed and scan.get("plot_skipped") and "meas_drive_freq_unit" not in scan:
        return RunTrace(
            x_phys=None,
            eta=None,
            offsets_gamma=None,
            q_sweep_exact=None,
            peak_offset_gamma=None,
            n_lines=None,
            **common,
        )

    true_signal = rebuild_true_signal(generator, generator_name, repeat_idx, row, scan)
    if true_signal.model.controls_drive:
        raise ValueError("Drive-controlled signals are not covered (their sensitivity depends on the drive setting).")
    x_lo, x_hi = true_signal.get_param_bounds("zero_field_freq")
    if abs((x_hi - x_lo) - float(row["window_phys"])) > 1e-6 * (x_hi - x_lo):
        raise ValueError("The rebuilt drive window differs from the stored one.")

    if is_sbed or "meas_drive_freq_unit" in scan:
        x_phys = _sbed_positions_phys(scan, n_steps, x_lo, x_hi)
    elif is_sweep(strategy):
        k = sweep_divisor(strategy, row)
        n_points = rr.sweep_n_points(x_hi - x_lo, _sweep_prior_fwhm_phys(true_signal), k)
        x_phys = rr.sweep_positions_phys(x_lo, x_hi, n_points, n_steps)
    else:
        raise ValueError(f"{strategy!r} is not Bayesian-SBED(-KnownShape) or a SimpleSweep.")

    sens = primary_sensitivity_at(true_signal, primary, x_lo, x_hi, x_phys, [None] * n_steps)
    eta = sens.measured / max(sens.best, float(sens.measured.max()))

    grid, g2 = primary_sensitivity_curve(true_signal, primary, x_lo, x_hi)
    q_sweep, _ = rr.exact_sweep_ratio(g2)
    centres = line_centres_phys(true_signal, x_lo, x_hi)
    gamma = 0.5 * float(row["fwhm_phys"])
    offsets = rr.nearest_centre_offsets(x_phys, centres, gamma)
    peak_offset = abs(float(rr.nearest_centre_offsets(grid[[int(np.argmax(g2))]], centres, gamma)[0]))
    return RunTrace(
        x_phys=x_phys,
        eta=eta,
        offsets_gamma=offsets,
        q_sweep_exact=q_sweep,
        peak_offset_gamma=peak_offset,
        n_lines=int(centres.size),
        **{**common, "window_phys": float(x_hi - x_lo)},
    )


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _clean(x: Any) -> Any:
    """JSON-ready: NaN / inf become ``None``, arrays become lists."""
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, float | np.floating):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _locator_summary(traces: list[RunTrace], i_max: float, taus: np.ndarray) -> dict[str, Any]:
    """Placement, rate fit, cost curve and offset histogram of one locator in one use case."""
    placement_efficiencies = [q for t in traces if (q := t.placement_efficiency) is not None]
    n_grid = np.linspace(1.0, float(max(t.end for t in traces)), 240)
    inv = rr.inverse_mse_curve(
        [t.steps for t in traces], [t.errs for t in traces], np.array([t.end for t in traces]), n_grid
    )
    window = rr.late_window(inv, [t.dip_step for t in traces if t.dip_step is not None])
    fit = None if window is None else rr.fit_rate_line(inv, i_max, window)
    # Mean eta over the SAME measurements the rate line was fitted on, so the fitted rate / placement_efficiency_window
    # compares like with like
    # (placement_efficiency also counts the localize stage, where the placement is still improving).
    in_window = (
        []
        if fit is None
        else [float(t.eta[int(fit.n_lo) - 1 : int(fit.n_hi)].mean()) for t in traces if t.eta is not None]
    )
    placement_efficiency_window = float(np.median(in_window)) if in_window else None
    cost = rr.cost_curve([t.steps for t in traces], [t.errs for t in traces], taus)

    def hist(part: str) -> dict[str, Any] | None:
        chunks = []
        for t in traces:
            if t.offsets_gamma is None:
                continue
            cut = t.end if t.dip_step is None else t.dip_step
            chunk = t.offsets_gamma[:cut] if part == "before" else t.offsets_gamma[cut:]
            if chunk.size:
                chunks.append(rr.offset_histogram(chunk, HIST_EDGES))
        if not chunks:
            return None
        return {
            "fraction": np.mean([c.fraction for c in chunks], axis=0),
            "outside": float(np.mean([c.outside for c in chunks])),
            "n_runs": len(chunks),
        }

    return {
        "n_runs": len(traces),
        "n_dip": sum(t.dip_step is not None for t in traces),
        "n_placement": len(placement_efficiencies),
        "capped_share": float(np.mean([t.capped for t in traces])),
        "placement_efficiency_median": _median(placement_efficiencies),
        "placement_efficiency_mean": float(np.mean(placement_efficiencies)) if placement_efficiencies else None,
        "fitted_rate": None if fit is None else fit._asdict(),
        "fitted_over_placement": None
        if fit is None or not placement_efficiencies
        else fit.rate / float(np.median(placement_efficiencies)),
        "placement_efficiency_window": placement_efficiency_window,
        "fitted_over_placement_window": None
        if fit is None or not placement_efficiency_window
        else fit.rate / placement_efficiency_window,
        "curve": {"n": n_grid, "inv_mse": inv.inv_mse},
        "cost": {"t_median": cost.t_median, "t_p75": cost.t_p75, "share": cost.share},
        "hist": {"before": hist("before"), "after": hist("after")},
        "_cost": cost,
    }


def use_case_summary(generator: str, noise: str, traces: dict[str, list[RunTrace]]) -> dict[str, Any]:
    """Everything the "Refine & speedup" section shows for one use case (one generator at one noise level).

    ``traces``: strategy name (:data:`SBED` or ``SimpleSweep-{k}perFWHM``) -> its repeats' :class:`RunTrace`.
    Returns a JSON-ready dict; see ``docs/evaluation_metrics.md`` for what each field is. The yardstick of the
    speedups is the best sweep spacing, never an arbitrary one (rule 3).
    """
    traces = {s: t for s, t in traces.items() if t}
    if not traces:
        raise ValueError("No runs.")
    every = [t for ts in traces.values() for t in ts]
    i_max = float(np.median([t.fisher_star for t in every]))
    sigma0 = float(np.median([t.sigma0_phys for t in every]))
    taus = sigma0 / COST_GAINS
    fwhm = float(np.median([t.fwhm_phys for t in every]))
    window = float(np.median([t.window_phys for t in every]))
    placed = [t for t in every if t.q_sweep_exact is not None]
    if not placed:
        raise ValueError("No run of this use case has its measurement positions (SBED figures stored, or a sweep).")
    q_exact = float(np.median([t.q_sweep_exact for t in placed]))

    locators = {s: _locator_summary(ts, i_max, taus) for s, ts in traces.items()}
    sweeps = [s for s in locators if is_sweep(s)]
    out: dict[str, Any] = {
        "generator": generator,
        "noise": noise,
        "snr": _snr(noise),
        "i_max": i_max,
        "sigma0_phys": sigma0,
        "gain": COST_GAINS,
        "tau_phys": taus,
        "hist_edges": HIST_EDGES,
        "window_phys": window,
        "fwhm_phys": fwhm,
        "window_over_fwhm": window / fwhm,
        "n_lines": int(np.median([t.n_lines for t in placed])),
        "theory": {
            "q_sweep_lorentzian": rr.lorentzian_sweep_q(fwhm, window),
            "speedup_lorentzian": rr.lorentzian_speedup(fwhm, window),
            "q_sweep_exact": q_exact,
            "speedup_exact": 1.0 / q_exact,
            "slope_offset_gamma": rr.LORENTZIAN_SLOPE_OFFSET_GAMMA,
            "peak_offset_gamma": float(np.median([t.peak_offset_gamma for t in placed])),
        },
        "locators": {s: {k: v for k, v in d.items() if k != "_cost"} for s, d in locators.items()},
    }
    if SBED in locators and sweeps:
        out["speedup"] = _speedup(locators, sweeps)
    return _clean(out)


def _speedup(locators: dict[str, dict[str, Any]], sweeps: list[str]) -> dict[str, Any]:
    """Speedup of SBED over the best sweep: ``T_sweep / T_SBED`` per target and ``R_inf`` measured two ways."""
    ratio = rr.speedup_curve(locators[SBED]["_cost"], [locators[s]["_cost"] for s in sweeps])
    sbed = locators[SBED]

    def best(get: Any) -> tuple[str | None, float | None]:
        vals = {s: get(locators[s]) for s in sweeps}
        vals = {s: v for s, v in vals.items() if v is not None and v > 0.0}
        return (max(vals, key=vals.get), max(vals.values())) if vals else (None, None)

    q_err_sbed = None if sbed["fitted_rate"] is None else sbed["fitted_rate"]["rate"]
    err_sweep, q_err_sweep = best(lambda d: None if d["fitted_rate"] is None else d["fitted_rate"]["rate"])
    place_sweep, placement_sweep = best(lambda d: d["placement_efficiency_median"])
    return {
        "ratio": ratio,
        "speedup_fitted": q_err_sbed / q_err_sweep if q_err_sbed and q_err_sweep else None,
        "best_sweep_fitted": err_sweep,
        "speedup_placement": sbed["placement_efficiency_median"] / placement_sweep
        if sbed["placement_efficiency_median"] and placement_sweep
        else None,
        "best_sweep_placement": place_sweep,
    }


def _snr(noise: str) -> float | None:
    """``x`` of a noise name ``SNR(x)``; ``None`` for any other noise."""
    m = re.fullmatch(r"SNR\(([\d.eE+-]+)\)", noise)
    return float(m.group(1)) if m else None
