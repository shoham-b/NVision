"""Per-repeat plot generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nvision.models.experiment import CoreExperiment
from nvision.models.observer import RunResult
from nvision.runner.convert import belief_mode_estimates
from nvision.viz import Viz
from nvision.viz.measurements import compute_scan_plot_data

log = logging.getLogger(__name__)


def _resolve_scan_param(strat_obj: Any, run_result: RunResult) -> str:
    """Parameter used for 1D posterior animation (matches BayesianLocator scan axis)."""
    if isinstance(strat_obj, dict):
        cfg = strat_obj.get("config") or {}
        sp = cfg.get("scan_param")
        if isinstance(sp, str) and sp.strip():
            return sp.strip()
    if run_result.snapshots:
        names = run_result.snapshots[0].belief.model.parameter_names()
        if names:
            return names[0]
    return "frequency"


def _extract_unit_cube_posterior_inputs(
    run_result: RunResult,
    scan_param: str,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    from nvision.signal.unit_cube_grid_belief import UnitCubeGridBeliefDistribution

    b0 = run_result.snapshots[0].belief
    if not isinstance(b0, UnitCubeGridBeliefDistribution):
        return None

    grid = b0.physical_param_grid(scan_param)
    hist: list[np.ndarray] = []
    for s in run_result.snapshots:
        b = s.belief
        assert isinstance(b, UnitCubeGridBeliefDistribution)
        p = next((pp for pp in b.parameters if pp.name == scan_param), None)
        if p is None or not hasattr(p, "posterior"):
            return None
        hist.append(p.posterior.copy())
    return hist, grid


def _extract_grid_posterior_inputs(
    run_result: RunResult,
    scan_param: str,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    from nvision.signal.grid_belief import GridBeliefDistribution

    b0 = run_result.snapshots[0].belief
    if not isinstance(b0, GridBeliefDistribution):
        return None

    p0 = b0.get_param(scan_param)
    if not hasattr(p0, "grid"):
        return None

    grid = p0.grid
    hist: list[np.ndarray] = []
    for s in run_result.snapshots:
        p = s.belief.get_param(scan_param)
        if not hasattr(p, "posterior"):
            return None
        hist.append(p.posterior.copy())
    return hist, grid


def _extract_smc_posterior_inputs(
    run_result: RunResult,
    scan_param: str,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    from nvision.signal.smc_belief import SMCBeliefDistribution

    b0 = run_result.snapshots[0].belief
    if not isinstance(b0, SMCBeliefDistribution):
        return None

    idx = b0._param_names.index(scan_param)
    hist: list[np.ndarray] = []
    for s in run_result.snapshots:
        b = s.belief
        assert isinstance(b, SMCBeliefDistribution)
        col = b._particles[:, idx]
        hist.append(col.reshape(-1, 1))
    # Unused for particle / histogram mode; required by API
    return hist, np.linspace(0.0, 1.0, 2)


def _posterior_animation_inputs(
    run_result: RunResult,
    scan_param: str,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """Build (posterior_history, freq_grid) for ``plot_posterior_animation``."""
    if not run_result.snapshots:
        return None

    for extractor in (
        _extract_unit_cube_posterior_inputs,
        _extract_grid_posterior_inputs,
        _extract_smc_posterior_inputs,
    ):
        result = extractor(run_result, scan_param)
        if result is not None:
            return result

    log.debug("No posterior animation extraction for belief type %s", type(run_result.snapshots[0].belief).__name__)
    return None


def _is_bayesian_run(strat_name: str, strat_obj: Any) -> bool:
    if "Bayesian" in strat_name:
        return True
    if isinstance(strat_obj, dict):
        cls = strat_obj.get("class")
        if isinstance(cls, type):
            from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

            try:
                return issubclass(cls, SequentialBayesianLocator)
            except TypeError:
                return False
    return False


def _initial_sweep_steps_from_strategy(strat_obj: Any) -> int:
    """Infer initial coarse sweep length from strategy config/class defaults."""
    if not isinstance(strat_obj, dict):
        return 0

    config = strat_obj.get("config", {}) or {}
    steps = int(config.get("initial_sweep_steps", 0) or 0)
    if steps > 0:
        return steps

    cls = strat_obj.get("class")
    if isinstance(cls, type):
        try:
            from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

            if issubclass(cls, SequentialBayesianLocator):
                default_steps = int(getattr(cls, "DEFAULT_INITIAL_SWEEP_STEPS", 0) or 0)
                return max(0, default_steps)
        except Exception:
            pass

        try:
            from nvision.sim.locs.coarse.two_phase_sweep_locator import TwoPhaseSweepLocator

            if cls is TwoPhaseSweepLocator:
                return int(config.get("phase1_max_steps", 0) or 0)
        except Exception:
            pass

    return 0


def _focus_window_from_history(
    history: pl.DataFrame,
    scan: CoreExperiment,
    strat_obj: Any,
) -> tuple[float, float] | None:
    """Infer Bayesian focus window from coarse-phase measurements.

    Mirrors the locator-side Sobol focus heuristic so the UI can show the region
    used by Bayesian acquisition after the initial sweep.
    """
    if (
        history.is_empty()
        or "phase" not in history.columns
        or "x" not in history.columns
        or "signal_values" not in history.columns
    ):
        return None

    coarse = history.filter(pl.col("phase") == "coarse")
    if coarse.height < 3:
        return None

    focus_info_quantile = 0.6
    focus_padding_fraction = 0.1
    focus_min_width_fraction = 0.15
    if isinstance(strat_obj, dict):
        cfg = strat_obj.get("config", {}) or {}
        focus_info_quantile = float(cfg.get("focus_info_quantile", focus_info_quantile))
        focus_padding_fraction = float(cfg.get("focus_padding_fraction", focus_padding_fraction))
        focus_min_width_fraction = float(cfg.get("focus_min_width_fraction", focus_min_width_fraction))
        if not bool(cfg.get("focus_after_sweep", True)):
            return None

    xs = np.asarray(coarse.get_column("x").to_list(), dtype=float)
    ys = np.asarray(coarse.get_column("signal_values").to_list(), dtype=float)
    if xs.size < 3 or ys.size < 3:
        return None

    center = float(np.median(ys))
    info = np.abs(ys - center)
    q = float(np.clip(focus_info_quantile, 0.0, 0.95))
    thr = float(np.quantile(info, q))
    keep = info >= thr
    if not np.any(keep):
        return None

    x_inf = np.sort(xs[keep])
    lo = float(x_inf[0])
    hi = float(x_inf[-1])
    span = max(hi - lo, 1e-12)
    lo -= max(0.0, focus_padding_fraction) * span
    hi += max(0.0, focus_padding_fraction) * span
    lo = float(np.clip(lo, scan.x_min, scan.x_max))
    hi = float(np.clip(hi, scan.x_min, scan.x_max))

    min_width = float(np.clip(focus_min_width_fraction, 0.0, 1.0)) * float(scan.x_max - scan.x_min)
    if hi - lo < min_width:
        mid = 0.5 * (lo + hi)
        half = 0.5 * min_width
        lo = float(np.clip(mid - half, scan.x_min, scan.x_max))
        hi = float(np.clip(mid + half, scan.x_min, scan.x_max))
        if hi - lo < min_width:
            lo = max(float(scan.x_min), hi - min_width)
            hi = min(float(scan.x_max), lo + min_width)

    return (lo, hi) if hi > lo else None


def generate_attempt_plots(
    viz: Viz,
    entry_base: dict[str, Any],
    attempt_idx_in_combo: int,
    current_scan: CoreExperiment,
    current_history_df: pl.DataFrame,
    noise_obj: Any,
    strat_obj: Any,
    slug_base: str,
    out_dir: Path,
    scans_dir: Path,
    bayes_dir: Path,
    run_result: RunResult | None = None,
) -> list[dict[str, Any]]:
    """Generate visualizations and graph manifest entries for a single repeat."""
    attempt_slug = f"{slug_base}_r{attempt_idx_in_combo + 1}"
    out_path = scans_dir / f"{attempt_slug}.html"

    history_with_phase = current_history_df

    # Annotate coarse vs fine phase for strategies that perform an initial sweep.
    initial_sweep_steps = _initial_sweep_steps_from_strategy(strat_obj)
    if "step" in current_history_df.columns and initial_sweep_steps > 0:
        history_with_phase = current_history_df.with_columns(
            pl.when(pl.col("step") < initial_sweep_steps)
            .then(pl.lit("coarse"))
            .otherwise(pl.lit("fine"))
            .alias("phase")
        )

    viz.plot_scan_measurements(
        current_scan,
        history_with_phase,
        out_path,
        over_frequency_noise=noise_obj.over_frequency_noise if noise_obj else None,
        mode_estimates=belief_mode_estimates(run_result.snapshots[-1].belief)
        if run_result and run_result.snapshots
        else None,
        focus_window=_focus_window_from_history(history_with_phase, current_scan, strat_obj),
    )

    scan_entry = entry_base.copy()
    scan_entry["type"] = "scan"
    scan_entry["path"] = out_path.relative_to(out_dir).as_posix()
    scan_entry["plot_data"] = compute_scan_plot_data(
        current_scan,
        history_with_phase,
        noise_obj.over_frequency_noise if noise_obj else None,
        focus_window=_focus_window_from_history(history_with_phase, current_scan, strat_obj),
    )
    entries: list[dict[str, Any]] = [scan_entry]

    strat_name = str(entry_base.get("strategy", ""))
    if run_result is not None and _is_bayesian_run(strat_name, strat_obj):
        scan_param = _resolve_scan_param(strat_obj, run_result)
        try:
            anim_inputs = _posterior_animation_inputs(run_result, scan_param)
            if anim_inputs is not None:
                posterior_history, freq_grid = anim_inputs
                interactive_path = bayes_dir / f"{attempt_slug}_posterior.html"
                viz.plot_posterior_animation(posterior_history, freq_grid, interactive_path)
                if interactive_path.exists():
                    ie = entry_base.copy()
                    ie["type"] = "bayesian_interactive"
                    ie["path"] = interactive_path.relative_to(out_dir).as_posix()
                    entries.append(ie)

            param_hist = [s.belief.estimates() for s in run_result.snapshots]
            if param_hist:
                conv_path = bayes_dir / f"{attempt_slug}_param_convergence.html"
                viz.plot_parameter_convergence(param_hist, conv_path)
                if conv_path.exists():
                    ce = entry_base.copy()
                    ce["type"] = "bayesian_parameter_convergence"
                    ce["path"] = conv_path.relative_to(out_dir).as_posix()
                    entries.append(ce)
        except Exception:
            log.exception(
                "Bayesian auxiliary plots failed for %s repeat %s",
                strat_name,
                attempt_idx_in_combo + 1,
            )

    return entries
