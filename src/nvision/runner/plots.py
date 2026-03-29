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
from nvision.signal.unit_cube import UnitCubeSignalModel
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


def _posterior_animation_inputs(
    run_result: RunResult,
    scan_param: str,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """Build (posterior_history, freq_grid) for ``plot_posterior_animation``."""
    if not run_result.snapshots:
        return None

    from nvision.belief.grid_belief import GridBeliefDistribution
    from nvision.belief.smc_belief import SMCBeliefDistribution
    from nvision.belief.unit_cube_grid_belief import UnitCubeGridBeliefDistribution

    b0 = run_result.snapshots[0].belief
    if isinstance(b0, UnitCubeGridBeliefDistribution):
        grid = b0.physical_param_grid(scan_param)
        # UnitCube overrides get_param() to physical Parameter metadata; use grid PMF via base.
        hist = [GridBeliefDistribution.get_param(s.belief, scan_param).posterior.copy() for s in run_result.snapshots]
        return hist, grid
    if isinstance(b0, GridBeliefDistribution):
        grid = b0.get_param(scan_param).grid
        hist = [s.belief.get_param(scan_param).posterior.copy() for s in run_result.snapshots]
        return hist, grid

    if isinstance(b0, SMCBeliefDistribution):
        idx = b0._param_names.index(scan_param)
        hist: list[np.ndarray] = []
        for s in run_result.snapshots:
            b = s.belief
            assert isinstance(b, SMCBeliefDistribution)
            col = b._particles[:, idx]
            hist.append(col.reshape(-1, 1))
        # Unused for particle / histogram mode; required by API
        return hist, np.linspace(0.0, 1.0, 2)

    log.debug("No posterior animation extraction for belief type %s", type(b0).__name__)
    return None


def _posterior_animation_inputs_all_params(
    run_result: RunResult,
) -> dict[str, tuple[list[np.ndarray], np.ndarray]] | None:
    """Marginal posterior history + axis grid for every model parameter (for faceted animation)."""
    if not run_result.snapshots:
        return None

    from nvision.belief.grid_belief import GridBeliefDistribution
    from nvision.belief.smc_belief import SMCBeliefDistribution
    from nvision.belief.unit_cube_grid_belief import UnitCubeGridBeliefDistribution

    b0 = run_result.snapshots[0].belief
    names = list(b0.model.parameter_names())
    if not names:
        return None

    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}

    if isinstance(b0, UnitCubeGridBeliefDistribution):
        for scan_param in names:
            grid = b0.physical_param_grid(scan_param)
            hist = [
                GridBeliefDistribution.get_param(s.belief, scan_param).posterior.copy() for s in run_result.snapshots
            ]
            out[scan_param] = (hist, grid)
        return out

    if isinstance(b0, GridBeliefDistribution):
        for scan_param in names:
            grid = b0.get_param(scan_param).grid
            hist = [s.belief.get_param(scan_param).posterior.copy() for s in run_result.snapshots]
            out[scan_param] = (hist, grid)
        return out

    if isinstance(b0, SMCBeliefDistribution):
        stub_grid = np.linspace(0.0, 1.0, 2)
        for scan_param in names:
            idx = b0._param_names.index(scan_param)
            hist: list[np.ndarray] = []
            for s in run_result.snapshots:
                b = s.belief
                assert isinstance(b, SMCBeliefDistribution)
                col = b._particles[:, idx]
                hist.append(col.reshape(-1, 1))
            out[scan_param] = (hist, stub_grid)
        return out

    log.debug("No multi-parameter posterior extraction for belief type %s", type(b0).__name__)
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


def _bayesian_auxiliary_entries(
    viz: Viz,
    entry_base: dict[str, Any],
    run_result: RunResult,
    strat_obj: Any,
    attempt_slug: str,
    bayes_dir: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Posterior animation (all parameters when supported), parameter convergence plot."""
    extra: list[dict[str, Any]] = []
    scan_param = _resolve_scan_param(strat_obj, run_result)
    true_params = run_result.true_signal.parameter_values()
    interactive_path = bayes_dir / f"{attempt_slug}_posterior.html"
    anim_all = _posterior_animation_inputs_all_params(run_result)
    if anim_all is not None:
        viz.plot_posterior_animation_all_params(anim_all, interactive_path, true_params=true_params)
    else:
        anim_inputs = _posterior_animation_inputs(run_result, scan_param)
        if anim_inputs is not None:
            posterior_history, freq_grid = anim_inputs
            true_one = true_params.get(scan_param)
            viz.plot_posterior_animation(
                posterior_history,
                freq_grid,
                interactive_path,
                true_value=float(true_one) if true_one is not None else None,
            )
    if interactive_path.exists():
        ie = entry_base.copy()
        ie["type"] = "bayesian_interactive"
        ie["path"] = interactive_path.relative_to(out_dir).as_posix()
        extra.append(ie)

    param_hist = [s.belief.uncertainty().as_dict() for s in run_result.snapshots]
    if param_hist:
        conv_path = bayes_dir / f"{attempt_slug}_param_convergence.html"
        viz.plot_parameter_convergence(param_hist, conv_path)
        if conv_path.exists():
            ce = entry_base.copy()
            ce["type"] = "bayesian_parameter_convergence"
            ce["path"] = conv_path.relative_to(out_dir).as_posix()
            extra.append(ce)
    return extra


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

    focus_window = run_result.focus_window if run_result is not None else None

    strat_name = str(entry_base.get("strategy", ""))
    mode_estimates: dict[str, float] | None = None
    belief_unit_cube: UnitCubeSignalModel | None = None
    if run_result is not None and _is_bayesian_run(strat_name, strat_obj) and run_result.snapshots:
        last_belief = run_result.snapshots[-1].belief
        me = belief_mode_estimates(last_belief)
        if me:
            mode_estimates = me
        m = getattr(last_belief, "model", None)
        if isinstance(m, UnitCubeSignalModel):
            belief_unit_cube = m

    viz.plot_scan_measurements(
        current_scan,
        history_with_phase,
        out_path,
        over_frequency_noise=noise_obj.over_frequency_noise if noise_obj else None,
        mode_estimates=mode_estimates,
        focus_window=focus_window,
        belief_unit_cube=belief_unit_cube,
    )

    scan_entry = entry_base.copy()
    scan_entry["type"] = "scan"
    scan_entry["path"] = out_path.relative_to(out_dir).as_posix()
    scan_entry["plot_data"] = compute_scan_plot_data(
        current_scan,
        history_with_phase,
        noise_obj.over_frequency_noise if noise_obj else None,
        focus_window=focus_window,
        mode_estimates=mode_estimates,
        belief_unit_cube=belief_unit_cube,
    )
    entries: list[dict[str, Any]] = [scan_entry]

    if run_result is not None and _is_bayesian_run(strat_name, strat_obj):
        try:
            entries.extend(
                _bayesian_auxiliary_entries(
                    viz,
                    entry_base,
                    run_result,
                    strat_obj,
                    attempt_slug,
                    bayes_dir,
                    out_dir,
                )
            )
        except Exception:
            log.exception(
                "Bayesian auxiliary plots failed for %s repeat %s",
                strat_name,
                attempt_idx_in_combo + 1,
            )

    return entries
