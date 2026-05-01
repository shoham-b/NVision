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
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.viz import Viz
from nvision.viz.bayesian import _get_nv_parameter_descriptions, _get_signal_formula

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
    start_idx: int = 0,
) -> tuple[list[np.ndarray], np.ndarray] | None:
    """Build (posterior_history, freq_grid) for ``plot_posterior_animation``.

    Parameters
    ----------
    run_result : RunResult
        Full result with snapshots
    scan_param : str
        Parameter to extract posterior for
    start_idx : int
        Starting index to slice snapshots (used to exclude initial sweep stages)
    """
    if not run_result.snapshots:
        return None

    snapshots = run_result.snapshots[start_idx:] if start_idx > 0 else run_result.snapshots
    if not snapshots:
        return None

    from nvision.belief.grid_marginal import GridMarginalDistribution
    from nvision.belief.smc_marginal import SMCMarginalDistribution
    from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution

    b0 = snapshots[0].belief
    if isinstance(b0, UnitCubeGridMarginalDistribution):
        grid = b0.physical_param_grid(scan_param)
        # Use base get_grid_param to access unit-cube PMF directly.
        hist = [GridMarginalDistribution.get_grid_param(s.belief, scan_param).posterior.copy() for s in snapshots]
        return hist, grid
    if isinstance(b0, GridMarginalDistribution):
        grid = b0.get_grid_param(scan_param).grid
        hist = [s.belief.get_grid_param(scan_param).posterior.copy() for s in snapshots]
        return hist, grid

    if isinstance(b0, SMCMarginalDistribution):
        idx = b0._param_names.index(scan_param)
        hist: list[np.ndarray] = []

        is_unit_cube = False
        lo, hi = 0.0, 1.0
        if hasattr(b0, "model") and isinstance(b0.model, UnitCubeSignalModel):
            is_unit_cube = True
            lo, hi = b0.model.param_bounds_phys[scan_param]

        for s in snapshots:
            b = s.belief
            assert isinstance(b, SMCMarginalDistribution)
            col = b._particles[:, idx].copy()
            if is_unit_cube:
                col = lo + col * (hi - lo)
            hist.append(col.reshape(-1, 1))
        # Unused for particle / histogram mode; required by API
        return hist, np.linspace(0.0, 1.0, 2)

    log.debug("No posterior animation extraction for belief type %s", type(b0).__name__)
    return None


def _posterior_animation_inputs_all_params(
    run_result: RunResult,
    start_idx: int = 0,
) -> dict[str, tuple[list[np.ndarray], np.ndarray]] | None:
    """Marginal posterior history + axis grid for every model parameter (for faceted animation).

    Parameters
    ----------
    run_result : RunResult
        Full result with snapshots
    start_idx : int
        Starting index to slice snapshots (used to exclude initial sweep stages)
    """
    if not run_result.snapshots:
        return None

    snapshots = run_result.snapshots[start_idx:] if start_idx > 0 else run_result.snapshots
    if not snapshots:
        return None

    b0 = snapshots[0].belief
    names = list(b0.model.parameter_names())
    if not names:
        return None

    from nvision.belief.grid_marginal import GridMarginalDistribution
    from nvision.belief.smc_marginal import SMCMarginalDistribution
    from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution

    if isinstance(b0, UnitCubeGridMarginalDistribution):
        return _extract_unit_cube_grid_posterior(snapshots, names)

    if isinstance(b0, GridMarginalDistribution):
        return _extract_grid_posterior(snapshots, names)

    if isinstance(b0, SMCMarginalDistribution):
        return _extract_smc_posterior(snapshots, names)

    log.debug("No multi-parameter posterior extraction for belief type %s", type(b0).__name__)
    return None


def _extract_unit_cube_grid_posterior(
    snapshots: list, names: list[str]
) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    from nvision.belief.grid_marginal import GridMarginalDistribution

    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    b0 = snapshots[0].belief
    for scan_param in names:
        grid = b0.physical_param_grid(scan_param)
        hist = [GridMarginalDistribution.get_grid_param(s.belief, scan_param).posterior.copy() for s in snapshots]
        out[scan_param] = (hist, grid)
    return out


def _extract_grid_posterior(snapshots: list, names: list[str]) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    b0 = snapshots[0].belief
    for scan_param in names:
        grid = b0.get_grid_param(scan_param).grid
        hist = [s.belief.get_grid_param(scan_param).posterior.copy() for s in snapshots]
        out[scan_param] = (hist, grid)
    return out


def _extract_smc_posterior(snapshots: list, names: list[str]) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    from nvision.belief.smc_marginal import SMCMarginalDistribution

    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    b0 = snapshots[0].belief
    stub_grid = np.linspace(0.0, 1.0, 2)
    is_unit_cube = hasattr(b0, "model") and isinstance(b0.model, UnitCubeSignalModel)

    for scan_param in names:
        idx = b0._param_names.index(scan_param)
        hist: list[np.ndarray] = []
        lo, hi = 0.0, 1.0
        if is_unit_cube:
            lo, hi = b0.model.param_bounds_phys[scan_param]

        for s in snapshots:
            b = s.belief
            assert isinstance(b, SMCMarginalDistribution)
            col = b._particles[:, idx].copy()
            if is_unit_cube:
                col = lo + col * (hi - lo)
            hist.append(col.reshape(-1, 1))
        out[scan_param] = (hist, stub_grid)
    return out


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

    return 0


def _bayesian_auxiliary_entries(  # noqa: C901
    viz: Viz,
    entry_base: dict[str, Any],
    run_result: RunResult,
    strat_obj: Any,
    attempt_slug: str,
    bayes_dir: Path,
    out_dir: Path,
    experiment: CoreExperiment,
) -> list[dict[str, Any]]:
    """Posterior animation (all parameters when supported), parameter convergence plot."""
    extra: list[dict[str, Any]] = []
    scan_param = _resolve_scan_param(strat_obj, run_result)
    true_params = run_result.true_signal.parameter_values()
    experiment_domain = (float(experiment.x_min), float(experiment.x_max))
    interactive_path = bayes_dir / f"{attempt_slug}_posterior.html"
    # Filter out initial sweep stages from posterior animation
    sweep_steps = run_result.sweep_steps
    anim_all = _posterior_animation_inputs_all_params(run_result, start_idx=sweep_steps)
    log.info("Posterior animation inputs: %s", "available" if anim_all is not None else "None")
    if anim_all is not None:
        # Extract per-step narrowed bounds from snapshots for dynamic UI (only Bayesian stages)
        bayesian_snapshots = run_result.snapshots[sweep_steps:] if sweep_steps > 0 else run_result.snapshots
        per_step_narrowed_bounds = []
        for snapshot in bayesian_snapshots:
            if snapshot.narrowed_param_bounds:
                per_step_narrowed_bounds.append(snapshot.narrowed_param_bounds)
            else:
                per_step_narrowed_bounds.append({})
        # Generate parameter descriptions and signal formula from the signal model
        signal_model = run_result.true_signal.model
        param_descriptions = _get_nv_parameter_descriptions(signal_model)
        signal_formula = _get_signal_formula(signal_model)
        viz.plot_posterior_animation_all_params(
            anim_all,
            interactive_path,
            true_params=true_params,
            acquisition_window=run_result.focus_window,
            acquisition_param=scan_param,
            experiment_domain=experiment_domain,
            narrowed_param_bounds=run_result.narrowed_param_bounds,
            per_step_narrowed_bounds=per_step_narrowed_bounds,
            param_descriptions=param_descriptions,
            signal_formula=signal_formula,
        )
    else:
        anim_inputs = _posterior_animation_inputs(run_result, scan_param, start_idx=sweep_steps)
        if anim_inputs is not None:
            posterior_history, freq_grid = anim_inputs
            true_one = true_params.get(scan_param)
            viz.plot_posterior_animation(
                posterior_history,
                freq_grid,
                interactive_path,
                true_value=float(true_one) if true_one is not None else None,
                acquisition_window=run_result.focus_window,
                experiment_domain=experiment_domain,
            )
    if interactive_path.exists():
        ie = entry_base.copy()
        ie["type"] = "bayesian_interactive"
        ie["path"] = interactive_path.relative_to(out_dir).as_posix()
        ie["param_count"] = len(anim_all) if anim_all is not None else 1
        extra.append(ie)

    # Filter out initial sweep stages from parameter convergence plot
    sweep_steps = run_result.sweep_steps
    bayesian_snapshots = run_result.snapshots[sweep_steps:] if sweep_steps > 0 else run_result.snapshots
    param_hist = [s.belief.uncertainty().as_dict() for s in bayesian_snapshots]
    if param_hist:
        conv_path = bayes_dir / f"{attempt_slug}_param_convergence.html"
        viz.plot_parameter_convergence(param_hist, conv_path)
        if conv_path.exists():
            ce = entry_base.copy()
            ce["type"] = "bayesian_parameter_convergence"
            ce["path"] = conv_path.relative_to(out_dir).as_posix()
            extra.append(ce)

    # Fisher information bounds vs actual uncertainty for SMC beliefs
    from nvision.belief.smc_marginal import SMCMarginalDistribution
    from nvision.models.fisher_information import fisher_information_matrix, single_shot_marginal_stds_from_fim

    if bayesian_snapshots and isinstance(bayesian_snapshots[0].belief, SMCMarginalDistribution):
        param_names = list(bayesian_snapshots[0].belief.model.parameter_names())
        n_params = len(param_names)

        # Compute cumulative Fisher information and bounds at each step
        fisher_hist = []  # Cumulative FIM at each step
        fisher_bounds_hist = []  # sqrt(diag(inv(FIM))) - theoretical minimum uncertainty
        actual_uncertainty_hist = []  # Actual SMC uncertainty

        cum_fim = np.zeros((n_params, n_params))
        for s in bayesian_snapshots:
            # Get observation point and compute Fisher contribution
            x_obs = s.obs.x
            model = s.belief.model
            params = s.belief.estimates()

            fim_i = fisher_information_matrix(
                x=x_obs,
                model=model,
                parameters=params,
                last_obs=s.obs,
            )
            if fim_i is not None:
                cum_fim = cum_fim + fim_i

            fisher_hist.append(cum_fim.copy())
            fisher_bounds_hist.append(single_shot_marginal_stds_from_fim(cum_fim, n_params))
            actual_uncertainty_hist.append(s.belief.uncertainty().as_dict())

        # Skip Fisher plots if no model supports gradients (cum_fim stayed zero)
        fim_is_degenerate = not np.any(cum_fim != 0)
        if fisher_hist and len(param_names) >= 2 and not fim_is_degenerate:
            # Fisher bounds vs actual uncertainty plot (marginals)
            fisher_path = bayes_dir / f"{attempt_slug}_fisher_bounds.html"
            viz.plot_fisher_vs_crlb(
                fisher_bounds_hist,
                actual_uncertainty_hist,
                param_names,
                fisher_path,
                true_params=true_params,
            )
            if fisher_path.exists():
                che = entry_base.copy()
                che["type"] = "bayesian_fisher_bounds"
                che["path"] = fisher_path.relative_to(out_dir).as_posix()
                extra.append(che)

            # Fisher CRLB confidence ellipses for parameter pairs (correlations)
            fisher_pairs_path = bayes_dir / f"{attempt_slug}_fisher_crlb_pairs.html"
            viz.plot_fisher_crlb_pairs(
                fisher_hist,
                param_names,
                fisher_pairs_path,
                true_params=true_params,
            )
            if fisher_pairs_path.exists():
                che = entry_base.copy()
                che["type"] = "bayesian_fisher_crlb_pairs"
                che["path"] = fisher_pairs_path.relative_to(out_dir).as_posix()
                extra.append(che)

    # Convergence metrics visualization for all Bayesian beliefs
    if run_result.snapshots:
        # Extract convergence-related metrics from each snapshot
        # Note: The actual convergence threshold and patience are locator config,
        # not stored per-snapshot. We use typical defaults for visualization.
        convergence_threshold = 0.01  # Default from SequentialBayesianLocator
        convergence_patience = 8  # Default patience steps

        conv_metrics = []
        for i, s in enumerate(run_result.snapshots):
            belief = s.belief
            param_names = list(belief.model.parameter_names())
            uncertainties = belief.uncertainty().as_dict()

            # Check which parameters are converged (uncertainty < threshold)
            converged_params = {
                name: float(uncertainties.get(name, float("inf"))) < convergence_threshold for name in param_names
            }

            # Compute convergence streak (consecutive steps where all params converged)
            all_converged = all(converged_params.values())

            conv_metrics.append(
                {
                    "step": i,
                    "uncertainties": uncertainties,
                    "converged_params": converged_params,
                    "all_converged": all_converged,
                }
            )

        # Compute convergence streak
        streak = 0
        for cm in conv_metrics:
            if cm["all_converged"]:
                streak += 1
            else:
                streak = 0
            cm["convergence_streak"] = streak
            cm["convergence_achieved"] = streak >= convergence_patience

        conv_path = bayes_dir / f"{attempt_slug}_convergence_metrics.html"
        viz.plot_convergence_metrics(
            conv_metrics,
            param_names,
            convergence_threshold,
            convergence_patience,
            conv_path,
        )
        if conv_path.exists():
            ce = entry_base.copy()
            ce["type"] = "bayesian_convergence_metrics"
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

    # Annotate coarse vs secondary vs fine phase for strategies that perform sweeps.
    # Use actual sweep_steps from run_result if available (captured from locator).
    sweep_steps = run_result.sweep_steps if run_result is not None else 0
    secondary_sweep_steps = run_result.secondary_sweep_steps if run_result is not None else 0
    if sweep_steps == 0:
        sweep_steps = entry_base.get("sweep_steps") or _initial_sweep_steps_from_strategy(strat_obj)
    # DEBUG: Log phase assignment values
    import logging

    log = logging.getLogger("nvision")
    log.info(
        f"[PHASE DEBUG] sweep_steps={sweep_steps}, secondary_sweep_steps={secondary_sweep_steps}, "
        f"history steps: min={current_history_df['step'].min() if 'step' in current_history_df.columns else 'N/A'}, "
        f"max={current_history_df['step'].max() if 'step' in current_history_df.columns else 'N/A'}, "
        f"height={current_history_df.height}"
    )
    if "step" in current_history_df.columns and sweep_steps > 0:
        total_sweep_end = sweep_steps + secondary_sweep_steps
        log.info(
            f"[PHASE DEBUG] total_sweep_end={total_sweep_end}, "
            f"coarse: step < {sweep_steps}, secondary: step < {total_sweep_end}"
        )
        history_with_phase = current_history_df.with_columns(
            pl.when(pl.col("step") < sweep_steps)
            .then(pl.lit("coarse"))
            .when(pl.col("step") < total_sweep_end)
            .then(pl.lit("secondary"))
            .otherwise(pl.lit("fine"))
            .alias("phase")
        )

    focus_window = run_result.focus_window if run_result is not None else None
    # Fallback to narrowed_param_bounds only when they are genuinely tighter than
    # the full domain.  Prefer a frequency-like scan parameter, otherwise skip.
    if focus_window is None and run_result is not None and run_result.narrowed_param_bounds:
        nb = run_result.narrowed_param_bounds
        scan_param_name = None
        for name in nb:
            if "freq" in name.lower() or name in ("x", "frequency"):
                scan_param_name = name
                break
        if scan_param_name is None:
            scan_param_name = next(iter(nb))
        lo, hi = nb[scan_param_name]
        domain_width = current_scan.x_max - current_scan.x_min
        if hi - lo < domain_width * (1.0 - 1e-9):
            focus_window = (lo, hi)
    per_dip_windows = run_result.per_dip_windows if run_result is not None else None

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
        per_dip_windows=per_dip_windows,
        belief_unit_cube=belief_unit_cube,
        narrowed_param_bounds=run_result.narrowed_param_bounds if run_result is not None else None,
    )

    scan_entry = entry_base.copy()
    scan_entry["type"] = "scan"
    scan_entry["path"] = out_path.relative_to(out_dir).as_posix()
    # plot_data is loaded on-demand by UI from scan HTML to keep manifest small

    # Add per-phase breakdown for Bayesian runs with a preliminary sweep
    if _is_bayesian_run(strat_name, strat_obj):
        sweep_steps = scan_entry.get("sweep_steps") or 0
        locator_steps = scan_entry.get("locator_steps") or 0
        if sweep_steps and locator_steps:
            scan_entry["coarse"] = {
                "label": "Preliminary (Sobol)",
                "measurements": sweep_steps,
                "sweep_steps": sweep_steps,
                "locator_steps": 0,
            }
            scan_entry["fine"] = {
                "label": "Bayesian inference",
                "measurements": locator_steps,
                "sweep_steps": 0,
                "locator_steps": locator_steps,
                "abs_err_x": scan_entry.get("abs_err_x"),
                "uncert": scan_entry.get("uncert"),
                "duration_ms": scan_entry.get("duration_ms"),
            }

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
                    current_scan,
                )
            )
        except Exception:
            log.exception(
                "Bayesian auxiliary plots failed for %s repeat %s",
                strat_name,
                attempt_idx_in_combo + 1,
            )

    return entries
