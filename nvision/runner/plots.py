"""Per-repeat plot generation."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nvision.models.experiment import CoreExperiment
from nvision.models.observer import RunResult
from nvision.runner.convert import belief_mode_estimates
from nvision.runner.plots_data import (
    write_convergence_metrics_data,
    write_covariance_data,
    write_fisher_data,
    write_parameter_convergence_data,
    write_posterior_data,
)
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS,
    param_converged,
    param_convergence_bound_width,
)
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.viz import Viz

log = logging.getLogger(__name__)

# Maximum number of snapshots fed into visualization loops.
# SimpleSobol and other convergence-driven locators can accumulate thousands of
# steps; iterating all of them for Fisher info, covariance ellipses, and
# posterior animations is the dominant post-run cost.  Subsampling to this cap
# keeps plots informative while cutting render time proportionally.
_MAX_VIZ_SNAPSHOTS = 1000

# Maximum particles retained per snapshot for posterior visualization.
# Must match write_posterior_data's n_particles default: the UI renders at most
# this many particles per step, so keeping the full population (snapshots x
# params x N particles) in memory only to discard >99% at serialization time
# wastes hundreds of MB on large filters.
_MAX_VIZ_PARTICLES = 60


def _viz_particle_subsample(weights: np.ndarray, max_particles: int = _MAX_VIZ_PARTICLES):
    """Weighted random subsample indices for posterior visualization.

    Returns ``(idx, sub_weights)``; ``idx`` is ``None`` when the population is
    already within the cap. One draw per snapshot is shared across all
    parameters so the UI shows a consistent particle subset per step.
    """
    n = len(weights)
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-30)
    if n <= max_particles:
        return None, w
    idx = np.random.choice(n, size=max_particles, replace=False, p=w)
    sub_w = w[idx]
    sub_w /= sub_w.sum()
    return idx, sub_w


def _subsample_snapshots(snapshots: list, max_frames: int = _MAX_VIZ_SNAPSHOTS) -> list:
    """Return a uniformly-subsampled view of *snapshots* capped at *max_frames*.

    Always keeps the first and last snapshot so the trajectory endpoints are
    preserved.  When ``len(snapshots) <= max_frames`` the original list is
    returned unchanged (no copy).
    """
    n = len(snapshots)
    if n <= max_frames:
        return snapshots
    # Pick indices spread evenly across [0, n-1], always including 0 and n-1.
    indices = sorted(set([0] + [round(i * (n - 1) / (max_frames - 1)) for i in range(1, max_frames - 1)] + [n - 1]))
    return [snapshots[i] for i in indices]


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


def _posterior_animation_inputs(  # noqa: C901
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

        frame_memo: dict[int, np.ndarray] = {}
        for s in snapshots:
            b = s.belief
            assert isinstance(b, SMCMarginalDistribution)
            frame = frame_memo.get(id(b))
            if frame is None:
                sub_idx, sub_w = _viz_particle_subsample(b._weights)
                col = b._particles[sub_idx, idx] if sub_idx is not None else b._particles[:, idx].copy()
                if is_unit_cube:
                    col = lo + col * (hi - lo)
                frame = frame_memo[id(b)] = np.column_stack([col, sub_w])
            hist.append(frame)
        # Unused for particle / histogram mode; required by API
        return hist, np.linspace(0.0, 1.0, 2)

    log.debug("No posterior animation extraction for belief type %s", type(b0).__name__)
    return None


def _posterior_animation_inputs_all_params(  # noqa: C901
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
    if getattr(b0, "_use_rao_blackwell_noise", False):
        if "noise_sigma" not in names:
            names.append("noise_sigma")
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
    from nvision.spectra.unit_cube import UnitCubeSignalModel

    b0 = snapshots[0].belief
    stub_grid = np.linspace(0.0, 1.0, 2)
    is_unit_cube = hasattr(b0, "model") and isinstance(b0.model, UnitCubeSignalModel)
    use_rb = getattr(b0, "_use_rao_blackwell_noise", False)

    # Resolve particle column indices once; physical bounds are resolved
    # per snapshot because the frequency window can narrow during a run.
    param_idx = {
        scan_param: (None if (scan_param == "noise_sigma" and use_rb) else b0._param_names.index(scan_param))
        for scan_param in names
    }

    hists: dict[str, list[np.ndarray]] = {scan_param: [] for scan_param in names}
    # Snapshots from buffered locators share belief objects between batch
    # flushes (observer dedup): extract once per unique belief and reuse the
    # frame arrays for the repeated steps.
    frames_memo: dict[int, dict[str, np.ndarray]] = {}
    for s in snapshots:
        b = s.belief
        assert isinstance(b, SMCMarginalDistribution)
        frames = frames_memo.get(id(b))
        if frames is None:
            # One weighted subsample per snapshot, shared across all parameters,
            # so memory stays O(max_particles) instead of O(num_particles).
            sub_idx, sub_w = _viz_particle_subsample(b._weights)
            frames = {}
            for scan_param in names:
                idx = param_idx[scan_param]
                if idx is None:
                    col = np.sqrt(b._noise_betas / b._noise_alphas)
                    if sub_idx is not None:
                        col = col[sub_idx]
                else:
                    col = b._particles[sub_idx, idx] if sub_idx is not None else b._particles[:, idx].copy()

                    lo, hi = 0.0, 1.0
                    if hasattr(b, "physical_param_bounds") and scan_param in b.physical_param_bounds:
                        lo, hi = b.physical_param_bounds[scan_param]
                    elif is_unit_cube and hasattr(b, "model") and hasattr(b.model, "param_bounds_phys"):
                        lo, hi = b.model.param_bounds_phys[scan_param]

                    if lo != 0.0 or hi != 1.0:
                        col = lo + col * (hi - lo)

                frames[scan_param] = np.column_stack([col, sub_w])
            frames_memo[id(b)] = frames
        for scan_param in names:
            hists[scan_param].append(frames[scan_param])

    return {scan_param: (hists[scan_param], stub_grid) for scan_param in names}


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
    """Posterior animation (all parameters when supported), parameter convergence plot.

    All generated bytes are stored in each entry under ``"_bytes"`` (stripped
    before manifests are written) rather than written to disk.
    """
    extra: list[dict[str, Any]] = []
    scan_param = _resolve_scan_param(strat_obj, run_result)
    true_params = run_result.true_signal.parameter_values()
    if experiment is not None and experiment.noise is not None:
        try:
            true_params["noise_sigma"] = float(experiment.noise.estimated_noise_std())
        except Exception:
            pass
    experiment_domain = (float(experiment.x_min), float(experiment.x_max))
    posterior_path = bayes_dir / f"{attempt_slug}_posterior.json.gz"

    # Subsample snapshots for all visualization loops.  Locators like SimpleSobol
    # can produce thousands of steps; iterating all of them for Fisher info,
    # covariance ellipses, and posterior animations is the dominant post-run cost.
    # We build a subsampled RunResult used only for viz — the original is untouched.
    sweep_steps = run_result.sweep_steps
    all_snapshots = run_result.snapshots
    bayesian_snapshots_full = all_snapshots[sweep_steps:] if sweep_steps > 0 else all_snapshots
    bayesian_snapshots = _subsample_snapshots(bayesian_snapshots_full)
    n_subsampled = len(bayesian_snapshots)
    if n_subsampled < len(bayesian_snapshots_full):
        log.info(
            "Subsampled %d -> %d snapshots for visualization (%s)",
            len(bayesian_snapshots_full),
            n_subsampled,
            attempt_slug,
        )

    # Build a lightweight RunResult with subsampled snapshots for the viz helpers
    # that call _posterior_animation_inputs_all_params / _posterior_animation_inputs.
    from dataclasses import replace as _dc_replace

    viz_run_result = _dc_replace(run_result, snapshots=list(all_snapshots[:sweep_steps]) + bayesian_snapshots)

    resampled_steps = [i for i, s in enumerate(bayesian_snapshots) if getattr(s, "resampled", False)]

    # Calculate parameter uncertainties history.
    # Snapshots from buffered locators (SimpleSobol/SimpleSweep) share belief
    # objects between batch flushes (observer dedup), so memoize the
    # O(particles x params) uncertainty/estimates passes per unique belief
    # instead of per step.
    unc_memo: dict[int, dict[str, float]] = {}
    est_memo: dict[int, dict[str, float]] = {}

    def _unc_of(belief) -> dict[str, float]:
        out = unc_memo.get(id(belief))
        if out is None:
            out = unc_memo[id(belief)] = belief.reported_uncertainty().as_dict()
        return out

    def _est_of(belief) -> dict[str, float]:
        out = est_memo.get(id(belief))
        if out is None:
            out = est_memo[id(belief)] = belief.estimates()
        return out

    param_hist = [_unc_of(s.belief) for s in bayesian_snapshots]
    estimates_hist = [_est_of(s.belief) for s in bayesian_snapshots]

    anim_all = _posterior_animation_inputs_all_params(viz_run_result, start_idx=sweep_steps)
    log.debug("Posterior animation inputs: %s", "available" if anim_all is not None else "None")

    def _rel(path: Path) -> str:
        return path.relative_to(out_dir).as_posix()

    if anim_all is not None:
        physical_bounds = (
            getattr(bayesian_snapshots[0].belief, "physical_param_bounds", {}) if bayesian_snapshots else {}
        )
        ess_threshold = getattr(bayesian_snapshots[0].belief, "ess_threshold", None) if bayesian_snapshots else None
        data = write_posterior_data(
            anim_all,
            true_params=true_params,
            resampled_steps=resampled_steps,
            physical_bounds=physical_bounds,
            ess_threshold=ess_threshold,
            param_hist=param_hist,
            convergence_threshold=NVISION_CONVERGENCE_THRESHOLD,
            absolute_thresholds=PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS,
        )
        if data is not None:
            ie = entry_base.copy()
            ie["type"] = "bayesian_posterior_data"
            ie["path"] = _rel(posterior_path)
            ie["param_count"] = len(anim_all)
            ie["resampled_count"] = len(resampled_steps)
            ie["_bytes"] = data
            extra.append(ie)
    else:
        # Fallback for non-SMC/grid beliefs: single-param posterior animation
        anim_inputs = _posterior_animation_inputs(viz_run_result, scan_param, start_idx=sweep_steps)
        if anim_inputs is not None:
            posterior_history, freq_grid = anim_inputs
            anim_single = {scan_param: (posterior_history, freq_grid)}
            physical_bounds = (
                getattr(bayesian_snapshots[0].belief, "physical_param_bounds", {}) if bayesian_snapshots else {}
            )
            ess_threshold = getattr(bayesian_snapshots[0].belief, "ess_threshold", None) if bayesian_snapshots else None
            data = write_posterior_data(
                anim_single,
                true_params=true_params,
                resampled_steps=resampled_steps,
                physical_bounds=physical_bounds,
                ess_threshold=ess_threshold,
                param_hist=param_hist,
                convergence_threshold=NVISION_CONVERGENCE_THRESHOLD,
                absolute_thresholds=PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS,
            )
            if data is not None:
                ie = entry_base.copy()
                ie["type"] = "bayesian_posterior_data"
                ie["path"] = _rel(posterior_path)
                ie["param_count"] = 1
                ie["resampled_count"] = len(resampled_steps)
                ie["_bytes"] = data
                extra.append(ie)

    if param_hist:
        conv_path = bayes_dir / f"{attempt_slug}_param_convergence.json.gz"
        data = write_parameter_convergence_data(
            param_hist,
            estimates_hist,
            true_params=true_params,
        )
        if data is not None:
            ce = entry_base.copy()
            ce["type"] = "bayesian_parameter_convergence_data"
            ce["path"] = _rel(conv_path)
            ce["_bytes"] = data
            extra.append(ce)

    # Actual SMC covariance history for ellipses and jitter
    from nvision.belief.smc_marginal import SMCMarginalDistribution

    if bayesian_snapshots and isinstance(bayesian_snapshots[0].belief, SMCMarginalDistribution):
        cov_hist = [s.belief.covariance_matrix() for s in bayesian_snapshots]
        param_names = list(bayesian_snapshots[0].belief._param_names)

        # Select pairs for 2D visualization (up to 3 distinct pairs for richer coupling analysis)
        pairs = []
        try:
            priority_pairs = [
                ("frequency", "split"),
                ("frequency", "linewidth"),
                ("split", "linewidth"),
                ("frequency", "dip_depth"),
                ("dip_depth", "linewidth"),
                ("frequency", "c_total"),
                ("c_total", "linewidth"),
            ]
            for p1, p2 in priority_pairs:
                if p1 in param_names and p2 in param_names:
                    idx1 = param_names.index(p1)
                    idx2 = param_names.index(p2)
                    pair = (min(idx1, idx2), max(idx1, idx2))
                    if pair not in pairs:
                        pairs.append(pair)

            if len(pairs) < 3:
                for idx1 in range(len(param_names)):
                    for idx2 in range(idx1 + 1, len(param_names)):
                        pair = (idx1, idx2)
                        if pair not in pairs:
                            pairs.append(pair)
                            if len(pairs) >= 3:
                                break
                    if len(pairs) >= 3:
                        break
        except (ValueError, IndexError):
            pass

        pairs = pairs[:3]
        if not pairs and len(param_names) >= 2:
            pairs.append((0, 1))

        if pairs:
            ellipse_path = bayes_dir / f"{attempt_slug}_covariance_ellipses.json.gz"
            physical_bounds = (
                getattr(bayesian_snapshots[0].belief, "physical_param_bounds", {}) if bayesian_snapshots else {}
            )
            data = write_covariance_data(
                cov_hist,
                param_names,
                pairs,
                estimates_hist,
                true_params=true_params,
                physical_bounds=physical_bounds,
            )
            if data is not None:
                ee = entry_base.copy()
                ee["type"] = "bayesian_covariance_ellipses_data"
                ee["path"] = _rel(ellipse_path)
                ee["_bytes"] = data
                extra.append(ee)

        # Add numerical jitter metrics (standard deviation of estimates over last 20 steps)
        if len(estimates_hist) >= 2:
            subset = estimates_hist[-20:]
            jitter = {k: float(np.std([s.get(k, 0.0) for s in subset])) for k in param_names}
            final_cov = cov_hist[-1]

            je = entry_base.copy()
            je["type"] = "bayesian_jitter"
            je["jitter"] = jitter
            # Include final covariance diagonal (variances)
            je["variances"] = {name: float(final_cov[i, i]) for i, name in enumerate(param_names)}
            # Also include the full correlation matrix for the UI to show couplings
            stds = np.sqrt(np.diag(final_cov) + 1e-20)
            corr = final_cov / np.outer(stds, stds)
            je["correlations"] = {
                name_i: {name_j: float(corr[i, j]) for j, name_j in enumerate(param_names)}
                for i, name_i in enumerate(param_names)
            }
            extra.append(je)

    # Fisher information bounds vs actual uncertainty for SMC beliefs
    from nvision.belief.smc_marginal import SMCMarginalDistribution
    from nvision.models.fisher_information import fisher_information_matrix, single_shot_marginal_stds_from_fim

    if bayesian_snapshots and isinstance(bayesian_snapshots[0].belief, SMCMarginalDistribution):
        param_names = list(bayesian_snapshots[0].belief.model.parameter_names())
        n_params = len(param_names)

        # Compute cumulative Fisher information and bounds at each step.
        # Estimates and uncertainties are reused from the histories computed
        # above instead of re-deriving them from the particle population
        # (each estimates()/uncertainty() call is a full O(N x d) pass).
        fisher_hist = []  # Cumulative FIM at each step
        fisher_bounds_hist = []  # sqrt(diag(inv(FIM))) - theoretical minimum uncertainty
        actual_uncertainty_hist = param_hist  # Actual SMC uncertainty (already computed)

        cum_fim = np.zeros((n_params, n_params))
        for i, s in enumerate(bayesian_snapshots):
            fim_i = fisher_information_matrix(
                x=s.obs.x,
                model=s.belief.model,
                parameters=estimates_hist[i],
                last_obs=s.obs,
            )
            if fim_i is not None:
                cum_fim = cum_fim + fim_i

            fisher_hist.append(cum_fim.copy())
            fisher_bounds_hist.append(single_shot_marginal_stds_from_fim(cum_fim, n_params))

        # Skip Fisher plots if no model supports gradients (cum_fim stayed zero)
        fim_is_degenerate = not np.any(cum_fim != 0)
        if fisher_hist and len(param_names) >= 2 and not fim_is_degenerate:
            fisher_path = bayes_dir / f"{attempt_slug}_fisher.json.gz"
            data = write_fisher_data(
                fisher_bounds_hist,
                actual_uncertainty_hist,
                fisher_hist,
                param_names,
                true_params=true_params,
            )
            if data is not None:
                che = entry_base.copy()
                che["type"] = "bayesian_fisher_data"
                che["path"] = _rel(fisher_path)
                che["_bytes"] = data
                extra.append(che)

    # Convergence metrics visualization for all Bayesian beliefs
    # Use the subsampled bayesian_snapshots for consistency with other plots.
    viz_snapshots_for_conv = bayesian_snapshots  # already subsampled
    if viz_snapshots_for_conv:
        # Extract convergence-related metrics from each snapshot
        # Note: The actual convergence threshold and patience are locator config,
        # not stored per-snapshot. We use typical defaults for visualization.
        convergence_threshold = NVISION_CONVERGENCE_THRESHOLD
        convergence_patience = 8  # Default patience steps

        # Parameter names are identical across snapshots; resolve once.
        first_belief = viz_snapshots_for_conv[0].belief
        param_names = list(first_belief.model.parameter_names())
        if getattr(first_belief, "_use_rao_blackwell_noise", False) and "noise_sigma" not in param_names:
            param_names.append("noise_sigma")

        conv_metrics = []
        for i, s in enumerate(viz_snapshots_for_conv):
            belief = s.belief
            # Reuse the uncertainty history computed once above (param_hist)
            # instead of another O(N x d) particle pass per snapshot.
            uncertainties = param_hist[i]
            bounds = belief.physical_param_bounds

            # Check which parameters are converged
            relative_uncertainties: dict[str, float] = {}
            converged_params: dict[str, bool] = {}
            for name in param_names:
                unc = float(uncertainties.get(name, float("inf")))
                converged_params[name] = param_converged(name, unc, convergence_threshold, bounds)
                if name in PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS:
                    relative_uncertainties[name] = unc
                else:
                    bound_width = param_convergence_bound_width(name, convergence_threshold, bounds)
                    relative_uncertainties[name] = unc / bound_width if bound_width > 0 else float("inf")

            # Compute convergence streak (consecutive steps where all params converged)
            all_converged = all(converged_params.values())

            conv_metrics.append(
                {
                    "step": i,
                    "uncertainties": relative_uncertainties,
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

        # Collect bound ranges from the first snapshot for display
        param_bounds = dict(viz_snapshots_for_conv[0].belief.physical_param_bounds)
        if getattr(viz_snapshots_for_conv[0].belief, "_use_rao_blackwell_noise", False):
            noise_spec = viz_snapshots_for_conv[0].belief.noise_model.spec
            if "noise_sigma" in noise_spec.bounds:
                param_bounds["noise_sigma"] = noise_spec.bounds["noise_sigma"]

        conv_path = bayes_dir / f"{attempt_slug}_convergence_metrics.json.gz"
        data = write_convergence_metrics_data(
            conv_metrics,
            param_names,
            convergence_threshold,
            convergence_patience,
            param_bounds=param_bounds,
            absolute_thresholds=PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS,
        )
        if data is not None:
            ce = entry_base.copy()
            ce["type"] = "bayesian_convergence_metrics_data"
            ce["path"] = _rel(conv_path)
            ce["_bytes"] = data
            extra.append(ce)

    return extra


def get_or_run_sobol_baseline(
    experiment: Any,
    seed: int,
    generator_name: str,
    noise_name: str,
    repeat_idx: int,
) -> dict[str, Any] | None:
    """Retrieve Sobol baseline data from cache, or run the simulation if not cached or missing detailed data."""
    from nvision.runner.sweep_cache import get_cached_sobol_baseline, put_cached_sobol_baseline

    sobol_data = get_cached_sobol_baseline(
        experiment,
        seed,
        generator_name,
        noise_name,
        repeat_idx,
    )

    if sobol_data is not None and "sobol_xs" in sobol_data:
        return sobol_data

    # Otherwise, simulate it dynamically!
    import math
    import random

    from nvision.runner.convert import belief_mode_estimates
    from nvision.runner.repeat_keys import measurement_repeat_key, repeat_seed_int
    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
    from nvision.sim.locs.bayesian.sobol_bayesian_locator import SimpleSobolBayesianLocator

    # 1. Setup locator noise/bounds
    noise_std = 0.05
    noise_max_dev = None
    if experiment.noise is not None:
        noise_std = float(experiment.noise.estimated_noise_std())
        if hasattr(experiment.noise, "estimated_max_noise_deviation"):
            noise_max_dev = float(experiment.noise.estimated_max_noise_deviation(n_samples=6))

    domain_width = float(experiment.x_max - experiment.x_min)
    signal_max_span = None
    model = experiment.true_signal.model
    if hasattr(model, "signal_max_span") and callable(model.signal_max_span):
        signal_max_span = model.signal_max_span(domain_width)

    # Inject bounds (replicating Executor._injected_parameter_bounds(experiment))
    bounds: dict[str, tuple[float, float]] = {}
    for name, bounds_val in experiment.true_signal.bounds.items():
        if name == "_priors":
            bounds[name] = bounds_val
            continue
        if name.startswith("_"):
            continue
        lo_raw, hi_raw = bounds_val
        lo, hi = float(lo_raw), float(hi_raw)
        if hi > lo:
            name_lc = name.lower()
            if ("amplitude" in name_lc or "depth" in name_lc) and hi > noise_std:
                lo = max(lo, noise_std)
                if lo >= hi:
                    lo = float(lo_raw)
        bounds[name] = (lo, hi)

    if experiment.true_signal.noise_bounds:
        bounds.update(experiment.true_signal.noise_bounds)

    belief = nv_center_smc_belief(bounds)

    locator = SimpleSobolBayesianLocator(
        belief=belief,
        max_steps=10000,
        noise_std=noise_std,
        **({} if noise_max_dev is None else {"noise_max_dev": noise_max_dev}),
        **({} if signal_max_span is None else {"signal_max_span": signal_max_span}),
    )

    key = measurement_repeat_key(seed, generator_name, "sobol_baseline", noise_name, repeat_idx)
    sobol_rng = random.Random(repeat_seed_int(key))

    sobol_xs = []
    sobol_ys = []
    sobol_freq_steps = None
    sobol_freq_uncert_at_conv = None
    sobol_freq_err_at_conv = None
    true_freq = experiment.true_signal.get_param_value("frequency")

    while not locator.done():
        x_current = locator.next()
        obs = experiment.measure(x_current, sobol_rng)
        locator.observe(obs)
        sobol_xs.append(float(obs.x))
        sobol_ys.append(float(obs.signal_value))

        # Record metrics at the exact moment of frequency convergence
        if sobol_freq_steps is None and locator.freq_converged_step is not None:
            sobol_freq_steps = locator.freq_converged_step
            sobol_freq_uncert_at_conv = float(locator.belief.reported_uncertainty().get("frequency", math.nan))
            est_f = float(locator.belief.estimates().get("frequency", math.nan))
            sobol_freq_err_at_conv = abs(est_f - true_freq) if not math.isnan(est_f) else math.nan

    sobol_mode_estimates = belief_mode_estimates(locator.belief)

    sobol_final_uncert = float(locator.belief.reported_uncertainty().get("frequency", math.nan))
    est_f_final = float(locator.belief.estimates().get("frequency", math.nan))
    sobol_final_err = abs(est_f_final - true_freq) if not math.isnan(est_f_final) else math.nan

    new_sobol_data = {
        "sobol_baseline_steps": locator.step_count,
        "sobol_freq_steps": sobol_freq_steps,
        "sobol_freq_uncert_at_conv": sobol_freq_uncert_at_conv,
        "sobol_freq_err_at_conv": sobol_freq_err_at_conv,
        "sobol_baseline_uncert": sobol_final_uncert,
        "sobol_baseline_err": sobol_final_err,
        "sobol_xs": sobol_xs,
        "sobol_ys": sobol_ys,
        "sobol_mode_estimates": sobol_mode_estimates,
    }

    put_cached_sobol_baseline(
        experiment,
        seed,
        generator_name,
        noise_name,
        repeat_idx,
        new_sobol_data,
    )

    return new_sobol_data


def get_or_run_simplesweep_baseline(
    experiment: Any,
    seed: int,
    generator_name: str,
    noise_name: str,
    repeat_idx: int,
) -> dict[str, Any] | None:
    """Retrieve SimpleSweep baseline data from cache, or run the simulation if not cached."""
    from nvision.runner.sweep_cache import get_cached_simplesweep_baseline, put_cached_simplesweep_baseline

    data = get_cached_simplesweep_baseline(experiment, seed, generator_name, noise_name, repeat_idx)
    if data is not None and "sweep_xs" in data:
        return data

    import math
    import random

    from nvision.runner.convert import belief_mode_estimates
    from nvision.runner.repeat_keys import measurement_repeat_key, repeat_seed_int
    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
    from nvision.sim.locs.coarse.generic_sweep_locator import GenericSweepLocator

    noise_std = 0.05
    noise_max_dev = None
    if experiment.noise is not None:
        noise_std = float(experiment.noise.estimated_noise_std())
        if hasattr(experiment.noise, "estimated_max_noise_deviation"):
            noise_max_dev = float(experiment.noise.estimated_max_noise_deviation(n_samples=6))

    domain_width = float(experiment.x_max - experiment.x_min)
    signal_max_span = None
    model = experiment.true_signal.model
    if hasattr(model, "signal_max_span") and callable(model.signal_max_span):
        signal_max_span = model.signal_max_span(domain_width)

    bounds: dict[str, tuple[float, float]] = {}
    for name, bounds_val in experiment.true_signal.bounds.items():
        if name == "_priors" or name.startswith("_"):
            bounds[name] = bounds_val
            continue
        lo, hi = float(bounds_val[0]), float(bounds_val[1])
        if hi > lo:
            name_lc = name.lower()
            if ("amplitude" in name_lc or "depth" in name_lc) and hi > noise_std:
                lo = max(lo, noise_std)
                if lo >= hi:
                    lo = float(bounds_val[0])
        bounds[name] = (lo, hi)
    if experiment.true_signal.noise_bounds:
        bounds.update(experiment.true_signal.noise_bounds)

    belief = nv_center_smc_belief(bounds)

    f_lo, f_hi = bounds.get("frequency", (experiment.x_min, experiment.x_max))
    f_domain_width = float(f_hi - f_lo)
    if "linewidth" in bounds:
        min_linewidth = float(bounds["linewidth"][0])
    elif "fwhm_total" in bounds:
        min_linewidth = float(bounds["fwhm_total"][0])
    else:
        min_linewidth = 200e3
    max_steps = max(30, math.ceil(f_domain_width / min_linewidth))

    locator = GenericSweepLocator(
        belief=belief,
        signal_model=model,
        max_steps=max_steps,
        noise_std=noise_std,
        **({} if noise_max_dev is None else {"noise_max_dev": noise_max_dev}),
        **({} if signal_max_span is None else {"signal_max_span": signal_max_span}),
    )

    key = measurement_repeat_key(seed, generator_name, "simplesweep_baseline", noise_name, repeat_idx)
    sweep_rng = random.Random(repeat_seed_int(key))

    sweep_xs: list[float] = []
    sweep_ys: list[float] = []

    while not locator.done():
        x_current = locator.next()
        obs = experiment.measure(x_current, sweep_rng)
        locator.observe(obs)
        sweep_xs.append(float(obs.x))
        sweep_ys.append(float(obs.signal_value))

    # finalize() flushes the deferred belief updates and runs the dip fit;
    # without it the belief is still the prior.
    locator.finalize()
    sweep_mode_estimates = belief_mode_estimates(locator.belief)

    new_data = {
        "sweep_xs": sweep_xs,
        "sweep_ys": sweep_ys,
        "sweep_mode_estimates": sweep_mode_estimates,
    }

    put_cached_simplesweep_baseline(experiment, seed, generator_name, noise_name, repeat_idx, new_data)
    return new_data


def generate_attempt_plots(  # noqa: C901
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
    """Generate visualizations and graph manifest entries for a single repeat.

    All generated bytes are stored in each entry under ``"_bytes"`` (stripped
    before manifests are written).  No files are written to disk.
    """
    attempt_slug = f"{slug_base}_r{attempt_idx_in_combo + 1}"
    out_path = scans_dir / f"{attempt_slug}.json.gz"

    history_with_phase = current_history_df

    # Annotate coarse vs secondary vs fine phase for strategies that perform sweeps.
    # Use actual sweep_steps from run_result if available (captured from locator).
    sweep_steps = run_result.sweep_steps if run_result is not None else 0
    secondary_sweep_steps = run_result.secondary_sweep_steps if run_result is not None else 0

    # Only fall back to strategy defaults if we don't have a reliable count from run_result
    # and it's not explicitly a NoSweep strategy.
    if sweep_steps == 0 and run_result is None:
        strat_name = str(entry_base.get("strategy", ""))
        if "NoSweep" not in strat_name:
            sweep_steps = entry_base.get("sweep_steps") or _initial_sweep_steps_from_strategy(strat_obj)
    if "step" in current_history_df.columns and sweep_steps > 0:
        tertiary_sweep_steps = run_result.tertiary_sweep_steps if run_result is not None else 0
        total_sweep_end = sweep_steps + secondary_sweep_steps
        total_tertiary_end = total_sweep_end + tertiary_sweep_steps
        history_with_phase = current_history_df.with_columns(
            pl.when(pl.col("step") < sweep_steps)
            .then(pl.lit("coarse"))
            .when(pl.col("step") < total_sweep_end)
            .then(pl.lit("secondary"))
            .when(pl.col("step") < total_tertiary_end)
            .then(pl.lit("tertiary"))
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
    if run_result is not None and run_result.snapshots:
        last_belief = run_result.snapshots[-1].belief
        me = belief_mode_estimates(last_belief)
        if me:
            mode_estimates = me
        m = getattr(last_belief, "model", None)
        if isinstance(m, UnitCubeSignalModel):
            belief_unit_cube = m

    # Retrieve Sobol and SimpleSweep baseline measurements & estimates
    sobol_xs: list[float] | None = None
    sobol_ys: list[float] | None = None
    sobol_mode_estimates: dict[str, float] | None = None
    sweep_xs: list[float] | None = None
    sweep_ys: list[float] | None = None
    sweep_mode_estimates: dict[str, float] | None = None
    if strat_name not in ("SimpleSobol", "SimpleSweep"):
        try:
            seed = int(entry_base.get("seed", 0))
            generator_name = str(entry_base.get("generator", ""))
            noise_name = str(entry_base.get("noise", ""))
            sobol_data = get_or_run_sobol_baseline(
                current_scan,
                seed,
                generator_name,
                noise_name,
                attempt_idx_in_combo,
            )
            if sobol_data:
                sobol_xs = sobol_data.get("sobol_xs")
                sobol_ys = sobol_data.get("sobol_ys")
                sobol_mode_estimates = sobol_data.get("sobol_mode_estimates")
        except Exception as exc:
            log.warning("Failed to retrieve or simulate Sobol baseline for plotting: %s", exc)
        try:
            seed = int(entry_base.get("seed", 0))
            generator_name = str(entry_base.get("generator", ""))
            noise_name = str(entry_base.get("noise", ""))
            simplesweep_data = get_or_run_simplesweep_baseline(
                current_scan,
                seed,
                generator_name,
                noise_name,
                attempt_idx_in_combo,
            )
            if simplesweep_data:
                sweep_xs = simplesweep_data.get("sweep_xs")
                sweep_ys = simplesweep_data.get("sweep_ys")
                sweep_mode_estimates = simplesweep_data.get("sweep_mode_estimates")
        except Exception as exc:
            log.warning("Failed to retrieve or simulate SimpleSweep baseline for plotting: %s", exc)

    scan_entry = entry_base.copy()
    scan_entry["type"] = "scan"
    scan_entry["path"] = out_path.relative_to(out_dir).as_posix()

    # Per-step (error, uncertainty) series for the UI Highlights view.
    # Attached to the scan entry only — entry_base is also copied into the
    # Bayesian auxiliary entries and must stay slim.
    if run_result is not None:
        try:
            from nvision.metrics.series import _round_sig, extract_step_series

            step_series = extract_step_series(run_result)
            if step_series:
                # The finalize fit (e.g. a sweep's dip fit) can land outside the
                # belief snapshots; the series endpoint must match the reported
                # final metrics so the anytime curve ends at the true accuracy.
                final_err = entry_base.get("abs_err_x")
                final_unc = entry_base.get("uncert")
                if isinstance(final_err, int | float) and math.isfinite(final_err):
                    step_series["e"][-1] = _round_sig(float(final_err))
                if isinstance(final_unc, int | float) and math.isfinite(final_unc):
                    step_series["u"][-1] = _round_sig(float(final_unc))
                scan_entry["series"] = step_series
        except Exception:
            log.debug("Step series extraction failed for %s", attempt_slug, exc_info=True)

    # Build true_params for all strategies so it can be embedded in figure meta
    # (the manifest strips it; UI reads it from fig.layout.meta.true_params)
    _true_params_dict: dict | None = None
    if run_result is not None and run_result.true_signal:
        _true_params_dict = {
            "label": "True Signal Parameters",
            "params": run_result.true_signal.parameter_values(),
            "bounds": run_result.true_signal.all_bounds(),
        }

    scan_entry["_bytes"] = viz.plot_scan_measurements(
        current_scan,
        history_with_phase,
        over_frequency_noise=noise_obj.over_frequency_noise if noise_obj else None,
        mode_estimates=mode_estimates,
        focus_window=focus_window,
        per_dip_windows=per_dip_windows,
        belief_unit_cube=belief_unit_cube,
        narrowed_param_bounds=run_result.narrowed_param_bounds if run_result is not None else None,
        sobol_xs=sobol_xs,
        sobol_ys=sobol_ys,
        sobol_mode_estimates=sobol_mode_estimates,
        sweep_xs=sweep_xs,
        sweep_ys=sweep_ys,
        sweep_mode_estimates=sweep_mode_estimates,
        true_params=_true_params_dict,
    )
    # plot_data is loaded on-demand by UI from scan JSON to keep manifest small

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
                "last_run": scan_entry.get("last_run"),
            }
            scan_entry["fine"] = {
                "label": "Bayesian inference",
                "measurements": locator_steps,
                "sweep_steps": 0,
                "locator_steps": locator_steps,
                "abs_err_x": scan_entry.get("abs_err_x"),
                "uncert": scan_entry.get("uncert"),
                "duration_ms": scan_entry.get("duration_ms"),
                "last_run": scan_entry.get("last_run"),
                "steps_to_fb": scan_entry.get("steps_to_fb"),
                "sobol_freq_steps": scan_entry.get("sobol_freq_steps"),
                "sobol_baseline_steps": scan_entry.get("sobol_baseline_steps"),
                "sobol_freq_uncert_at_conv": scan_entry.get("sobol_freq_uncert_at_conv"),
                "sobol_freq_err_at_conv": scan_entry.get("sobol_freq_err_at_conv"),
                "uncert_fb_at_milestone": scan_entry.get("uncert_fb_at_milestone"),
                "err_fb_at_milestone": scan_entry.get("err_fb_at_milestone"),
                "err_fc_at_milestone": scan_entry.get("err_fc_at_milestone"),
                "err_fc_diff": scan_entry.get("err_fc_diff"),
            }

        # true_params is now embedded in fig.layout.meta; keep here for backward compat
        if _true_params_dict is not None:
            scan_entry["true_params"] = _true_params_dict

    # Also set true_params for non-Bayesian entries (not inside the Bayesian block above)
    if _true_params_dict is not None and "true_params" not in scan_entry:
        scan_entry["true_params"] = _true_params_dict

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
