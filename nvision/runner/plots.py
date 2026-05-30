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
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS,
    param_converged,
    param_convergence_bound_width,
)
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.viz import Viz
from nvision.viz.bayesian import _get_nv_parameter_descriptions, _get_signal_formula


log = logging.getLogger(__name__)

# Maximum number of snapshots fed into visualization loops.
# SimpleSobol and other convergence-driven locators can accumulate thousands of
# steps; iterating all of them for Fisher info, covariance ellipses, and
# posterior animations is the dominant post-run cost.  Subsampling to this cap
# keeps plots informative while cutting render time proportionally.
_MAX_VIZ_SNAPSHOTS = 50


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


def _make_focused_ekf_grid(snapshots: list, scan_param: str, default_bounds: tuple[float, float]) -> np.ndarray:
    """Construct a focused grid that covers the mean trajectory and resolves the narrow final peak."""
    means = [float(s.belief.estimates().get(scan_param, 0.0)) for s in snapshots]
    uncs = [float(s.belief.uncertainty().get(scan_param, 1.0)) for s in snapshots]

    final_unc = uncs[-1] if uncs else 1.0
    if final_unc <= 0:
        final_unc = 1e-9

    lo = min(means) - 5 * final_unc
    hi = max(means) + 5 * final_unc

    phys_lo, phys_hi = default_bounds
    lo = max(lo, phys_lo)
    hi = min(hi, phys_hi)

    if hi <= lo:
        lo, hi = phys_lo, phys_hi

    return np.linspace(lo, hi, 250)


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

    from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution
    from nvision.belief.grid_marginal import GridMarginalDistribution
    from nvision.belief.smc_marginal import SMCMarginalDistribution
    from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
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
            weights = b._weights.copy()
            hist.append(np.column_stack([col, weights]))
        # Unused for particle / histogram mode; required by API
        return hist, np.linspace(0.0, 1.0, 2)

    if isinstance(b0, StudentsTMixtureMarginalDistribution):
        out = _extract_mixture_posterior(snapshots, [scan_param])
        if scan_param in out:
            return out[scan_param]

    if isinstance(b0, GaussianMixtureMarginalDistribution):
        out = _extract_gaussian_mixture_posterior(snapshots, [scan_param])
        if scan_param in out:
            return out[scan_param]

    from nvision.sim.locs.ekf.belief import EKFBelief, EKFParticleFrequencyBelief

    if isinstance(b0, EKFParticleFrequencyBelief):
        if scan_param == "frequency":
            hist = []
            for s in snapshots:
                col = s.belief._frequency_particles.copy()
                weights = s.belief._frequency_weights.copy()
                hist.append(np.column_stack([col, weights]))
            return hist, np.linspace(0.0, 1.0, 2)
        else:
            grid = _make_focused_ekf_grid(snapshots, scan_param, b0.physical_param_bounds[scan_param])
            hist = [s.belief.marginal_pdf(scan_param, grid) for s in snapshots]
            return hist, grid

    if isinstance(b0, EKFBelief):
        grid = _make_focused_ekf_grid(snapshots, scan_param, b0.physical_param_bounds[scan_param])
        hist = [s.belief.marginal_pdf(scan_param, grid) for s in snapshots]
        return hist, grid

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

    from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution
    from nvision.belief.grid_marginal import GridMarginalDistribution
    from nvision.belief.smc_marginal import SMCMarginalDistribution
    from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
    from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution
    from nvision.sim.locs.ekf.belief import EKFBelief, EKFParticleFrequencyBelief

    if isinstance(b0, UnitCubeGridMarginalDistribution):
        return _extract_unit_cube_grid_posterior(snapshots, names)

    if isinstance(b0, GridMarginalDistribution):
        return _extract_grid_posterior(snapshots, names)

    if isinstance(b0, SMCMarginalDistribution):
        return _extract_smc_posterior(snapshots, names)

    if isinstance(b0, StudentsTMixtureMarginalDistribution):
        return _extract_mixture_posterior(snapshots, names)

    if isinstance(b0, GaussianMixtureMarginalDistribution):
        return _extract_gaussian_mixture_posterior(snapshots, names)

    if isinstance(b0, EKFBelief):
        return _extract_ekf_posterior(snapshots, names)

    if isinstance(b0, EKFParticleFrequencyBelief):
        return _extract_ekf_particle_frequency_posterior(snapshots, names)

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

    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    b0 = snapshots[0].belief
    stub_grid = np.linspace(0.0, 1.0, 2)
    is_unit_cube = hasattr(b0, "model") and isinstance(b0.model, UnitCubeSignalModel)

    for scan_param in names:
        if scan_param == "noise_sigma" and getattr(b0, "_use_rao_blackwell_noise", False):
            hist: list[np.ndarray] = []
            for s in snapshots:
                b = s.belief
                col = np.sqrt(b._noise_betas / b._noise_alphas)
                weights = b._weights.copy()
                hist.append(np.column_stack([col, weights]))
            out[scan_param] = (hist, stub_grid)
            continue

        idx = b0._param_names.index(scan_param)
        hist: list[np.ndarray] = []

        for s in snapshots:
            b = s.belief
            assert isinstance(b, SMCMarginalDistribution)
            col = b._particles[:, idx].copy()

            lo, hi = 0.0, 1.0
            if hasattr(b, "physical_param_bounds") and scan_param in b.physical_param_bounds:
                lo, hi = b.physical_param_bounds[scan_param]
            elif is_unit_cube and hasattr(b, "model") and hasattr(b.model, "param_bounds_phys"):
                lo, hi = b.model.param_bounds_phys[scan_param]

            if lo != 0.0 or hi != 1.0:
                col = lo + col * (hi - lo)

            weights = b._weights.copy()
            hist.append(np.column_stack([col, weights]))
        out[scan_param] = (hist, stub_grid)
    return out


def _extract_mixture_posterior(snapshots: list, names: list[str]) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    from scipy.stats import t

    from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution

    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    b0 = snapshots[0].belief
    assert isinstance(b0, StudentsTMixtureMarginalDistribution)

    for scan_param in names:
        idx = b0._param_names.index(scan_param)
        lo, hi = b0._physical_param_bounds[scan_param]
        # Generate a grid for PDF evaluation
        grid = np.linspace(lo, hi, 250)

        hist: list[np.ndarray] = []
        for s in snapshots:
            b = s.belief
            assert isinstance(b, StudentsTMixtureMarginalDistribution)
            K = b.n_components  # noqa: N806
            D = b._dim  # noqa: N806

            comp_pdfs = []
            weighted_comp_pdfs = []
            for k in range(K):
                mu = float(b.means[k, idx])
                sigma = float(np.sqrt(max(b._covariances[k, idx, idx], 1e-18)))
                # degrees of freedom for the marginal of a multivariate t is nu - dim + 1
                df = float(max(b.nus[k] - D + 1.0, 1.0))

                # Handle unit-cube conversion if applicable
                if getattr(b, "_is_unit_cube", False):
                    mu = lo + mu * (hi - lo)
                    sigma = sigma * (hi - lo)

                # Raw component PDF (unweighted)
                raw_pdf = t.pdf(grid, df=df, loc=mu, scale=sigma)
                comp_pdfs.append(raw_pdf)
                weighted_comp_pdfs.append(float(b.weights[k]) * raw_pdf)

            # Row-major: [comp0, comp1, ..., compK-1, weighted0, weighted1, ..., weightedK-1, total]
            total_pdf = np.sum(weighted_comp_pdfs, axis=0)
            snapshot_data = np.vstack([comp_pdfs, weighted_comp_pdfs, total_pdf])
            hist.append(snapshot_data)

        out[scan_param] = (hist, grid)
    return out


def _extract_gaussian_mixture_posterior(
    snapshots: list, names: list[str]
) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    from scipy.stats import norm

    from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution

    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    b0 = snapshots[0].belief
    assert isinstance(b0, GaussianMixtureMarginalDistribution)

    for scan_param in names:
        idx = b0._param_names.index(scan_param)
        lo, hi = b0._physical_param_bounds[scan_param]
        # Generate a grid for PDF evaluation
        grid = np.linspace(lo, hi, 250)

        hist: list[np.ndarray] = []
        for s in snapshots:
            b = s.belief
            assert isinstance(b, GaussianMixtureMarginalDistribution)
            K = b.n_components  # noqa: N806

            comp_pdfs = []
            weighted_comp_pdfs = []
            for k in range(K):
                mu = float(b.means[k, idx])
                sigma = float(np.sqrt(max(b._covariances[k, idx, idx], 1e-12)))

                # Handle unit-cube conversion if applicable
                if getattr(b, "_is_unit_cube", False):
                    mu = lo + mu * (hi - lo)
                    sigma = sigma * (hi - lo)

                # Raw component PDF (unweighted)
                raw_pdf = norm.pdf(grid, loc=mu, scale=sigma)
                comp_pdfs.append(raw_pdf)
                weighted_comp_pdfs.append(float(b.weights[k]) * raw_pdf)

            # Row-major: [comp0, comp1, ..., compK-1, weighted0, weighted1, ..., weightedK-1, total]
            total_pdf = np.sum(weighted_comp_pdfs, axis=0)
            snapshot_data = np.vstack([comp_pdfs, weighted_comp_pdfs, total_pdf])
            hist.append(snapshot_data)

        out[scan_param] = (hist, grid)
    return out


def _extract_ekf_posterior(snapshots: list, names: list[str]) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    from nvision.sim.locs.ekf.belief import EKFBelief

    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    b0 = snapshots[0].belief
    assert isinstance(b0, EKFBelief)

    for scan_param in names:
        grid = _make_focused_ekf_grid(snapshots, scan_param, b0.physical_param_bounds[scan_param])

        hist: list[np.ndarray] = []
        for s in snapshots:
            b = s.belief
            assert isinstance(b, EKFBelief)
            pdf_vals = b.marginal_pdf(scan_param, grid)
            hist.append(pdf_vals)

        out[scan_param] = (hist, grid)
    return out


def _extract_ekf_particle_frequency_posterior(
    snapshots: list, names: list[str]
) -> dict[str, tuple[list[np.ndarray], np.ndarray]]:
    from nvision.sim.locs.ekf.belief import EKFParticleFrequencyBelief

    out: dict[str, tuple[list[np.ndarray], np.ndarray]] = {}
    b0 = snapshots[0].belief
    assert isinstance(b0, EKFParticleFrequencyBelief)

    for scan_param in names:
        if scan_param == "frequency":
            hist: list[np.ndarray] = []
            for s in snapshots:
                b = s.belief
                assert isinstance(b, EKFParticleFrequencyBelief)
                col = b._frequency_particles.copy()
                weights = b._frequency_weights.copy()
                hist.append(np.column_stack([col, weights]))
            out[scan_param] = (hist, np.linspace(0.0, 1.0, 2))
        else:
            grid = _make_focused_ekf_grid(snapshots, scan_param, b0.physical_param_bounds[scan_param])
            hist = []
            for s in snapshots:
                b = s.belief
                assert isinstance(b, EKFParticleFrequencyBelief)
                pdf_vals = b.marginal_pdf(scan_param, grid)
                hist.append(pdf_vals)
            out[scan_param] = (hist, grid)
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

    # Filter out initial sweep stages from posterior animation
    # Track which steps involved a resampling event (used in both branches below)
    resampled_steps = [i for i, s in enumerate(bayesian_snapshots) if getattr(s, "resampled", False)]

    anim_all = _posterior_animation_inputs_all_params(viz_run_result, start_idx=sweep_steps)
    log.info("Posterior animation inputs: %s", "available" if anim_all is not None else "None")
    if anim_all is not None:
        # Extract per-step narrowed bounds from snapshots for dynamic UI (only Bayesian stages)
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

        # Track which steps involved a resampling event
        resampled_steps = [i for i, s in enumerate(bayesian_snapshots) if getattr(s, "resampled", False)]

        # Extract particle weight history if it is a particle-based filter
        from nvision.belief.smc_marginal import SMCMarginalDistribution

        weight_history = None
        if bayesian_snapshots and isinstance(bayesian_snapshots[0].belief, SMCMarginalDistribution):
            weight_history = [s.belief._weights.copy() for s in bayesian_snapshots]

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
            resampled_steps=resampled_steps,
            weight_history=weight_history,
        )
    else:
        anim_inputs = _posterior_animation_inputs(viz_run_result, scan_param, start_idx=sweep_steps)
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
                resampled_steps=resampled_steps,
            )
    if interactive_path.exists():
        ie = entry_base.copy()
        ie["type"] = "bayesian_interactive"
        ie["path"] = interactive_path.relative_to(out_dir).as_posix()
        ie["param_count"] = len(anim_all) if anim_all is not None else 1
        ie["resampled_count"] = len(resampled_steps)
        extra.append(ie)

    # Filter out initial sweep stages from parameter convergence plot
    # (bayesian_snapshots is already subsampled from the block above)
    param_hist = [s.belief.uncertainty().as_dict() for s in bayesian_snapshots]
    estimates_hist = [s.belief.estimates() for s in bayesian_snapshots]
    if param_hist:
        conv_path = bayes_dir / f"{attempt_slug}_param_convergence.html"
        viz.plot_parameter_convergence(
            param_hist,
            conv_path,
            estimates_history=estimates_hist,
            true_params=true_params,
        )
        if conv_path.exists():
            ce = entry_base.copy()
            ce["type"] = "bayesian_parameter_convergence"
            ce["path"] = conv_path.relative_to(out_dir).as_posix()
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
            ellipse_path = bayes_dir / f"{attempt_slug}_covariance_ellipses.html"
            viz.plot_covariance_ellipses(
                cov_hist,
                param_names,
                pairs,
                ellipse_path,
                means_history=estimates_hist,
                true_params=true_params,
            )
            if ellipse_path.exists():
                ee = entry_base.copy()
                ee["type"] = "bayesian_covariance_ellipses"
                ee["path"] = ellipse_path.relative_to(out_dir).as_posix()
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
    # Use the subsampled bayesian_snapshots for consistency with other plots.
    viz_snapshots_for_conv = bayesian_snapshots  # already subsampled
    if viz_snapshots_for_conv:
        # Extract convergence-related metrics from each snapshot
        # Note: The actual convergence threshold and patience are locator config,
        # not stored per-snapshot. We use typical defaults for visualization.
        convergence_threshold = NVISION_CONVERGENCE_THRESHOLD
        convergence_patience = 8  # Default patience steps

        conv_metrics = []
        for i, s in enumerate(viz_snapshots_for_conv):
            belief = s.belief
            param_names = list(belief.model.parameter_names())
            if getattr(belief, "_use_rao_blackwell_noise", False):
                if "noise_sigma" not in param_names:
                    param_names.append("noise_sigma")
            uncertainties = belief.uncertainty().as_dict()
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

        conv_path = bayes_dir / f"{attempt_slug}_convergence_metrics.html"
        viz.plot_convergence_metrics(
            conv_metrics,
            param_names,
            convergence_threshold,
            convergence_patience,
            conv_path,
            param_bounds=param_bounds,
        )
        if conv_path.exists():
            ce = entry_base.copy()
            ce["type"] = "bayesian_convergence_metrics"
            ce["path"] = conv_path.relative_to(out_dir).as_posix()
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
    import random
    import math
    from nvision.runner.repeat_keys import repeat_seed_int, measurement_repeat_key
    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
    from nvision.sim.locs.bayesian.sobol_bayesian_locator import SimpleSobolBayesianLocator
    from nvision.runner.convert import belief_mode_estimates
    
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
        max_steps=1500,
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
            sobol_freq_uncert_at_conv = float(locator.belief.uncertainty().get("frequency", math.nan))
            est_f = float(locator.belief.estimates().get("frequency", math.nan))
            sobol_freq_err_at_conv = abs(est_f - true_freq) if not math.isnan(est_f) else math.nan
            
    sobol_mode_estimates = belief_mode_estimates(locator.belief)
    
    new_sobol_data = {
        "sobol_baseline_steps": locator.step_count,
        "sobol_freq_steps": sobol_freq_steps,
        "sobol_freq_uncert_at_conv": sobol_freq_uncert_at_conv,
        "sobol_freq_err_at_conv": sobol_freq_err_at_conv,
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
    """Generate visualizations and graph manifest entries for a single repeat."""
    attempt_slug = f"{slug_base}_r{attempt_idx_in_combo + 1}"
    out_path = scans_dir / f"{attempt_slug}.html"

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

    # Retrieve Sobol baseline measurements & estimates if this strategy is not SimpleSobol itself!
    sobol_xs: list[float] | None = None
    sobol_ys: list[float] | None = None
    sobol_mode_estimates: dict[str, float] | None = None
    if strat_name != "SimpleSobol":
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
        sobol_xs=sobol_xs,
        sobol_ys=sobol_ys,
        sobol_mode_estimates=sobol_mode_estimates,
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

        if run_result is not None and run_result.true_signal:
            scan_entry["true_params"] = {
                "label": "True Signal Parameters",
                "params": run_result.true_signal.parameter_values(),
                "bounds": run_result.true_signal.all_bounds(),
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
