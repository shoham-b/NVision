"""Abstract base class for all Bayesian locators."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from dotenv import load_dotenv

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    param_convergence_bound_width,
)

_POSTERIOR_NARROWING_INTERVAL: int = 20
_POSTERIOR_CREDIBLE_LEVEL: float = 0.95
_POSTERIOR_MIN_NARROWING_FRACTION: float = 0.05
_CONVERGENCE_CHECK_INTERVAL: int = 100

load_dotenv()

# Number of Bayesian acquisitions to buffer before the first belief update.
# Sequential w *= likelihood calls during warmup compound multiplicatively,
# drifting weights far toward one mode before resampling can correct them.
# Buffering and flushing as one batch_update (log-space, epistemically tempered)
# gives the filter a gentler, more robust initial update.
_WARMUP_BUFFER_SIZE: int = int(os.getenv("NVISION_BAYESIAN_WARMUP_STEPS", "5"))


class SequentialBayesianLocator(Locator):
    """Shared Bayesian loop infrastructure for all acquisition strategies.

    Handles the mechanics common to every Bayesian locator:
    - incrementing the step counter
    - convergence-based stopping
    - extracting results from the belief posterior

    Subclasses must implement:
    - ``create(**config)`` — build the model-specific ``BeliefSignal`` with priors.
    - ``_acquire()``       — select the next measurement position from the current belief.

    The only behavioral difference between Bayesian strategies is *how* the
    next measurement position is chosen.  All other wiring is identical and
    lives here so it never needs to be repeated.

    Parameters
    ----------
    belief : AbstractMarginalDistribution
        Initial belief (usually a flat / uniform prior over all parameters).
    max_steps : int
        Hard upper bound on Bayesian inference steps.
    convergence_threshold : float
        Relative uncertainty threshold (fraction of bound width) below which
        we consider parameters converged and stop early.  Default ``0.01`` = 1 %.
    scan_param : str | None
        The parameter we are proposing measurements along. Defaults to the
        first parameter in the belief.
    """

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int = 450,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        scan_param: str | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        initial_sweep_steps: int | None = None,
    ) -> None:
        super().__init__(belief)
        self.initial_sweep_steps: int = 0
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.convergence_threshold = convergence_threshold
        # Total measurement count
        self.step_count: int = 0
        # Bayesian acquisition count
        self.inference_step_count: int = 0
        self._scan_param = scan_param or belief.model.parameter_names()[0]
        # Default convergence target is all model parameters.
        self._convergence_params: tuple[str, ...] = (
            tuple(convergence_params) if convergence_params is not None else tuple(self.belief.model.parameter_names())
        )
        self._convergence_patience_steps = max(1, int(convergence_patience_steps))
        self._convergence_streak = 0

        if noise_std is None or float(noise_std) <= 0:
            raise ValueError(f"noise_std must be a positive float; got {noise_std!r}")
        self._noise_std: float = float(noise_std)
        self._noise_max_dev: float | None = (
            float(noise_max_dev) if (noise_max_dev is not None and noise_max_dev > 0) else None
        )
        self._signal_max_span: float | None = (
            float(signal_max_span) if (signal_max_span is not None and signal_max_span > 0) else None
        )
        self._true_signal = None

        # Set domain bounds for acquisition.
        bounds = getattr(self.belief, "physical_param_bounds", self.belief.parameter_bounds)
        self._scan_lo, self._scan_hi = bounds[self._scan_param]
        self._full_domain_lo, self._full_domain_hi = float(self._scan_lo), float(self._scan_hi)

        # Post-sweep interval in physical units where _acquire may search; starts at full scan.
        self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
        # Non-scan parameter bounds narrowed (empty = not yet set).
        self._narrowed_param_bounds: dict[str, tuple[float, float]] = {}
        # Buffer for the first _WARMUP_BUFFER_SIZE Bayesian observations (narrow mode only).
        self._warmup_obs_buffer: list[Observation] = []

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = NVISION_CONVERGENCE_THRESHOLD,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
        **grid_config: object,
    ) -> SequentialBayesianLocator:
        """Generic factory for model-agnostic Bayesian locators.

        Subclasses for specific models (like NVCenterBayesianLocator) should
        override this to provide their own hard-coded belief setup.
        """
        if builder is None:
            raise ValueError(
                f"{cls.__name__} requires a 'builder' callable to create the AbstractMarginalDistribution."
            )
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )

    # ------------------------------------------------------------------
    # SMC lifecycle hooks — subclasses may override any stage
    # ------------------------------------------------------------------

    def _native_scan_candidates(self, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (candidates, probabilities) from the belief's native discretization.

        - Grid beliefs: grid points of the scan parameter within [lo, hi],
          probabilities are the normalized posterior masses.
        - SMC beliefs: particle scan-parameter values within [lo, hi],
          probabilities are the normalized particle weights.
        Returns empty arrays when the belief has no native discretization.
        """
        belief = self.belief
        scan = self._scan_param

        if hasattr(belief, "get_grid_param"):
            p = belief.get_grid_param(scan)
            grid = np.asarray(p.grid, dtype=float)
            mask = (grid >= lo) & (grid <= hi)
            if mask.any():
                candidates = grid[mask]
                probs = np.asarray(p.posterior, dtype=float)[mask]
                total = float(probs.sum())
                if total > 0:
                    return candidates, probs / total
            return np.array([]), np.array([])

        if hasattr(belief, "_particles") and hasattr(belief, "_weights"):
            param_names = getattr(belief, "_param_names", None)
            if param_names is not None and scan in param_names:
                idx = param_names.index(scan)
                vals = np.asarray(belief._particles[:, idx], dtype=float)
                # Map internal particles (which may be unit-cube) to physical space
                if hasattr(belief, "physical_param_bounds") and scan in belief.physical_param_bounds:
                    p_lo, p_hi = belief.physical_param_bounds[scan]
                    phys_vals = p_lo + vals * (p_hi - p_lo)
                else:
                    phys_vals = vals

                mask = (phys_vals >= lo) & (phys_vals <= hi)
                if mask.any():
                    candidates = phys_vals[mask]
                    probs = np.asarray(belief._weights, dtype=float)[mask]
                    total = float(probs.sum())
                    if total > 0:
                        return candidates, probs / total
            return np.array([]), np.array([])

        return np.array([]), np.array([])

    def _acquire(self) -> float:
        """Acquisition logic must be implemented by subclasses.

        Returns
        -------
        float
            Position in **physical units** to measure next. The base class
            will automatically normalize this to [0, 1] for the experiment.
        """
        raise NotImplementedError("Subclasses must implement _acquire()")

    def _observe_acquisition(self, obs: Observation) -> None:
        """Route acquisition observations to the belief immediately."""
        self.belief.update(obs)

    def _acquisition_done(self) -> bool:
        """Return whether the Bayesian acquisition phase should stop.

        Default: stop when step budget is exhausted or target parameters
        have stayed converged for ``convergence_patience_steps``.
        """
        return self.inference_step_count >= self.max_steps

    # ------------------------------------------------------------------
    # Locator interface — thin orchestrators that delegate to hooks above
    # ------------------------------------------------------------------

    def next(self) -> float:
        """Propose next measurement."""
        self.step_count += 1
        self.inference_step_count += 1

        physical_value = self._acquire()
        return self._to_experiment_normalized(physical_value)

    def observe(self, obs: Observation) -> None:
        """Route observation to the appropriate lifecycle hook."""
        self._observe_acquisition(obs)

    def done(self) -> bool:
        """Stop when converged or step budget is exhausted."""
        return self._acquisition_done()

    def _target_params_converged(self) -> bool:
        """Check convergence on configured target parameters.

        Convergence requires the uncertainty of each target parameter to be
        below ``convergence_threshold`` as a fraction of its physical bound
        width (e.g. ``0.01`` = 1 %).  Parameters listed in
        ``PARAM_ABSOLUTE_CONVERGENCE_THRESHOLDS`` (from env, e.g.
        ``NVISION_FREQ_CONVERGENCE_THRESHOLD``) use an absolute uncertainty
        ceiling instead.  The overall (RMS) relative uncertainty across all
        target parameters must also be below the same threshold.
        """
        target_params = (
            list(self._convergence_params) if self._convergence_params else list(self.belief.model.parameter_names())
        )
        physical_uncertainties = self.belief.uncertainty()
        bounds = self.belief.physical_param_bounds

        relative_uncertainties: dict[str, float] = {}
        for name in target_params:
            if name not in physical_uncertainties:
                continue
            unc = float(physical_uncertainties[name])
            bound_width = param_convergence_bound_width(name, self.convergence_threshold, bounds)
            if bound_width <= 0:
                return False
            relative_uncertainties[name] = unc / bound_width

        if not relative_uncertainties:
            return False

        # Check 1: Each individual parameter must be below threshold
        individual_converged = all(u < self.convergence_threshold for u in relative_uncertainties.values())
        if not individual_converged:
            return False

        # Check 2: Overall (RMS) relative uncertainty must also be below threshold
        uncertainties_array = np.array(list(relative_uncertainties.values()))
        rms_uncertainty = float(np.sqrt(np.mean(uncertainties_array**2)))
        overall_converged = rms_uncertainty < self.convergence_threshold

        return overall_converged

    def result(self) -> dict[str, float]:
        """Return posterior-mean estimates for all parameters, along with sweep metrics if available."""
        res = dict(self.belief.estimates())
        expected_uniform = self._compute_expected_uniform_points()
        if expected_uniform > 0:
            res["expected_uniform_points"] = expected_uniform
        return res

    def _compute_expected_uniform_points(self) -> float:  # noqa: C901
        """Compute the expected points a simple Sobol sweep would require."""
        true_signal = getattr(self, "_true_signal", None)
        if true_signal is None:
            return 0.0

        bounds = getattr(self.belief, "physical_param_bounds", self.belief.parameter_bounds)
        if self._scan_param not in bounds:
            return 0.0
        lo_phys, hi_phys = bounds[self._scan_param]
        domain_width = hi_phys - lo_phys
        if domain_width <= 0:
            return 0.0

        # 1. Evaluate ground-truth signal
        n = 20000
        xs_phys = np.linspace(lo_phys, hi_phys, n)
        ys = np.array([true_signal(float(x)) for x in xs_phys], dtype=float)
        xs_norm = (xs_phys - lo_phys) / domain_width

        # 2. Find dip segments on the ground-truth signal using noise_std=1e-6
        background = float(np.percentile(ys, 95))
        dip_depth = background - float(np.min(ys))
        if dip_depth <= 0:
            return 0.0
        threshold_drop = max(0.2 * dip_depth, 3.0 * 1e-6)
        threshold = background - threshold_drop
        below = ys < threshold
        if not np.any(below):
            return 0.0
        changes = np.diff(below.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if below[0]:
            starts = np.concatenate([[0], starts])
        if below[-1]:
            ends = np.concatenate([ends, [len(below)]])

        segments = []
        for s, e in zip(starts, ends, strict=False):
            if e > s and (e - s) >= 3:
                segments.append((float(xs_norm[s]), float(xs_norm[e - 1])))

        # Merge segments
        if len(segments) > 1:
            merged_segs = [segments[0]]
            for lo, hi in segments[1:]:
                if lo - merged_segs[-1][1] <= 0.02:
                    merged_segs[-1] = (merged_segs[-1][0], hi)
                else:
                    merged_segs.append((lo, hi))
            segments = merged_segs

        if not segments:
            return 0.0

        # Check if dips are merged
        if len(segments) == 1:
            merged = True
        else:
            total_width = sum(hi - lo for lo, hi in segments)
            total_span = segments[-1][1] - segments[0][0]
            merged = bool(total_span <= total_width * 1.5)

        true_total = float(sum(hi - lo for lo, hi in segments) * domain_width)
        true_span = float((segments[-1][1] - segments[0][0]) * domain_width)

        effective_width = true_span if merged else true_total

        if effective_width > 0:
            return float(2.0 * domain_width / effective_width)

        # Fallback to model min_span
        min_span = None
        model = getattr(true_signal, "model", None)
        if model is not None and hasattr(model, "signal_min_span") and callable(model.signal_min_span):
            min_span = model.signal_min_span(domain_width)
        if min_span is not None and min_span > 0:
            return float(6.0 * domain_width / min_span)

        return float(self.max_steps)

    def _acquisition_bounds(self) -> tuple[float, float]:
        """Physical bounds where :meth:`_acquire` searches (post-sweep window)."""
        bounds = getattr(self.belief, "physical_param_bounds", self.belief.parameter_bounds)
        lo, hi = bounds[self._scan_param]
        return (min(lo, hi), max(lo, hi))

    def effective_initial_sweep_steps(self) -> int:
        """Effective initial sweep step count (always 0 since sweep is removed)."""
        return 0

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Return the tight focus window (always None since sweep is removed)."""
        return None

    def per_dip_windows(self) -> list[tuple[float, float]] | None:
        """Individual per-dip focus windows (always None since sweep is removed)."""
        return None

    def narrowed_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds of non-scan parameters narrowed (always empty since sweep is removed)."""
        return {}

    # ------------------------------------------------------------------
    # Utility helpers available to all acquisition implementations
    # ------------------------------------------------------------------

    def _generate_candidates(self, num_candidates: int) -> np.ndarray:
        """Generate grid from acquisition bounds (log-uniform for scale params)."""
        lo, hi = self._acquisition_bounds()
        is_scale = getattr(self.belief.model, "is_scale_parameter", lambda name: False)(self._scan_param)
        if is_scale and lo > 0 and hi > lo:
            return np.exp(np.linspace(np.log(lo), np.log(hi), num_candidates))
        return np.linspace(lo, hi, num_candidates)

    def _apply_parameter_weight_bias(
        self,
        utilities: np.ndarray,
        mu_preds: np.ndarray,
        sampled: object,
        candidates: np.ndarray | None = None,
    ) -> np.ndarray:
        """Boost utilities toward measurements that are most informative about high-weight parameters.

        For each candidate ``x_i`` the **frequency-specificity** is the squared
        correlation between the signal predictions ``S(x_i, θ)`` and the
        frequency dimension of the posterior samples::

            R²_f(x_i) = Cov(S(x_i,θ), θ_f)² / [Var(S(x_i,θ)) · Var(θ_f)]

        This is in ``[0, 1]``.  A measurement where predictions are perfectly
        correlated with frequency uncertainty gets its utility multiplied by
        ``freq_weight``; a measurement uncorrelated with frequency is unchanged.

        Additionally, when candidates are provided, a **center-frequency proximity**
        bonus is applied: measurements near the posterior mean frequency receive
        higher weight, with a Gaussian falloff::

            center_boost(x_i) = 1.0 + center_freq_weight * exp(-0.5 * ((x_i - f_mean) / f_std)²)

        Only parameters with weight > 1 contribute (default weight = 1 → no-op).

        Parameters
        ----------
        utilities : np.ndarray
            Shape ``(n_candidates,)`` — base acquisition utilities.
        mu_preds : np.ndarray
            Shape ``(n_candidates, n_samples)`` — signal predictions over posterior samples.
        sampled : ParameterValues[np.ndarray]
            Posterior parameter samples (unit-cube or physical; scale does not matter).
        candidates : np.ndarray | None
            Shape ``(n_candidates,)`` — candidate positions in physical units.
            Required for center-frequency proximity weighting.

        Returns
        -------
        np.ndarray
            Biased utilities, same shape as input.
        """
        inner_model = getattr(self.belief.model, "inner", self.belief.model)
        if not hasattr(inner_model, "parameter_weights"):
            return utilities
        weights = inner_model.parameter_weights()

        result = utilities.copy()
        pred_var = np.var(mu_preds, axis=1)  # (n_candidates,)

        for param_name, weight in weights.items():
            if weight <= 1.0:
                continue
            try:
                param_particles = np.asarray(sampled[param_name], dtype=np.float64)
            except (KeyError, TypeError):
                continue
            p_var = float(np.var(param_particles))
            if p_var < 1e-30:
                continue

            p_mean = param_particles.mean()
            p_dev = param_particles - p_mean
            mu_mean = mu_preds.mean(axis=1)
            cov = ((mu_preds - mu_mean[:, None]) * p_dev[None, :]).mean(axis=1)
            r2 = cov**2 / (pred_var * p_var + 1e-30)
            r2 = np.clip(r2, 0.0, 1.0)

            result = result * (1.0 + (weight - 1.0) * r2)

        # Center-frequency proximity weighting: boost utilities near posterior mean frequency
        if candidates is not None and "frequency" in weights:
            freq_weight = weights["frequency"]
            try:
                freq_particles = np.asarray(sampled["frequency"], dtype=np.float64)
                f_mean = float(np.mean(freq_particles))
                f_std = float(np.std(freq_particles))
                if f_std > 1e-12:
                    # Gaussian proximity factor: 1.0 at center, decays with distance
                    z = (candidates - f_mean) / f_std
                    proximity = np.exp(-0.5 * z * z)
                    # Boost: up to (freq_weight - 1.0) additional weight at center
                    center_boost = 1.0 + (freq_weight - 1.0) * proximity
                    result = result * center_boost
            except (KeyError, TypeError):
                pass

        return result

    def _to_experiment_normalized(self, physical_value: float) -> float:
        """Map a physical scan position to ``[0, 1]`` for :meth:`CoreExperiment.measure`."""
        lo, hi = self._full_domain_lo, self._full_domain_hi
        tol = 1e-7 * (hi - lo)
        if physical_value < lo - tol or physical_value > hi + tol:
            raise ValueError(
                f"Proposed physical scan position {physical_value} is outside the full domain bounds {(lo, hi)}"
            )
        unit_val = (physical_value - lo) / (hi - lo)
        return float(np.clip(unit_val, 0.0, 1.0))
