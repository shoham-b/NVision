"""Abstract base class for all Bayesian locators."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation, ObservationHistory
from nvision.sim.locs.coarse.sobol_locator import StagedSobolSweepLocator
from nvision.sim.locs.refocus import infer_focus_window_physical as _refocus_infer_focus_window

_POSTERIOR_NARROWING_INTERVAL: int = 20
_POSTERIOR_CREDIBLE_LEVEL: float = 0.95
_POSTERIOR_MIN_NARROWING_FRACTION: float = 0.05


def _posterior_credible_interval(
    belief: AbstractMarginalDistribution,
    param_name: str,
    level: float = _POSTERIOR_CREDIBLE_LEVEL,
) -> tuple[float, float] | None:
    """Return a ``level``-credible interval in physical units for ``param_name``.

    Uses weighted SMC particle quantiles when available. Returns ``None`` if the interval
    cannot be computed or is degenerate.
    """
    tail = (1.0 - level) / 2.0
    lo_phys, hi_phys = belief.physical_param_bounds[param_name]
    if hi_phys <= lo_phys:
        return None

    particles = getattr(belief, "_particles", None)
    param_names = getattr(belief, "_param_names", None)
    if particles is None or param_names is None:
        return None

    j = param_names.index(param_name)
    u_vals = particles[:, j]
    u_lo = float(np.quantile(u_vals, tail))
    u_hi = float(np.quantile(u_vals, 1.0 - tail))
    return (lo_phys + u_lo * (hi_phys - lo_phys), lo_phys + u_hi * (hi_phys - lo_phys))


class SequentialBayesianLocator(Locator):
    """Shared Bayesian loop infrastructure for all acquisition strategies.

    Handles the mechanics common to every Bayesian locator:
    - incrementing the step counter
    - convergence-based stopping
    - extracting results from the belief posterior

    After an initial Sobol sweep, a **single** physical interval is derived from the
    sweep data; all Bayesian :meth:`_acquire` calls search only inside that interval
    (same interval shown in the scan UI as the focus band). For unit-cube beliefs,
    the scan parameter's physical bounds (and probe-axis mapping when it matches the
    sweep axis) are updated accordingly so normalization and posteriors stay consistent.

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
        Hard upper bound on Bayesian inference steps (excludes initial sweep).
    convergence_threshold : float
        Relative uncertainty threshold (fraction of bound width) below which
        we consider parameters converged and stop early.  Default ``0.01`` = 1 %.
    scan_param : str | None
        The parameter we are proposing measurements along. Defaults to the
        first parameter in the belief.
    initial_sweep_steps : int | None
        Number of initial coarse sweep measurements to take before Bayesian
        acquisition starts. If ``None``, ``max_steps`` is used as the upper bound.
    """

    DEFAULT_INITIAL_SWEEP_STEPS = 24

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        max_steps: int = 450,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        noise_std: float | None = None,
        noise_max_dev: float | None = None,
        signal_max_span: float | None = None,
    ) -> None:
        super().__init__(belief)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.convergence_threshold = convergence_threshold
        # Total measurement count (includes initial sweep + Bayesian steps).
        self.step_count: int = 0
        # Bayesian acquisition count only (excludes initial sweep).
        self.inference_step_count: int = 0
        self._scan_param = scan_param or belief.model.parameter_names()[0]
        # Default convergence target is all model parameters.
        self._convergence_params: tuple[str, ...] = (
            tuple(convergence_params) if convergence_params is not None else tuple(self.belief.model.parameter_names())
        )
        self._convergence_patience_steps = max(1, int(convergence_patience_steps))
        self._convergence_streak = 0
        # Stored for noise-aware dip detection during sweep refocus.
        if noise_std is None or float(noise_std) <= 0:
            raise ValueError(f"noise_std must be a positive float; got {noise_std!r}")
        self._noise_std: float = float(noise_std)
        # Pre-computed maximum expected noise deviation for mid-sweep threshold.
        # When provided by the executor (via CompositeNoise.estimated_max_noise_deviation),
        # this is used directly as the dip threshold, bypassing the IQR fallback.
        self._noise_max_dev: float | None = (
            float(noise_max_dev) if (noise_max_dev is not None and noise_max_dev > 0) else None
        )
        # Physical max signal span (from signal spec's _signal_max_span bound).
        # Drives both sweep density and refocus window width directly.
        self._signal_max_span: float | None = (
            float(signal_max_span) if (signal_max_span is not None and signal_max_span > 0) else None
        )

        # Set domain bounds for sweep and acquisition.
        # Beliefs that carry physical bounds separately (unit-cube variants) expose
        # ``physical_param_bounds``; otherwise fall back to ``parameter_bounds``.
        bounds = getattr(self.belief, "physical_param_bounds", self.belief.parameter_bounds)
        self._scan_lo, self._scan_hi = bounds[self._scan_param]
        # Full scan axis for :class:`~nvision.models.experiment.CoreExperiment` (never narrowed).
        # Returned ``x`` must stay normalized to this full range so ``measure()`` probes
        # the intended frequency, even when the belief is narrowed after the sweep.
        self._full_domain_lo, self._full_domain_hi = float(self._scan_lo), float(self._scan_hi)

        # StagedSobolSweepLocator already performs its own signal-span detection
        # and stage transitions (1/2/3).  We let it run until it reports done()
        # rather than pre-computing a sweep length here.
        if initial_sweep_steps is None:
            self.initial_sweep_steps = self.max_steps
        else:
            self.initial_sweep_steps = max(0, int(initial_sweep_steps))

        # Create staged Sobol locator for initial sweep phase
        # belief is passed to satisfy Locator parent class
        # signal_model (belief.model) is used for sweep detection
        self._staged_sobol = StagedSobolSweepLocator(
            belief=self.belief,
            signal_model=self.belief.model,
            max_steps=self.initial_sweep_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
            scan_param=self._scan_param,
            domain_lo=self._full_domain_lo,
            domain_hi=self._full_domain_hi,
        )

        # Post-sweep interval in physical units where _acquire may search; starts at full scan.
        self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
        # Non-scan parameter bounds narrowed after the sweep (empty = not yet set).
        self._narrowed_param_bounds: dict[str, tuple[float, float]] = {}
        # Track actual step count when initial sweep completed (including any fallback)
        self._initial_sweep_completed_at_step: int | None = None
        # Per-dip focus windows: list of (lo, hi) tuples for individual dip targeting
        self._per_dip_windows: list[tuple[float, float]] | None = None
        # Current dip window index for round-robin acquisition across multiple dips
        self._current_dip_window_idx: int = 0
        # Buffer for sweep observations - batch update belief after sweep completes.
        # Capacity must cover the full staged-sobol budget, which is now max_steps.
        self._sweep_buffer = ObservationHistory(self.max_steps)

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
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
            initial_sweep_steps=initial_sweep_steps,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_max_span=signal_max_span,
        )

    # ------------------------------------------------------------------
    # SMC lifecycle hooks — subclasses may override any stage
    # ------------------------------------------------------------------

    def _sweep_next(self) -> float:
        """Return the next measurement point during the initial sweep phase."""
        return self._staged_sobol.next()

    def _on_sweep_complete(self) -> None:  # noqa: C901
        """Hook called once when the initial sweep phase finishes.

        Default implementation finalizes the staged locator, performs a batch
        belief update with all sweep observations, narrows the scan parameter
        bounds, and extracts per-dip windows if available.
        """
        if self._initial_sweep_completed_at_step is None:
            self._initial_sweep_completed_at_step = self._staged_sobol.step_count

        # Finalize the sweep (rigorously trims baseline tails for multi-dip)
        if hasattr(self._staged_sobol, "finalize"):
            self._staged_sobol.finalize()

        # Batch update belief with all sweep observations at once
        if self._sweep_buffer.count > 0:
            if hasattr(self.belief, "batch_update"):
                self.belief.batch_update(self._sweep_buffer.observations)
            else:
                for obs in self._sweep_buffer.observations:
                    self.belief.update(obs)
            self._sweep_buffer = ObservationHistory(self.max_steps)

        # Get acquisition window from staged Sobol locator
        self._acquisition_lo, self._acquisition_hi = self._staged_sobol.acquisition_window()

        # Propagate per-dip windows for UI overlay when available
        if hasattr(self._staged_sobol, "per_dip_windows"):
            pdw = self._staged_sobol.per_dip_windows()
            if pdw is not None:
                self._per_dip_windows = pdw

        # Narrow belief scan bounds (no-op for physical-space beliefs)
        self.belief.narrow_scan_parameter_physical_bounds(
            self._scan_param, self._acquisition_lo, self._acquisition_hi
        )
        phys_bounds = self.belief.physical_param_bounds
        slo, shi = phys_bounds[self._scan_param]
        self._acquisition_lo = min(slo, shi)
        self._acquisition_hi = max(slo, shi)

        if "split" in phys_bounds:
            window_width = self._acquisition_hi - self._acquisition_lo
            if window_width > 0:
                max_split = window_width / 2.0
                split_lo, split_hi = phys_bounds["split"]
                new_split_hi = min(float(split_hi), max_split)
                if new_split_hi > float(split_lo):
                    self.belief.narrow_scan_parameter_physical_bounds("split", float(split_lo), new_split_hi)

        self._narrowed_param_bounds = {
            name: (float(lo), float(hi)) for name, (lo, hi) in self.belief.physical_param_bounds.items()
        }

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
                mask = (vals >= lo) & (vals <= hi)
                if mask.any():
                    candidates = vals[mask]
                    probs = np.asarray(belief._weights, dtype=float)[mask]
                    total = float(probs.sum())
                    if total > 0:
                        return candidates, probs / total
            return np.array([]), np.array([])

        return np.array([]), np.array([])

    def _acquire(self) -> float:
        """Default acquisition: measure where the marginal posterior is maximised.

        Uses the belief's native discretization (grid or particles) when available,
        otherwise falls back to a uniform candidate grid. Subclasses may override
        this method entirely or override :meth:`_candidate_utilities` to keep the
        same selection logic with different weights.

        Returns
        -------
        float
            Position in **physical units** to measure next. The base class
            will automatically normalize this to [0, 1] for the experiment.
        """
        lo, hi = self._acquisition_bounds()
        candidates, probs = self._native_scan_candidates(lo, hi)
        if len(candidates) == 0:
            candidates = self._generate_candidates()
            pdf = self.belief.marginal_pdf(self._scan_param, candidates)
            pdf = np.asarray(pdf, dtype=float)
            total = float(np.sum(pdf))
            if not np.isfinite(total) or total <= 0.0:
                return float(np.random.choice(candidates))
            probs = pdf / total
        return float(candidates[int(np.argmax(probs))])

    def _observe_sweep(self, obs: Observation) -> None:
        """Handle an observation arriving during the initial sweep phase."""
        self._sweep_buffer.append(obs)
        self._staged_sobol.observe(obs)

    def _observe_acquisition(self, obs: Observation) -> None:
        """Handle an observation arriving during the Bayesian acquisition phase."""
        super().observe(obs)

    def _acquisition_done(self) -> bool:
        """Return whether the Bayesian acquisition phase should stop.

        Default: stop when step budget is exhausted or target parameters
        have stayed converged for ``convergence_patience_steps``.
        """
        if self.inference_step_count >= self.max_steps:
            return True
        if self._target_params_converged():
            self._convergence_streak += 1
        else:
            self._convergence_streak = 0
        return self._convergence_streak >= self._convergence_patience_steps

    # ------------------------------------------------------------------
    # Locator interface — thin orchestrators that delegate to hooks above
    # ------------------------------------------------------------------

    def next(self) -> float:
        """Propose next measurement with staged initial-sweep warm-start."""
        self.step_count += 1

        if not self._staged_sobol.done():
            value = self._sweep_next()
            if self._staged_sobol.done() and self._initial_sweep_completed_at_step is None:
                self._on_sweep_complete()
            return value

        # Sweep already finished before we could record it
        if self._initial_sweep_completed_at_step is None:
            self._on_sweep_complete()

        self.inference_step_count += 1
        physical_value = self._acquire()
        return self._to_experiment_normalized(physical_value)

    def _narrow_non_scan_params_from_posterior(self) -> None:
        """Periodically tighten non-scan parameter bounds using the current posterior.

        Computes a :data:`_POSTERIOR_CREDIBLE_LEVEL` credible interval from the
        live marginal posterior for every non-scan parameter and calls
        :meth:`narrow_scan_parameter_physical_bounds` to shrink the belief.
        Only narrows by at least :data:`_POSTERIOR_MIN_NARROWING_FRACTION` of the
        current prior width — never widens.
        """
        if not hasattr(self.belief, "_particles"):
            return
        param_names = list(self.belief.model.parameter_names())
        for param in param_names:
            if param == self._scan_param:
                continue
            if param not in self.belief.physical_param_bounds:
                continue
            interval = _posterior_credible_interval(self.belief, param)
            if interval is None:
                continue
            new_lo, new_hi = interval
            cur_lo, cur_hi = self.belief.physical_param_bounds[param]
            cur_width = cur_hi - cur_lo
            if cur_width <= 0:
                continue
            new_lo = max(new_lo, cur_lo)
            new_hi = min(new_hi, cur_hi)
            if new_hi <= new_lo:
                continue
            if (cur_width - (new_hi - new_lo)) / cur_width < _POSTERIOR_MIN_NARROWING_FRACTION:
                continue
            self.belief.narrow_scan_parameter_physical_bounds(param, new_lo, new_hi)

    def observe(self, obs: Observation) -> None:
        """Route observation to the appropriate lifecycle hook."""
        if not self._staged_sobol.done():
            self._observe_sweep(obs)
        else:
            self._observe_acquisition(obs)

    def done(self) -> bool:
        """Stop when converged (after warm-up) or step budget is exhausted."""
        if not self._staged_sobol.done():
            return False
        return self._acquisition_done()

    def _target_params_converged(self) -> bool:
        """Check convergence on configured target parameters.

        Convergence requires the uncertainty of each target parameter to be
        below ``convergence_threshold`` as a fraction of its physical bound
        width (e.g. ``0.01`` = 1 %).  The overall (RMS) relative uncertainty
        across all target parameters must also be below the same threshold.
        """
        target_params = (
            list(self._convergence_params) if self._convergence_params else list(self.belief.model.parameter_names())
        )
        physical_uncertainties = self.belief.uncertainty()

        relative_uncertainties: dict[str, float] = {}
        for name in target_params:
            if name not in physical_uncertainties:
                continue
            unc = float(physical_uncertainties[name])
            lo, hi = self.belief.physical_param_bounds.get(name, (0.0, 0.0))
            bound_width = hi - lo
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
        """Return posterior-mean estimates for all parameters."""
        return self.belief.estimates()

    def _acquisition_bounds(self) -> tuple[float, float]:
        """Physical bounds where :meth:`_acquire` searches (post-sweep window).

        When per-dip windows are active (for multi-dip signals like NV center),
        returns windows in round-robin order to focus measurements on each dip.
        """
        lo, hi = self._get_current_acquisition_bounds()
        return (min(lo, hi), max(lo, hi))

    def _get_current_acquisition_bounds(self) -> tuple[float, float]:
        """Return the current acquisition window bounds.

        If per-dip windows are active, returns the current dip's window
        and advances to the next dip for subsequent calls (round-robin).
        Otherwise returns the single acquisition window.
        """
        if self._per_dip_windows is not None and len(self._per_dip_windows) > 0:
            # Round-robin through per-dip windows
            window = self._per_dip_windows[self._current_dip_window_idx]
            self._current_dip_window_idx = (self._current_dip_window_idx + 1) % len(self._per_dip_windows)
            return window
        return (self._acquisition_lo, self._acquisition_hi)

    def effective_initial_sweep_steps(self) -> int:
        """Effective initial sweep step count including any fallback sweep."""
        if self._initial_sweep_completed_at_step is None:
            return self.step_count
        return self._initial_sweep_completed_at_step

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Return the tight focus window inferred after the staged initial sweep.

        Infers the focus window directly from the sweep history using the same
        ``_infer_tight_focus_window`` logic that ``StagedSobolSweepLocator`` uses.
        This keeps the focus-window computation consistent regardless of whether
        the sweep locator exposes its own ``bayesian_focus_window`` method.
        """
        # Sweep must have produced enough observations for reliable inference.
        if self._staged_sobol.history.count < 6:
            return None

        lo, hi = self._acquisition_lo, self._acquisition_hi
        # Prefer the staged sobol's own focus window if it exposes one.
        if hasattr(self._staged_sobol, "bayesian_focus_window"):
            focus = self._staged_sobol.bayesian_focus_window()
            if focus is not None:
                lo, hi = focus

        slo, shi = self._full_domain_lo, self._full_domain_hi
        # If neither the staged sobol nor the cached acquisition bounds narrowed
        # the window, infer directly from the sweep history.
        if (hi - lo) >= (shi - slo) * (1.0 - 1e-9):
            # Compute noise threshold consistent with legacy _infer_tight_focus_window
            ys = self._staged_sobol.history.ys
            p30 = float(np.percentile(ys, 30))
            noise_pts = ys[ys >= p30]
            noise_med = float(np.median(noise_pts))
            min_y = float(np.min(ys))
            dip_depth = noise_med - min_y
            noise_threshold = noise_med - 0.5 * dip_depth
            lo, hi = _refocus_infer_focus_window(self._staged_sobol.history, slo, shi, noise_threshold=noise_threshold)

        if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(slo) and np.isfinite(shi)):
            return None
        if hi <= lo or shi <= slo:
            return None
        span = shi - slo
        if span <= 0:
            return None
        if (hi - lo) >= span * (1.0 - 1e-9):
            return None
        return (lo, hi)

    def per_dip_windows(self) -> list[tuple[float, float]] | None:
        """Individual per-dip focus windows for multi-dip signals (e.g., NV center triplets).

        Returns a list of (lo, hi) tuples when per-dip targeting is active,
        or None when using a single acquisition window.
        """
        return self._per_dip_windows

    def narrowed_param_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical bounds of non-scan parameters narrowed after the initial sweep.

        Returns an empty dict when no sweep has been completed or no parameters
        could be narrowed.

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping of parameter name → ``(lo, hi)`` in physical units after
            narrowing.  Only parameters that were genuinely tightened are included.
        """
        return dict(self._narrowed_param_bounds)

    # ------------------------------------------------------------------
    # Utility helpers available to all acquisition implementations
    # ------------------------------------------------------------------

    def _generate_candidates(self, num_candidates: int = 2000) -> np.ndarray:
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
        return float(np.clip((physical_value - lo) / (hi - lo), 0.0, 1.0))
