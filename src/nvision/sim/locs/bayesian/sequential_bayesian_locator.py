"""Abstract base class for all Bayesian locators."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nvision.models.locator import Locator
from nvision.models.observation import Observation
from nvision.signal.abstract_belief import AbstractBeliefDistribution
from nvision.signal.signal import Parameter
from nvision.sim.locs.coarse.sobol_locator import sobol_1d_sequence


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
    belief : AbstractBeliefDistribution
        Initial belief (usually a flat / uniform prior over all parameters).
    max_steps : int
        Hard upper bound on Bayesian inference steps (excludes initial sweep).
    convergence_threshold : float
        Posterior-uncertainty threshold below which we consider all parameters
        converged and stop early.
    scan_param : str | None
        The parameter we are proposing measurements along. Defaults to the
        first parameter in the belief.
    initial_sweep_steps : int | None
        Number of initial coarse sweep measurements to take before Bayesian
        acquisition starts. If ``None``, a heuristic based on ``max_steps`` is used.
    initial_sweep_builder : Callable[[int], np.ndarray] | None
        Builder for initial sweep points in normalized ``[0, 1]`` coordinates.
        Defaults to a Sobol-like low-discrepancy 1D sequence.
    """

    DEFAULT_INITIAL_SWEEP_STEPS = 64

    def __init__(
        self,
        belief: AbstractBeliefDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        focus_after_sweep: bool = True,
        focus_info_quantile: float = 0.6,
        focus_padding_fraction: float = 0.1,
        focus_min_width_fraction: float = 0.15,
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
        self._scan_param = scan_param or belief.parameters[0].name
        # Default convergence target is all model parameters.
        self._convergence_params: tuple[str, ...] = (
            tuple(convergence_params) if convergence_params is not None else tuple(self.belief.model.parameter_names())
        )
        self._convergence_patience_steps = max(1, int(convergence_patience_steps))
        self._convergence_streak = 0
        self._focus_after_sweep = bool(focus_after_sweep)
        self._focus_info_quantile = float(np.clip(focus_info_quantile, 0.0, 0.95))
        self._focus_padding_fraction = float(max(0.0, focus_padding_fraction))
        self._focus_min_width_fraction = float(np.clip(focus_min_width_fraction, 0.0, 1.0))

        if initial_sweep_steps is None:
            initial_sweep_steps = self.DEFAULT_INITIAL_SWEEP_STEPS

        self.initial_sweep_steps = max(0, int(initial_sweep_steps))
        self._initial_sweep_builder = initial_sweep_builder or sobol_1d_sequence
        if self.initial_sweep_steps > 0:
            self._initial_sweep_points = self._initial_sweep_builder(self.initial_sweep_steps)
        else:
            self._initial_sweep_points = np.empty(0, dtype=float)

        self._scan_lo, self._scan_hi = self.belief.get_param(self._scan_param).bounds
        self._focus_lo, self._focus_hi = self._scan_lo, self._scan_hi
        self._sweep_observations: list[Observation] = []

    @classmethod
    def create(
        cls,
        builder: Callable[..., AbstractBeliefDistribution] | None = None,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
        focus_after_sweep: bool = True,
        focus_info_quantile: float = 0.6,
        focus_padding_fraction: float = 0.1,
        focus_min_width_fraction: float = 0.15,
        **grid_config: object,
    ) -> SequentialBayesianLocator:
        """Generic factory for model-agnostic Bayesian locators.

        Subclasses for specific models (like NVCenterBayesianLocator) should
        override this to provide their own hard-coded belief setup.
        """
        if builder is None:
            raise ValueError(f"{cls.__name__} requires a 'builder' callable to create the AbstractBeliefDistribution.")
        belief = builder(parameter_bounds, **grid_config)
        return cls(
            belief,
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            scan_param=scan_param,
            initial_sweep_steps=initial_sweep_steps,
            initial_sweep_builder=initial_sweep_builder,
            convergence_params=convergence_params,
            convergence_patience_steps=convergence_patience_steps,
            focus_after_sweep=focus_after_sweep,
            focus_info_quantile=focus_info_quantile,
            focus_padding_fraction=focus_padding_fraction,
            focus_min_width_fraction=focus_min_width_fraction,
        )

    @property
    def scan_posterior(self) -> Parameter:
        """Get the parameter bounds for the scan parameter."""
        return self.belief.get_param(self._scan_param)

    # ------------------------------------------------------------------
    # Extension point — the ONLY thing concrete subclasses must add
    # ------------------------------------------------------------------

    @abstractmethod
    def _acquire(self) -> float:
        """Choose the next measurement position from the current belief.

        Called once per step, *after* ``step_count`` has been incremented by
        ``next()``, so ``self.step_count`` is the index of the step being
        planned (1-based).

        Returns
        -------
        float
            Position in **physical units** to measure next. The base class
            will automatically normalize this to [0, 1] for the experiment.
        """

    # ------------------------------------------------------------------
    # Locator interface — implemented once for all Bayesian subclasses
    # ------------------------------------------------------------------

    def next(self) -> float:
        """Propose next measurement with initial-sweep warm-start."""
        self.step_count += 1

        if self.step_count == self.initial_sweep_steps + 1:
            self._update_focus_bounds_from_sweep()

        if self.step_count <= self.initial_sweep_steps:
            u = float(self._initial_sweep_points[self.step_count - 1])
            physical_value = self._scan_lo + u * (self._scan_hi - self._scan_lo)
            return self._normalize(self._scan_param, physical_value)

        self.inference_step_count += 1
        physical_value = self._acquire()
        return self._normalize(self._scan_param, physical_value)

    def observe(self, obs: Observation) -> None:
        """Update belief and cache Sobol-sweep observations for focus detection."""
        super().observe(obs)
        if self.step_count <= self.initial_sweep_steps:
            self._sweep_observations.append(obs)

    def done(self) -> bool:
        """Stop when converged (after warm-up) or step budget is exhausted."""
        if self.step_count < self.initial_sweep_steps:
            return False
        if self.inference_step_count >= self.max_steps:
            return True
        if self._target_params_converged():
            self._convergence_streak += 1
        else:
            self._convergence_streak = 0
        return self._convergence_streak >= self._convergence_patience_steps

    def _target_params_converged(self) -> bool:
        """Check convergence on configured target parameters.

        We use normalized uncertainties when available to keep thresholds like
        `0.01` meaningful across differently-scaled physical parameters.
        """
        if not self._convergence_params:
            return self.belief.converged(self.convergence_threshold)

        # Grid-style beliefs expose internal normalized marginals via `parameters`.
        if hasattr(self.belief, "parameters"):
            params = self.belief.parameters
            by_name = {getattr(p, "name", ""): p for p in params}
            if all(name in by_name for name in self._convergence_params):
                return all(
                    float(by_name[name].uncertainty()) < self.convergence_threshold for name in self._convergence_params
                )

        # SMC-style beliefs expose normalized per-dimension std via `_marginal_std`.
        if hasattr(self.belief, "_param_names") and hasattr(self.belief, "_marginal_std"):
            names = list(self.belief._param_names)
            if all(name in names for name in self._convergence_params):
                return all(
                    float(self.belief._marginal_std(names.index(name))) < self.convergence_threshold
                    for name in self._convergence_params
                )

        return self.belief.converged(self.convergence_threshold)

    def result(self) -> dict[str, float]:
        """Return posterior-mean estimates for all parameters."""
        return self.belief.estimates()

    def _acquisition_bounds(self) -> tuple[float, float]:
        """Current physical scan bounds for Bayesian acquisition."""
        return self._focus_lo, self._focus_hi

    def _update_focus_bounds_from_sweep(self) -> None:
        """Restrict Bayesian search to informative Sobol sweep region."""
        if not self._focus_after_sweep or len(self._sweep_observations) < 3:
            self._focus_lo, self._focus_hi = self._scan_lo, self._scan_hi
            return

        xs = np.array([float(o.x) for o in self._sweep_observations], dtype=float)
        ys = np.array([float(o.signal_value) for o in self._sweep_observations], dtype=float)
        if xs.size == 0:
            return

        center = float(np.median(ys))
        info = np.abs(ys - center)
        thr = float(np.quantile(info, self._focus_info_quantile))
        keep = info >= thr
        if not np.any(keep):
            self._focus_lo, self._focus_hi = self._scan_lo, self._scan_hi
            return

        x_inf_norm = np.sort(xs[keep])
        lo_norm = float(x_inf_norm[0])
        hi_norm = float(x_inf_norm[-1])

        # Pad and enforce a minimum width so Bayesian phase still has room to explore.
        span_norm = hi_norm - lo_norm
        lo_norm -= self._focus_padding_fraction * max(span_norm, 1e-9)
        hi_norm += self._focus_padding_fraction * max(span_norm, 1e-9)
        lo_norm = float(np.clip(lo_norm, 0.0, 1.0))
        hi_norm = float(np.clip(hi_norm, 0.0, 1.0))
        min_width = self._focus_min_width_fraction
        if hi_norm - lo_norm < min_width:
            mid = 0.5 * (lo_norm + hi_norm)
            half = 0.5 * min_width
            lo_norm = float(np.clip(mid - half, 0.0, 1.0))
            hi_norm = float(np.clip(mid + half, 0.0, 1.0))
            if hi_norm - lo_norm < min_width:
                lo_norm = max(0.0, hi_norm - min_width)
                hi_norm = min(1.0, lo_norm + min_width)

        self._focus_lo = self._denormalize(self._scan_param, lo_norm)
        self._focus_hi = self._denormalize(self._scan_param, hi_norm)

    # ------------------------------------------------------------------
    # Utility helpers available to all acquisition implementations
    # ------------------------------------------------------------------

    def _normalize(self, param_name: str, physical_value: float) -> float:
        """Convert a physical parameter value to the normalised [0, 1] range.

        Uses the bounds stored in the belief's corresponding
        ``ParameterWithPosterior`` so subclasses never need to hard-code
        domain limits.

        Parameters
        ----------
        param_name : str
            Name of the parameter whose bounds define the normalisation.
        physical_value : float
            Value in physical units to convert.

        Returns
        -------
        float
            Clipped normalised value in [0, 1].
        """
        param = self.belief.get_param(param_name)
        lo, hi = param.bounds
        return float(np.clip((physical_value - lo) / (hi - lo), 0.0, 1.0))

    def _denormalize(self, param_name: str, normalized_value: float) -> float:
        """Convert a normalised [0, 1] value back to physical units.

        Parameters
        ----------
        param_name : str
            Name of the parameter whose bounds define the mapping.
        normalized_value : float
            Value in [0, 1] to convert.

        Returns
        -------
        float
            Value in physical units.
        """
        param = self.belief.get_param(param_name)
        lo, hi = param.bounds
        return float(lo + normalized_value * (hi - lo))
