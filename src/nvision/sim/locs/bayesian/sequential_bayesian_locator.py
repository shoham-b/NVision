"""Abstract base class for all Bayesian locators."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping

import numpy as np

from nvision.models.locator import Locator
from nvision.signal.abstract_belief import AbstractBeliefDistribution
from nvision.signal.signal import Parameter


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
        Hard upper bound on Bayesian-inference steps (excludes initial sweep).
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

    def __init__(
        self,
        belief: AbstractBeliefDistribution,
        max_steps: int = 150,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
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

        if initial_sweep_steps is None:
            initial_sweep_steps = 128

        self.initial_sweep_steps = min(int(initial_sweep_steps), self.max_steps)
        self.initial_sweep_steps = max(0, self.initial_sweep_steps)
        self._initial_sweep_builder = initial_sweep_builder or self._sobol_1d
        if self.initial_sweep_steps > 0:
            self._initial_sweep_points = self._initial_sweep_builder(self.initial_sweep_steps)
        else:
            self._initial_sweep_points = np.empty(0, dtype=float)

        self._scan_lo, self._scan_hi = self.belief.get_param(self._scan_param).bounds

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
        )

    @staticmethod
    def _sobol_1d(n: int) -> np.ndarray:
        """Deterministic low-discrepancy 1D sequence on [0, 1]."""

        def vdc(k: int, base: int = 2) -> float:
            v = 0.0
            denom = 1.0
            while k:
                k, remainder = divmod(k, base)
                denom *= base
                v += remainder / denom
            return v

        return np.array([vdc(i + 1) for i in range(n)], dtype=float)

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

        if self.step_count <= self.initial_sweep_steps:
            u = float(self._initial_sweep_points[self.step_count - 1])
            physical_value = self._scan_lo + u * (self._scan_hi - self._scan_lo)
            return self._normalize(self._scan_param, physical_value)

        self.inference_step_count += 1
        physical_value = self._acquire()
        return self._normalize(self._scan_param, physical_value)

    def done(self) -> bool:
        """Stop when converged (after warm-up) or step budget is exhausted."""
        if self.step_count < self.initial_sweep_steps:
            return False
        if self.inference_step_count >= self.max_steps:
            return True
        return self.belief.converged(self.convergence_threshold)

    def result(self) -> dict[str, float]:
        """Return posterior-mean estimates for all parameters."""
        return self.belief.estimates()

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
