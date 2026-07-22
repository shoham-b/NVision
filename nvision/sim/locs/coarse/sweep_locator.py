"""Base class for sweeping locators: predetermined-grid stepping mechanics."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation, ObservationHistory
from nvision.spectra.signal import SignalModel

if TYPE_CHECKING:
    pass


class SweepingLocator(Locator):
    """Base class for sweeping locators: proposes a predetermined grid of points.

    Class Attributes
    ----------------
    USES_SWEEP_MAX_STEPS : bool
        If True, use sweep_max_steps instead of loc_max_steps.
    REQUIRES_BELIEF : bool
        If True, inject belief and signal_model parameters.

    This class only owns the shared sweep mechanics: stepping through a
    predetermined grid, belief bookkeeping, and reading the model's declared
    span constants. Signal detection, windowing, and sweep-efficiency metrics
    are the concrete locator's own responsibility (see ``GenericSweepLocator``
    and ``StagedSobolSweepLocator``, which each implement these differently
    and independently — there is no shared implementation to inherit).

    Subclasses must implement:
    - `_generate_sweep_points(n)`: Generate n sweep points in [0, 1]
    """

    USES_SWEEP_MAX_STEPS: bool = True
    REQUIRES_BELIEF: bool = True

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(belief)
        # Signal model is independent of belief - used for sweep detection
        self.signal_model = signal_model
        self.max_steps = max_steps
        self.step_count = 0
        self.noise_std = noise_std
        self._noise_std = noise_std
        self._noise_max_dev = noise_max_dev
        self._signal_min_span = signal_min_span
        self._signal_max_span = signal_max_span
        _names = signal_model.parameter_names()
        _fixed_vals = getattr(getattr(signal_model, "spec", None), "fixed_values", None) or {}
        if scan_param:
            self._scan_param = scan_param
        elif "frequency" in _names or "frequency" in _fixed_vals:
            # "frequency" is always the probe x-axis for NV-center models even
            # when fixed (not inferred) and therefore absent from _names. Read
            # via signal_model.spec (proxied through wrappers like
            # UnitCubeSignalModel) rather than a private attribute, which
            # doesn't survive wrapping.
            self._scan_param = "frequency"
        else:
            self._scan_param = _names[0] if _names else "x"
        self._domain_lo = domain_lo
        self._domain_hi = domain_hi

        # Generate initial sweep points (subclass provides method)
        self._sweep_points: np.ndarray = np.empty(0, dtype=float)
        self.history = ObservationHistory(max_steps)

        # Acquisition window (set when signal found or sweep completes)
        self._acquisition_lo = domain_lo
        self._acquisition_hi = domain_hi
        self._signal_found = False

    def observe(self, obs: Observation) -> None:
        """Record observation for sweep tracking and update belief.

        Sweep locators track observations for signal detection, and we also
        update the belief so we can show parameter estimates and the
        most likely locator signal in visualizations.
        """
        if self.step_count <= self.max_steps:
            self.history.append(obs)
            # Set last_obs so Observer can create snapshots for plotting
            self.belief.last_obs = obs
            self.belief.update(obs)

    def _inner_model(self):
        """Return the inner physical model (unwraps UnitCubeSignalModel if needed)."""
        return self.signal_model.inner

    def _model_signal_min_span(self) -> float | None:
        """Read signal_min_span from the inner model using the current domain width."""
        domain_width = self._domain_hi - self._domain_lo
        if domain_width <= 0:
            return None
        return self._inner_model().signal_min_span(domain_width)

    def _model_signal_max_span(self) -> float | None:
        """Read signal_max_span from the inner model using the current domain width."""
        domain_width = self._domain_hi - self._domain_lo
        if domain_width <= 0:
            return None
        return self._inner_model().signal_max_span(domain_width)

    @abstractmethod
    def _generate_sweep_points(self, n: int) -> np.ndarray:
        """Generate n sweep points in [0, 1]. Must be implemented by subclass."""
        raise NotImplementedError("Subclasses must implement _generate_sweep_points")

    def next(self) -> float:
        """Propose the next sweep measurement."""
        self.step_count += 1
        if self.step_count <= self.max_steps:
            return float(self._sweep_points[self.step_count - 1])
        return 0.5

    def done(self) -> bool:
        """Return True when sweep is complete."""
        return self.step_count >= self.max_steps

    def result(self) -> dict[str, float]:
        """Return the acquisition window bounds. Subclasses add their own metrics."""
        return {
            "acquisition_lo": self._acquisition_lo,
            "acquisition_hi": self._acquisition_hi,
            "domain_lo": self._domain_lo,
            "domain_hi": self._domain_hi,
            "signal_found": self._signal_found,
            "completed_at_step": self.effective_step_count(),
        }

    def effective_step_count(self) -> int:
        """Effective step count."""
        return self.step_count

    def effective_initial_sweep_steps(self) -> int:
        """Return effective sweep steps for UI phase coloring.

        This is called by the Observer to determine how many steps
        were part of the initial sweep phase (for marking measurements
        as 'coarse' phase in visualizations).

        Returns
        -------
        int
            Number of steps in the initial sweep phase.
        """
        return self.effective_step_count()


