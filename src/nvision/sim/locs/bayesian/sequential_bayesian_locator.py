"""Abstract base class for all Bayesian locators."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from nvision.belief.abstract_belief import AbstractBeliefDistribution
from nvision.belief.unit_cube_grid_belief import UnitCubeGridBeliefDistribution
from nvision.belief.unit_cube_smc_belief import UnitCubeSMCBeliefDistribution
from nvision.models.locator import Locator
from nvision.models.observation import Observation
from nvision.parameter import Parameter
from nvision.sim.locs.coarse.sobol_locator import sobol_1d_sequence


def _normalized_acquisition_interval_from_sweep(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    """Return ``(lo, hi)`` in normalized [0, 1] scan coordinates, or ``None`` to use the full domain."""
    if xs.size < 3 or ys.size < 3:
        return None

    info_quantile = 0.6
    padding_fraction = 0.1
    min_width_fraction = 0.15
    segment_peak_ratio = 0.6
    merge_gap_factor = 2.0

    center = float(np.median(ys))
    info = np.abs(ys - center)
    thr = float(np.quantile(info, info_quantile))
    keep = info >= thr
    if not np.any(keep):
        return None

    order = np.argsort(xs)
    xs_sorted = xs[order]
    info_sorted = info[order]
    keep_sorted = keep[order]
    x_keep = xs_sorted[keep_sorted]
    info_keep = info_sorted[keep_sorted]
    if x_keep.size == 0:
        return None

    diffs_all = np.diff(xs_sorted)
    positive_diffs = diffs_all[diffs_all > 0]
    median_dx = float(np.median(positive_diffs)) if positive_diffs.size else 0.0
    split_gap = max(3.0 * median_dx, 1e-6)
    seg_breaks = np.where(np.diff(x_keep) > split_gap)[0] + 1
    seg_xs = np.split(x_keep, seg_breaks)
    seg_infos = np.split(info_keep, seg_breaks)
    seg_peaks = np.array([float(np.max(s)) for s in seg_infos], dtype=float)
    if seg_peaks.size == 0:
        return None
    peak_thr = segment_peak_ratio * float(np.max(seg_peaks))
    selected = [i for i, peak in enumerate(seg_peaks) if peak >= peak_thr]
    if not selected:
        selected = [int(np.argmax(seg_peaks))]
    selected_intervals = sorted((float(seg_xs[i][0]), float(seg_xs[i][-1])) for i in selected)
    merge_gap = merge_gap_factor * median_dx
    merged: list[list[float]] = []
    for start, end in selected_intervals:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    lo_norm = float(merged[0][0])
    hi_norm = float(merged[-1][1])

    span_norm = hi_norm - lo_norm
    lo_norm -= padding_fraction * max(span_norm, 1e-9)
    hi_norm += padding_fraction * max(span_norm, 1e-9)
    lo_norm = float(np.clip(lo_norm, 0.0, 1.0))
    hi_norm = float(np.clip(hi_norm, 0.0, 1.0))
    if hi_norm - lo_norm < min_width_fraction:
        mid = 0.5 * (lo_norm + hi_norm)
        half = 0.5 * min_width_fraction
        lo_norm = float(np.clip(mid - half, 0.0, 1.0))
        hi_norm = float(np.clip(mid + half, 0.0, 1.0))
        if hi_norm - lo_norm < min_width_fraction:
            lo_norm = max(0.0, hi_norm - min_width_fraction)
            hi_norm = min(1.0, lo_norm + min_width_fraction)

    return (lo_norm, hi_norm)


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

    DEFAULT_INITIAL_SWEEP_STEPS = 32

    def __init__(
        self,
        belief: AbstractBeliefDistribution,
        max_steps: int = 450,
        convergence_threshold: float = 0.01,
        scan_param: str | None = None,
        initial_sweep_steps: int | None = None,
        initial_sweep_builder: Callable[[int], np.ndarray] | None = None,
        convergence_params: Sequence[str] | None = None,
        convergence_patience_steps: int = 8,
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

        if initial_sweep_steps is None:
            initial_sweep_steps = self.DEFAULT_INITIAL_SWEEP_STEPS

        self.initial_sweep_steps = max(0, int(initial_sweep_steps))
        self._initial_sweep_builder = initial_sweep_builder or sobol_1d_sequence
        if self.initial_sweep_steps > 0:
            self._initial_sweep_points = self._initial_sweep_builder(self.initial_sweep_steps)
        else:
            self._initial_sweep_points = np.empty(0, dtype=float)

        self._scan_lo, self._scan_hi = self.belief.get_param(self._scan_param).bounds
        # Full scan axis for :class:`~nvision.models.experiment.CoreExperiment` (never narrowed).
        # Belief / :meth:`_normalize` may use a tighter domain after the sweep; returned ``x`` must
        # stay normalized to this full range so ``measure()`` probes the intended frequency.
        self._full_domain_lo, self._full_domain_hi = float(self._scan_lo), float(self._scan_hi)
        # Post-sweep interval in physical units where _acquire may search; starts at full scan.
        self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
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
            self._set_acquisition_window_after_sweep()

        if self.step_count <= self.initial_sweep_steps:
            u = float(self._initial_sweep_points[self.step_count - 1])
            is_scale = getattr(self.belief.model, "is_scale_parameter", lambda name: False)(self._scan_param)
            lo, hi = self._full_domain_lo, self._full_domain_hi
            if is_scale and lo > 0 and hi > lo:
                physical_value = float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
            else:
                physical_value = lo + u * (hi - lo)
            return self._to_experiment_normalized(physical_value)

        self.inference_step_count += 1
        physical_value = self._acquire()
        return self._to_experiment_normalized(physical_value)

    def observe(self, obs: Observation) -> None:
        """Update belief and record sweep observations for the post-sweep window."""
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
        """Physical bounds where :meth:`_acquire` searches (post-sweep window)."""
        lo, hi = float(self._acquisition_lo), float(self._acquisition_hi)
        return (min(lo, hi), max(lo, hi))

    def _set_acquisition_window_after_sweep(self) -> None:
        """Set the single acquisition interval from informative sweep samples (fixed heuristic)."""
        xs = np.array([float(o.x) for o in self._sweep_observations], dtype=float)
        ys = np.array([float(o.signal_value) for o in self._sweep_observations], dtype=float)
        span = _normalized_acquisition_interval_from_sweep(xs, ys)
        if span is None:
            self._acquisition_lo, self._acquisition_hi = self._scan_lo, self._scan_hi
            return
        lo_norm, hi_norm = span
        self._acquisition_lo = self._denormalize(self._scan_param, lo_norm)
        self._acquisition_hi = self._denormalize(self._scan_param, hi_norm)

        if isinstance(self.belief, (UnitCubeGridBeliefDistribution, UnitCubeSMCBeliefDistribution)):
            self.belief.narrow_scan_parameter_physical_bounds(
                self._scan_param,
                self._acquisition_lo,
                self._acquisition_hi,
            )
            slo, shi = self.belief.get_param(self._scan_param).bounds
            self._acquisition_lo = min(slo, shi)
            self._acquisition_hi = max(slo, shi)

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        """Same physical interval as :meth:`_acquisition_bounds` for scan / manifest UI."""
        if self.initial_sweep_steps <= 0:
            return None
        lo, hi = self._acquisition_bounds()
        slo, shi = self._full_domain_lo, self._full_domain_hi
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

    # ------------------------------------------------------------------
    # Utility helpers available to all acquisition implementations
    # ------------------------------------------------------------------

    def _generate_candidates(self, num_candidates: int = 300) -> np.ndarray:
        """Generate grid from acquisition bounds (log-uniform for scale params)."""
        lo, hi = self._acquisition_bounds()
        is_scale = getattr(self.belief.model, "is_scale_parameter", lambda name: False)(self._scan_param)
        if is_scale and lo > 0 and hi > lo:
            return np.exp(np.linspace(np.log(lo), np.log(hi), num_candidates))
        return np.linspace(lo, hi, num_candidates)

    def _to_experiment_normalized(self, physical_value: float) -> float:
        """Map a physical scan position to ``[0, 1]`` for :meth:`CoreExperiment.measure`."""
        lo, hi = self._full_domain_lo, self._full_domain_hi
        return float(np.clip((physical_value - lo) / (hi - lo), 0.0, 1.0))

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
