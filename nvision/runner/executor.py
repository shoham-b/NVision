"""Task executor — runs a LocatorTask end-to-end."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.cache import CacheBridge
from nvision.models.experiment import CoreExperiment, Observation
from nvision.models.locator import Locator
from nvision.models.observer import Observer, RunResult
from nvision.models.task import LocatorTask
from nvision.runner.cache import embed_graph_content, restore_graphs, strip_heavy_fields
from nvision.runner.convert import run_result_to_finalize_record, run_result_to_history_df
from nvision.runner.metrics import generate_attempt_metrics
from nvision.runner.plots import generate_attempt_plots
from nvision.runner.repeat_keys import measurement_repeat_key, repeat_seed_int
from nvision.runner.signal_cache import get_shared_core_experiment
from nvision.sim import presets as sim_presets
from nvision.sim.combinations import CombinationGrid
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
from nvision.tools.log_context import reset_combination_log_initials, set_combination_log_initials
from nvision.viz import Viz

log = logging.getLogger(__name__)

# Process-level shared sweep cache (shared across all locators, all tasks)
_SWEEP_CACHE_LOCK = threading.Lock()
_SWEEP_OBSERVATIONS_BY_KEY: dict[str, list[Observation]] = {}


def _sweep_cache_key(experiment: CoreExperiment, sweep_steps: int) -> str:
    """Generate cache key from experiment characteristics.

    Key includes: x-range, sweep steps, noise config, and signal signature.
    """
    sig_bounds = getattr(experiment.true_signal, "bounds", {})
    noise_name = experiment.noise.__class__.__name__ if experiment.noise else "none"
    noise_seed = getattr(experiment.noise, "seed", "noseed") if experiment.noise else "noseed"
    # Include parameter values for signal instances with different true parameters
    param_values = getattr(experiment.true_signal, "parameter_values", lambda: {})()
    return (
        f"sweep:{experiment.x_min:.9f}:{experiment.x_max:.9f}:"
        f"{sweep_steps}:{noise_name}:{noise_seed}:"
        f"{hash(str(sorted(sig_bounds.items())))}:{hash(str(sorted(param_values.items())))}"
    )


def get_cached_sweep(experiment: CoreExperiment, sweep_steps: int) -> list[Observation] | None:
    """Retrieve cached sweep observations if available."""
    key = _sweep_cache_key(experiment, sweep_steps)
    with _SWEEP_CACHE_LOCK:
        return _SWEEP_OBSERVATIONS_BY_KEY.get(key)


def put_cached_sweep(experiment: CoreExperiment, sweep_steps: int, observations: list[Observation]) -> None:
    """Store sweep observations in shared cache."""
    key = _sweep_cache_key(experiment, sweep_steps)
    with _SWEEP_CACHE_LOCK:
        _SWEEP_OBSERVATIONS_BY_KEY[key] = observations


def has_cached_sweep(experiment: CoreExperiment, sweep_steps: int) -> bool:
    """Check if sweep is cached for this experiment."""
    key = _sweep_cache_key(experiment, sweep_steps)
    with _SWEEP_CACHE_LOCK:
        return key in _SWEEP_OBSERVATIONS_BY_KEY


def clear_sweep_cache() -> None:
    """Clear all cached sweep observations (useful for testing)."""
    with _SWEEP_CACHE_LOCK:
        _SWEEP_OBSERVATIONS_BY_KEY.clear()


def _create_sweep_belief(experiment: CoreExperiment) -> AbstractMarginalDistribution:
    """Create a minimal GridMarginalDistribution for sweeping locators.

    Sweeping locators need a belief to satisfy the Locator parent class,
    but they don't actually use it for sweep detection (they use signal_model).
    This creates a simple grid belief with the signal model's parameters.
    """
    model = experiment.true_signal.model
    param_names = model.parameter_names()

    # Create grid parameters for each model parameter
    parameters = []
    for name in param_names:
        # Use experiment bounds if available, otherwise default
        bounds = getattr(experiment.true_signal, "bounds", {}).get(name, (experiment.x_min, experiment.x_max))
        grid = np.linspace(bounds[0], bounds[1], 64)
        parameters.append(
            GridParameter(
                name=name,
                bounds=bounds,
                grid=grid,
                posterior=np.ones(64) / 64,
            )
        )

    return GridMarginalDistribution(model=model, parameters=parameters)


def precompute_sweep(
    locator_class: type[Locator],
    experiment: CoreExperiment,
    rng: random.Random,
    sweep_cache: SweepCache,
    **locator_config: Any,
) -> list[Observation] | None:
    """Pre-generate and cache sweep observations for Bayesian and sweep locators.

    This function creates a temporary locator, runs the initial sweep,
    and stores the observations in the shared sweep cache. Subsequent repeats
    can then use these cached observations without re-measuring.

    Returns the precomputed sweep observations, or None if the locator doesn't
    support sweep precomputation.
    """
    from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
    from nvision.sim.locs.coarse.sweep_locator import SweepingLocator

    # Only Bayesian locators and sweep locators need pre-computation
    is_bayesian = issubclass(locator_class, SequentialBayesianLocator)
    is_sweep = issubclass(locator_class, SweepingLocator)
    if not (is_bayesian or is_sweep):
        return None

    needs_belief = getattr(locator_class, "REQUIRES_BELIEF", False)
    if needs_belief and ("belief" not in locator_config or "signal_model" not in locator_config):
        locator_config.setdefault("belief", _create_sweep_belief(experiment))
        locator_config.setdefault("signal_model", experiment.true_signal.model)

    locator = locator_class.create(**locator_config)

    # For Bayesian: use initial_sweep_steps; for sweep: use max_steps (full sweep)
    sweep_steps = getattr(locator, "initial_sweep_steps", 0) if is_bayesian else getattr(locator, "max_steps", 0)

    if sweep_steps <= 0:
        return None

    # Check if already cached
    if sweep_cache.has(experiment, sweep_steps):
        return sweep_cache.get(experiment, sweep_steps)

    # Run the sweep phase
    observations: list[Observation] = []
    step = 0
    while not locator.done() and step < sweep_steps:
        step += 1
        x_physical = locator.next()
        # SweepingLocator returns physical x, but experiment.measure() expects normalized [0,1]
        x_normalized = experiment.normalize_x(x_physical)
        obs = experiment.measure(x_normalized, rng)
        locator.observe(obs)
        observations.append(obs)

    # Store in cache for subsequent repeats
    if observations:
        sweep_cache.put(experiment, sweep_steps, observations)

    return observations


@dataclass
class SweepCache:
    """Per-task sweep cache wrapper that uses the process-level shared cache.

    This allows each _TaskRunner to have its own cache reference while sharing
    the underlying storage across all locators in the process.
    """

    def get(self, experiment: CoreExperiment, sweep_steps: int) -> list[Observation] | None:
        return get_cached_sweep(experiment, sweep_steps)

    def put(self, experiment: CoreExperiment, sweep_steps: int, observations: list[Observation]) -> None:
        put_cached_sweep(experiment, sweep_steps, observations)

    def has(self, experiment: CoreExperiment, sweep_steps: int) -> bool:
        return has_cached_sweep(experiment, sweep_steps)


type RepeatResult = tuple[list[dict[str, Any]], dict[str, Any]]
type TaskResults = list[RepeatResult]


def run_loop(
    locator_class: type[Locator],
    experiment: CoreExperiment,
    rng: random.Random,
    sweep_cache: SweepCache | None = None,
    **locator_config: Any,
) -> Iterator[Locator]:
    """Run one repeat's measurement loop and yield locator states.

    For Bayesian locators with initial sweeps and sweep locators, checks
    ``sweep_cache`` for pre-computed observations to avoid redundant measurements.
    """
    from nvision.sim.locs.coarse.sweep_locator import SweepingLocator

    needs_belief = getattr(locator_class, "REQUIRES_BELIEF", False)
    if needs_belief and ("belief" not in locator_config or "signal_model" not in locator_config):
        locator_config.setdefault("belief", _create_sweep_belief(experiment))
        locator_config.setdefault("signal_model", experiment.true_signal.model)

    locator = locator_class.create(**locator_config)

    # Check if we can use cached sweep for Bayesian locators or sweep locators
    cached_sweep: list[Observation] | None = None
    if sweep_cache is not None:
        if isinstance(locator, SequentialBayesianLocator):
            sweep_steps = getattr(locator, "initial_sweep_steps", 0)
            if sweep_cache.has(experiment, sweep_steps):
                cached_sweep = sweep_cache.get(experiment, sweep_steps)
        elif isinstance(locator, SweepingLocator):
            sweep_steps = getattr(locator, "max_steps", 0)
            if sweep_cache.has(experiment, sweep_steps):
                cached_sweep = sweep_cache.get(experiment, sweep_steps)

    step = 0
    while not locator.done():
        step += 1
        x_current = locator.next()

        # Use cached observation if in sweep phase and cache available
        if cached_sweep is not None and step <= len(cached_sweep):
            cached_obs = cached_sweep[step - 1]
            # Create a new observation with the current x but cached signal value.
            # SweepingLocator subclasses (e.g. SobolSweep, StagedSobol) receive
            # physical x from next() but experiment.measure() always stores
            # normalized x.  Normalise here so the locator history stays consistent.
            cached_x = cached_obs.x
            if x_current < 0.0 or x_current > 1.0:
                cached_x = experiment.normalize_x(x_current)
            obs = Observation(
                x=cached_x,
                signal_value=cached_obs.signal_value,
                noise_std=cached_obs.noise_std,
                frequency_noise_model=cached_obs.frequency_noise_model,
            )
        else:
            # Some locators (SweepingLocator, StagedSobolSweepLocator) return physical x,
            # while others return normalized [0,1]. Normalize when x looks physical.
            if x_current < 0.0 or x_current > 1.0:
                x_norm = experiment.normalize_x(x_current)
                obs = experiment.measure(x_norm, rng)
            else:
                obs = experiment.measure(x_current, rng)

        locator.observe(obs)
        yield locator


def run_task(task: LocatorTask, *, cache_bridge: CacheBridge | None = None) -> TaskResults:
    """Run a task (cache -> repeats -> outputs -> cache).

    Pass a shared :class:`~nvision.cache.bridge.CacheBridge` from the CLI when running
    many tasks so SQLite is not opened and closed per task (large speedup on cache hits).
    """
    token = set_combination_log_initials(task.generator_name, task.noise_name, task.strategy_name)
    try:
        return _TaskRunner(task, cache_bridge=cache_bridge).run()
    finally:
        reset_combination_log_initials(token)


@dataclass(frozen=True, slots=True)
class _RepeatArtifacts:
    """Raw outputs produced by executing all repeats."""

    history_df: pl.DataFrame
    finalize_df: pl.DataFrame
    experiments: list[CoreExperiment]
    repeat_start_times: list[float]
    stop_reasons: list[str]
    run_results: list[RunResult]


class _TaskRunner:
    """Run one task through four explicit phases.

    1) restore from cache when possible
    2) execute repeats
    3) build metrics/plots
    4) persist cache artifacts
    """

    def __init__(self, task: LocatorTask, *, cache_bridge: CacheBridge | None = None) -> None:
        self.task = task
        self.generator_name = task.generator_name
        self.noise_name = task.noise_name
        self.strategy_name = task.strategy_name
        self.repeats = task.repeats
        self.skip_cache = task.ignore_cache_strategy is not None and task.strategy_name == task.ignore_cache_strategy
        category = CombinationGrid.generator_category(self.generator_name)
        if cache_bridge is not None:
            self.bridge = cache_bridge
            self._owns_bridge = False
        else:
            self.bridge = CacheBridge(task.cache_dir)
            self._owns_bridge = True
        self.cache = self.bridge.get_cache_for_category(category)
        self._viz: Viz | None = None
        # Shared sweep cache across all repeats for this task
        self._sweep_cache = SweepCache()

    @property
    def viz(self) -> Viz:
        if self._viz is None:
            self._viz = Viz(self.task.out_dir / "graphs")
        return self._viz

    def _combination_cache_kwargs(self) -> dict[str, Any]:
        """Arguments for :meth:`LocatorResultsRepository.get_cached_combination` / save methods."""
        # Use effective max_steps: CLI value overrides strategy default, but strategy
        # config may have its own default that differs from DEFAULT_LOC_MAX_STEPS.
        # We must match the actual value used in _run_single_repeat for cache hits.
        strategy_default_max_steps = self.task.strategy_spec.locator_config.get("max_steps")
        effective_max_steps = self.task.loc_max_steps
        if strategy_default_max_steps is not None and self.task.loc_max_steps == sim_presets.DEFAULT_LOC_MAX_STEPS:
            # User didn't override --loc-max-steps, use strategy's default for cache key
            effective_max_steps = strategy_default_max_steps
        return {
            "generator": self.generator_name,
            "noise": self.noise_name,
            "strategy": self.strategy_name,
            "repeats": self.repeats,
            "seed": self.task.seed,
            "max_steps": effective_max_steps,
            "timeout_s": self.task.loc_timeout_s,
        }

    def _flush_task_progress(self) -> None:
        """Mark this task fully done in the progress UI (cache hit or require-cache skip)."""
        pq = self.task.progress_queue
        tid = self.task.task_id
        if pq is not None and tid is not None:
            pq.put((tid, self.repeats))

    def _notify_repeat_finished(self, repeat_index: int) -> None:
        """Advance per-task repeat progress after simulation for one repeat completes."""
        pq = self.task.progress_queue
        tid = self.task.task_id
        if pq is not None and tid is not None:
            pq.put((tid, 1))
        else:
            log.debug(
                "Finished repeat %s for %s/%s/%s",
                repeat_index + 1,
                self.generator_name,
                self.noise_name,
                self.strategy_name,
            )

    def run(self) -> TaskResults:
        """Main pipeline: cache -> repeats -> outputs -> cache."""
        try:
            cached = self._restore_cached_results()
            if cached is not None:
                self._flush_task_progress()
                if cached:
                    log.info(
                        "Cache hit — skipped run: %s/%s/%s (%s repeats)",
                        self.generator_name,
                        self.noise_name,
                        self.strategy_name,
                        self.repeats,
                    )
                return cached

            log.debug(
                "Running task: %s/%s/%s (%s repeats)",
                self.generator_name,
                self.noise_name,
                self.strategy_name,
                self.repeats,
            )
            locator_class = self.task.strategy_spec.locator_class
            locator_config = dict(self.task.strategy_spec.locator_config)
            artifacts = self._run_repeats(locator_class, locator_config)
            results = self._build_repeat_outputs(artifacts)
            self._save_full_cache(results)
            return results
        finally:
            if self._owns_bridge:
                self.bridge.close()

    def _restore_cached_results(self) -> TaskResults | None:
        combo_kw = self._combination_cache_kwargs()
        if self.task.require_cache:
            cached = self.cache.get_cached_combination(**combo_kw)
            if cached:
                restore_graphs(cached, self.task.out_dir)
                log.debug(
                    "Cache hit for %s/%s/%s (seed=%s); restoring.",
                    self.generator_name,
                    self.noise_name,
                    self.strategy_name,
                    self.task.seed,
                )
                # Strip heavy fields to keep manifest small
                return [([strip_heavy_fields(e) for e in entries], row) for entries, row in cached]
            log.warning(
                "Cache miss for %s/%s/%s (seed=%s) with --require-cache. Skipping.",
                self.generator_name,
                self.noise_name,
                self.strategy_name,
                self.task.seed,
            )
            return []

        if self.task.use_cache and not self.skip_cache:
            cached = self.cache.get_cached_combination(**combo_kw)
            if cached:
                restore_graphs(cached, self.task.out_dir)
                log.debug(
                    "Cache hit for %s/%s/%s (seed=%s); restoring.",
                    self.generator_name,
                    self.noise_name,
                    self.strategy_name,
                    self.task.seed,
                )
                # Strip heavy fields to keep manifest small
                return [([strip_heavy_fields(e) for e in entries], row) for entries, row in cached]
        return None

    def _run_repeats(self, locator_class: type[Locator], locator_config: dict[str, Any]) -> _RepeatArtifacts:
        """Execute all repeats and return raw repeat artifacts."""
        repeat_rngs: list[random.Random] = []
        experiments: list[CoreExperiment] = []
        # Start times used for the `duration_ms` metadata stored in `locator_results.csv`.
        # These should align with when progress "advances" (i.e., right before each repeat begins).
        repeat_start_times: list[float] = [0.0] * self.repeats
        stop_reasons: list[str] = [""] * self.repeats

        for attempt_idx in range(self.repeats):
            measurement_rng = self._rng_for_measurement(attempt_idx)
            repeat_rngs.append(measurement_rng)
            experiments.append(get_shared_core_experiment(self.task, attempt_idx, self._build_experiment))

        # Pre-generate sweep for Bayesian locators so all repeats can share it
        # This is done once before any repeats run, ensuring the sweep is in cache
        if experiments:
            self._precompute_sweep_for_task(locator_class, locator_config, experiments[0], repeat_rngs[0])

        history_dfs: list[pl.DataFrame] = []
        finalize_records: list[dict[str, Any]] = []
        run_results: list[RunResult] = []

        for rid in range(self.repeats):
            repeat_start_times[rid] = time.perf_counter()
            hist_df, finalize_record, stop_reason, run_result = self._run_single_repeat(
                rid=rid,
                locator_class=locator_class,
                locator_config=locator_config,
                rng=repeat_rngs[rid],
                experiment=experiments[rid],
                repeat_start_time=repeat_start_times[rid],
            )
            stop_reasons[rid] = stop_reason
            run_results.append(run_result)
            if not hist_df.is_empty():
                history_dfs.append(hist_df)
            finalize_records.append(finalize_record)
            self._notify_repeat_finished(rid)

        empty_history = pl.DataFrame(
            {
                "repeat_id": pl.Series("repeat_id", [], dtype=pl.Int64),
                "step": pl.Series("step", [], dtype=pl.Int64),
                "x": pl.Series("x", [], dtype=pl.Float64),
                "signal_values": pl.Series("signal_values", [], dtype=pl.Float64),
            }
        )
        return _RepeatArtifacts(
            history_df=pl.concat(history_dfs) if history_dfs else empty_history,
            finalize_df=pl.DataFrame(finalize_records) if finalize_records else pl.DataFrame({"repeat_id": []}),
            experiments=experiments,
            repeat_start_times=repeat_start_times,
            stop_reasons=stop_reasons,
            run_results=run_results,
        )

    def _build_repeat_outputs(self, artifacts: _RepeatArtifacts) -> TaskResults:
        """Generate metrics/plots and optional per-repeat cache entries."""
        all_results: TaskResults = []

        for attempt_idx in range(self.repeats):
            entry_base, main_result_row, current_history_df = generate_attempt_metrics(
                n_repeats=self.repeats,
                attempt_idx_in_combo=attempt_idx,
                gen_name=self.generator_name,
                noise_name=self.noise_name,
                strat_name=self.strategy_name,
                repeat_stop_reasons=artifacts.stop_reasons,
                repeat_start_times=artifacts.repeat_start_times,
                current_scan=artifacts.experiments[attempt_idx],
                final_history_df=artifacts.history_df,
                finalize_results=artifacts.finalize_df,
                strat_obj=self.task.strategy,
            )

            entries = generate_attempt_plots(
                viz=self.viz,
                entry_base=entry_base,
                attempt_idx_in_combo=attempt_idx,
                current_scan=artifacts.experiments[attempt_idx],
                current_history_df=current_history_df,
                noise_obj=self.task.noise,
                strat_obj=self.task.strategy,
                slug_base=self.task.slug,
                out_dir=self.task.out_dir,
                scans_dir=self.task.scans_dir,
                bayes_dir=self.task.bayes_dir,
                run_result=artifacts.run_results[attempt_idx] if attempt_idx < len(artifacts.run_results) else None,
            )
            all_results.append((entries, main_result_row))

        return all_results

    def _save_full_cache(self, results: TaskResults) -> None:
        if self.skip_cache or not results:
            return
        full_results = [(embed_graph_content(entries, self.task.out_dir), row) for entries, row in results]
        self.cache.save_cached_combination(**self._combination_cache_kwargs(), results=full_results)

    def _rng_for_measurement(self, repeat_idx: int) -> random.Random:
        """RNG for measurement noise — still strategy-specific."""
        key = measurement_repeat_key(
            self.task.seed, self.generator_name, self.strategy_name, self.noise_name, repeat_idx
        )
        return random.Random(repeat_seed_int(key))

    def _build_experiment(self, rng: random.Random) -> CoreExperiment:
        true_signal = self.task.generator.generate(rng)
        x_min, x_max = self._domain_from_signal_params(true_signal)
        if x_min is None or x_max is None:
            x_min, x_max = self._domain_from_generator(self.task.generator)
        if x_min is None or x_max is None:
            raise ValueError("TrueSignal must expose x_min/x_max parameters or frequency bounds")

        return CoreExperiment(true_signal=true_signal, noise=self.task.noise, x_min=x_min, x_max=x_max)

    @staticmethod
    def _domain_from_signal_params(true_signal: Any) -> tuple[float | None, float | None]:
        """Infer scan domain from signal parameters."""
        x_min: float | None = None
        x_max: float | None = None
        freq_like_bounds: list[tuple[float, float]] = []

        values = true_signal.parameter_values()
        for name, value in values.items():
            if name == "x_min":
                x_min = float(value)
                continue
            if name == "x_max":
                x_max = float(value)
                continue
            try:
                lo, hi = true_signal.get_param_bounds(name)
            except KeyError:
                continue
            if hi > lo and "frequency" in name:
                freq_like_bounds.append((lo, hi))
                if name == "frequency" and x_min is None:
                    x_min, x_max = lo, hi

        if (x_min is None or x_max is None) and freq_like_bounds:
            x_min = min(lo for lo, _ in freq_like_bounds)
            x_max = max(hi for _, hi in freq_like_bounds)
        return x_min, x_max

    @staticmethod
    def _domain_from_generator(generator: Any) -> tuple[float | None, float | None]:
        """Fallback domain from generator attributes when present."""
        if not (hasattr(generator, "x_min") and hasattr(generator, "x_max")):
            return None, None
        try:
            x_min = float(generator.x_min)
            x_max = float(generator.x_max)
        except (TypeError, ValueError):
            return None, None
        return (x_min, x_max) if x_max > x_min else (None, None)

    def _resolve_sweep_max_steps(self, experiment: CoreExperiment) -> int:
        """Return the sweep step count for this experiment.

        If the user explicitly provided ``--sweep-max-steps``, that value is
        used directly.  Otherwise we compute the minimum number of uniformly
        spaced points required to resolve the narrowest expected dip in the
        signal model.
        """
        if self.task.sweep_max_steps is not None:
            return self.task.sweep_max_steps
        from nvision.sim.locs.coarse.sweep_steps import compute_sweep_max_steps

        return compute_sweep_max_steps(
            experiment.true_signal.model,
            float(experiment.x_min),
            float(experiment.x_max),
        )

    def _precompute_sweep_for_task(
        self,
        locator_class: type[Locator],
        locator_config: dict[str, Any],
        experiment: CoreExperiment,
        rng: random.Random,
    ) -> None:
        """Pre-generate and cache sweep observations for Bayesian locators.

        This ensures that when repeats are spawned, the sweep is already in cache
        and all repeats can share the same initial sweep measurements.
        """
        from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

        noise_std = 0.05
        noise_max_dev: float | None = None
        if experiment.noise is not None:
            noise_std = float(experiment.noise.estimated_noise_std())
            if hasattr(experiment.noise, "estimated_max_noise_deviation"):
                mid_n = SequentialBayesianLocator.DEFAULT_INITIAL_SWEEP_STEPS // 2
                noise_max_dev = float(experiment.noise.estimated_max_noise_deviation(n_samples=mid_n))

        domain_width = float(experiment.x_max - experiment.x_min)
        signal_min_span: float | None = None
        signal_max_span: float | None = None
        model = experiment.true_signal.model
        if hasattr(model, "signal_min_span") and callable(model.signal_min_span):
            signal_min_span = model.signal_min_span(domain_width)
        if hasattr(model, "signal_max_span") and callable(model.signal_max_span):
            signal_max_span = model.signal_max_span(domain_width)

        # Check locator class attributes to determine configuration
        uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)
        requires_belief = getattr(locator_class, "REQUIRES_BELIEF", False)
        max_steps = self._resolve_sweep_max_steps(experiment) if uses_sweep_max_steps else self.task.loc_max_steps

        cfg = {
            **locator_config,
            "max_steps": max_steps,
            "parameter_bounds": self._injected_parameter_bounds(experiment),
            "noise_std": noise_std,
            **({} if noise_max_dev is None else {"noise_max_dev": noise_max_dev}),
            **({} if signal_min_span is None else {"signal_min_span": signal_min_span}),
            **({} if signal_max_span is None else {"signal_max_span": signal_max_span}),
        }

        # For locators that require belief, add belief and signal_model
        if requires_belief:
            cfg["belief"] = _create_sweep_belief(experiment)
            cfg["signal_model"] = experiment.true_signal.model

        precompute_sweep(locator_class, experiment, rng, self._sweep_cache, **cfg)

    def _run_single_repeat(  # noqa: C901
        self,
        *,
        rid: int,
        locator_class: type[Locator],
        locator_config: dict[str, Any],
        rng: random.Random,
        experiment: CoreExperiment,
        repeat_start_time: float,
    ) -> tuple[pl.DataFrame, dict[str, Any], str, RunResult]:
        from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator

        noise_std = 0.05
        noise_max_dev: float | None = None
        if experiment.noise is not None:
            noise_std = float(experiment.noise.estimated_noise_std())
            if hasattr(experiment.noise, "estimated_max_noise_deviation"):
                # Use DEFAULT_INITIAL_SWEEP_STEPS // 2 as the n_samples count
                # so the threshold accounts for the actual mid-sweep sample size.
                mid_n = SequentialBayesianLocator.DEFAULT_INITIAL_SWEEP_STEPS // 2
                noise_max_dev = float(experiment.noise.estimated_max_noise_deviation(n_samples=mid_n))
        # Read signal spans from the model's declared methods.
        domain_width = float(experiment.x_max - experiment.x_min)
        signal_max_span: float | None = None
        model = experiment.true_signal.model
        if hasattr(model, "signal_min_span") and callable(model.signal_min_span):
            model.signal_min_span(domain_width)
        if hasattr(model, "signal_max_span") and callable(model.signal_max_span):
            signal_max_span = model.signal_max_span(domain_width)
        # Check locator class attributes to determine configuration
        uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)
        requires_belief = getattr(locator_class, "REQUIRES_BELIEF", False)
        max_steps = self._resolve_sweep_max_steps(experiment) if uses_sweep_max_steps else self.task.loc_max_steps

        cfg = {
            **locator_config,
            "max_steps": max_steps,
            "parameter_bounds": self._injected_parameter_bounds(experiment),
            "noise_std": noise_std,
            **({} if noise_max_dev is None else {"noise_max_dev": noise_max_dev}),
            **({} if signal_max_span is None else {"signal_max_span": signal_max_span}),
        }

        # For locators that require belief, add belief and signal_model
        if requires_belief:
            belief = _create_sweep_belief(experiment)
            cfg["belief"] = belief
            cfg["signal_model"] = experiment.true_signal.model

        observer = Observer(experiment.true_signal, experiment.x_min, experiment.x_max)

        try:
            result = observer.watch(run_loop(locator_class, experiment, rng, self._sweep_cache, **cfg))
            stop_reason = "locator_stop"
        except TimeoutError:
            result = RunResult(
                snapshots=observer.snapshots,
                true_signal=experiment.true_signal,
                focus_window=None,
            )
            stop_reason = "repeat_timeout"

        # Populate sweep cache from this repeat's observations (for sharing with subsequent repeats)
        if observer.last_locator is not None:
            last_loc = observer.last_locator
            if isinstance(last_loc, SequentialBayesianLocator):
                # Use initial_sweep_steps (calculated) for cache key consistency with lookup
                sweep_steps = getattr(last_loc, "initial_sweep_steps", 0)
                # _sweep_observations is on the staged_sobol, not the Bayesian locator itself
                staged_sobol = getattr(last_loc, "_staged_sobol", None)
                sweep_obs = getattr(staged_sobol, "_sweep_observations", []) if staged_sobol is not None else []
                if sweep_steps > 0 and sweep_obs and not self._sweep_cache.has(experiment, sweep_steps):
                    self._sweep_cache.put(experiment, sweep_steps, list(sweep_obs))

        last_loc = observer.last_locator
        if last_loc is not None:
            locator_instance = last_loc
        else:
            locator_instance = locator_class.create(**cfg)
            if result.snapshots:
                locator_instance.belief = result.snapshots[-1].belief
        # Provide the ground-truth signal so sweep locators can compute the
        # actual dip width (not the model worst-case minimum) for the
        # expected-uniform baseline.
        if hasattr(locator_instance, "_true_signal"):
            locator_instance._true_signal = experiment.true_signal
        locator_final_result = locator_instance.result()

        history_df = run_result_to_history_df(result, rid, experiment.x_min, experiment.x_max)
        finalize_record = run_result_to_finalize_record(
            result, locator_final_result, rid, experiment.x_min, experiment.x_max
        )
        # Used by the progress ETA estimator via cached `locator_results.csv` metadata.
        finalize_record["duration_ms"] = (time.perf_counter() - repeat_start_time) * 1000
        last_loc = observer.last_locator
        if last_loc is not None:
            # Use effective_initial_sweep_steps() to get actual steps taken (accounts for early stopping)
            eff_sweep_steps = getattr(last_loc, "effective_initial_sweep_steps", lambda: 0)()
            init_sweep_steps = getattr(last_loc, "initial_sweep_steps", 0)
            step_count = getattr(last_loc, "step_count", 0)
            inf_steps = getattr(last_loc, "inference_step_count", 0)
            max_steps = getattr(last_loc, "max_steps", 0)
            log = logging.getLogger("nvision")
            log.info(
                f"[STEP DEBUG] rid={rid} init_sweep={init_sweep_steps} eff_sweep={eff_sweep_steps} "
                f"step_count={step_count} inf_steps={inf_steps} max_steps={max_steps}"
            )
            # For sweep-only runs (no inference phase), all steps are sweep steps.
            if inf_steps == 0 and step_count > (eff_sweep_steps or 0):
                eff_sweep_steps = step_count
            finalize_record["sweep_steps"] = int(eff_sweep_steps or 0)
            finalize_record["locator_steps"] = int(inf_steps or 0)
        return history_df, finalize_record, stop_reason, result

    @staticmethod
    def _injected_parameter_bounds(experiment: CoreExperiment) -> dict[str, tuple[float, float]]:
        """Extract valid `(lo, hi)` bounds with noise-aware amplitude floor.

        Any amplitude-like parameter is clamped so its lower bound is at least
        the estimated measurement-noise standard deviation (when feasible).
        """
        bounds: dict[str, tuple[float, float]] = {}
        noise_std = 0.05
        if experiment.noise is not None:
            noise_std = float(experiment.noise.estimated_noise_std())
        for name, (lo_raw, hi_raw) in experiment.true_signal.bounds.items():
            lo, hi = float(lo_raw), float(hi_raw)
            if hi > lo:
                name_lc = name.lower()
                if ("amplitude" in name_lc or "depth" in name_lc) and hi > noise_std:
                    lo = max(lo, noise_std)
                    if lo >= hi:
                        lo = hi * 0.999
                bounds[name] = (lo, hi)
        return bounds
