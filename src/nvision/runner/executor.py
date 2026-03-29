"""Task executor — runs a LocatorTask end-to-end."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import polars as pl

from nvision.cache import CacheBridge
from nvision.models.experiment import CoreExperiment
from nvision.models.locator import Locator
from nvision.models.observer import Observer, RunResult
from nvision.models.task import LocatorTask
from nvision.runner.cache import embed_graph_content, restore_graphs
from nvision.runner.convert import run_result_to_finalize_record, run_result_to_history_df
from nvision.runner.metrics import generate_attempt_metrics
from nvision.runner.plots import generate_attempt_plots
from nvision.runner.repeat_keys import measurement_repeat_key, repeat_seed_int
from nvision.runner.signal_cache import get_shared_core_experiment
from nvision.sim.combinations import CombinationGrid
from nvision.tools.log_context import reset_combination_log_initials, set_combination_log_initials
from nvision.viz import Viz

log = logging.getLogger(__name__)

type RepeatResult = tuple[list[dict[str, Any]], dict[str, Any]]
type TaskResults = list[RepeatResult]


def run_loop(
    locator_class: type[Locator],
    experiment: CoreExperiment,
    rng: random.Random,
    **locator_config: Any,
) -> Iterator[Locator]:
    """Run one repeat's measurement loop and yield locator states."""
    locator = locator_class.create(**locator_config)
    while not locator.done():
        x_normalized = locator.next()
        obs = experiment.measure(x_normalized, rng)
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

    @property
    def viz(self) -> Viz:
        if self._viz is None:
            self._viz = Viz(self.task.out_dir / "graphs")
        return self._viz

    def _combination_cache_kwargs(self) -> dict[str, Any]:
        """Arguments for :meth:`LocatorResultsRepository.get_cached_combination` / save methods."""
        return {
            "generator": self.generator_name,
            "noise": self.noise_name,
            "strategy": self.strategy_name,
            "repeats": self.repeats,
            "seed": self.task.seed,
            "max_steps": self.task.loc_max_steps,
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
                return cached
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
                return cached
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
            self._save_repeat_cache(attempt_idx, entries, main_result_row)

        return all_results

    def _save_repeat_cache(self, repeat_idx: int, entries: list[dict[str, Any]], result_row: dict[str, Any]) -> None:
        if not self.task.use_cache or self.skip_cache:
            return
        self.cache.save_cached_repeat_slice(
            generator=self.generator_name,
            noise=self.noise_name,
            strategy=self.strategy_name,
            repeat=repeat_idx,
            seed=self.task.seed,
            max_steps=self.task.loc_max_steps,
            timeout_s=self.task.loc_timeout_s,
            entries=embed_graph_content(entries, self.task.out_dir),
            result_row=result_row,
        )

    def _save_full_cache(self, results: TaskResults) -> None:
        if not self.task.use_cache or self.skip_cache or not results:
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

    def _run_single_repeat(
        self,
        *,
        rid: int,
        locator_class: type[Locator],
        locator_config: dict[str, Any],
        rng: random.Random,
        experiment: CoreExperiment,
        repeat_start_time: float,
    ) -> tuple[pl.DataFrame, dict[str, Any], str, RunResult]:
        cfg = {**locator_config, "parameter_bounds": self._injected_parameter_bounds(experiment)}
        observer = Observer(experiment.true_signal, experiment.x_min, experiment.x_max)

        try:
            result = observer.watch(run_loop(locator_class, experiment, rng, **cfg))
            stop_reason = "locator_stop"
        except TimeoutError:
            result = RunResult(
                snapshots=observer.snapshots,
                true_signal=experiment.true_signal,
                focus_window=None,
            )
            stop_reason = "repeat_timeout"

        locator_instance = locator_class.create(**cfg)
        if result.snapshots:
            locator_instance.belief = result.snapshots[-1].belief
        locator_final_result = locator_instance.result()

        history_df = run_result_to_history_df(result, rid, experiment.x_min, experiment.x_max)
        finalize_record = run_result_to_finalize_record(
            result, locator_final_result, rid, experiment.x_min, experiment.x_max
        )
        # Used by the progress ETA estimator via cached `locator_results.csv` metadata.
        finalize_record["duration_ms"] = (time.perf_counter() - repeat_start_time) * 1000
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
                if "amplitude" in name.lower() and hi > noise_std:
                    lo = max(lo, noise_std)
                    if lo >= hi:
                        lo = hi * 0.999
                bounds[name] = (lo, hi)
        return bounds
