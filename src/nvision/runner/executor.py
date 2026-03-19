"""Task executor — runs a LocatorTask end-to-end."""

from __future__ import annotations

import hashlib
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
from nvision.sim.combinations import CombinationGrid
from nvision.viz import Viz

log = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 2

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


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    """Derived values used throughout task execution."""

    generator_name: str
    noise_name: str
    strategy_name: str
    repeats: int
    skip_cache: bool
    combo_cache_key: dict[str, Any]

    @classmethod
    def from_task(cls, task: LocatorTask) -> "_TaskSpec":
        """Construct the execution spec directly from a task."""
        return cls(
            generator_name=task.generator_name,
            noise_name=task.noise_name,
            strategy_name=task.strategy_name,
            repeats=task.repeats,
            skip_cache=task.ignore_cache_strategy is not None and task.strategy_name == task.ignore_cache_strategy,
            combo_cache_key={
                "kind": "locator_combination",
                "schema_version": CACHE_SCHEMA_VERSION,
                "generator": task.generator_name,
                "noise": task.noise_name,
                "strategy": task.strategy_name,
                "repeats": task.repeats,
                "seed": task.seed,
                "max_steps": task.loc_max_steps,
                "timeout_s": task.loc_timeout_s,
            },
        )


@dataclass(frozen=True, slots=True)
class _RepeatArtifacts:
    """Raw outputs produced by executing all repeats."""

    history_df: pl.DataFrame
    finalize_df: pl.DataFrame
    experiments: list[CoreExperiment]
    repeat_start_times: list[float]
    stop_reasons: list[str]


class _TaskRunner:
    """Run one task through four explicit phases.

    1) restore from cache when possible
    2) execute repeats
    3) build metrics/plots
    4) persist cache artifacts
    """

    def __init__(self, task: LocatorTask) -> None:
        self.task = task
        self.spec = _TaskSpec.from_task(task)
        category = CombinationGrid.generator_category(self.spec.generator_name)
        self.bridge = CacheBridge(task.cache_dir)
        self.cache = self.bridge.get_cache_for_category(category)
        self.viz = Viz(task.out_dir / "graphs")

    def run(self) -> TaskResults:
        """Main pipeline: cache -> repeats -> outputs -> cache."""
        try:
            cached = self._restore_cached_results()
            if cached is not None:
                return cached

            locator_class = self.task.strategy_spec.locator_class
            locator_config = dict(self.task.strategy_spec.locator_config)
            artifacts = self._run_repeats(locator_class, locator_config)
            results = self._build_repeat_outputs(artifacts)
            self._save_full_cache(results)
            return results
        finally:
            self.bridge.close()

    def _restore_cached_results(self) -> TaskResults | None:
        if self.task.require_cache:
            cached = self.cache.get_cached_results(self.spec.combo_cache_key)
            if cached:
                restore_graphs(cached, self.task.out_dir)
                log.debug(
                    "Cache hit for %s/%s/%s (seed=%s); restoring.",
                    self.spec.generator_name,
                    self.spec.noise_name,
                    self.spec.strategy_name,
                    self.task.seed,
                )
                return cached
            log.warning(
                "Cache miss for %s/%s/%s (seed=%s) with --require-cache. Skipping.",
                self.spec.generator_name,
                self.spec.noise_name,
                self.spec.strategy_name,
                self.task.seed,
            )
            return []

        if self.task.use_cache and not self.spec.skip_cache:
            cached = self.cache.get_cached_results(self.spec.combo_cache_key)
            if cached:
                restore_graphs(cached, self.task.out_dir)
                log.debug(
                    "Cache hit for %s/%s/%s (seed=%s); restoring.",
                    self.spec.generator_name,
                    self.spec.noise_name,
                    self.spec.strategy_name,
                    self.task.seed,
                )
                return cached
        return None

    def _run_repeats(self, locator_class: type[Locator], locator_config: dict[str, Any]) -> _RepeatArtifacts:
        """Execute all repeats and return raw repeat artifacts."""
        repeat_rngs: list[random.Random] = []
        experiments: list[CoreExperiment] = []
        repeat_start_times: list[float] = []
        stop_reasons: list[str] = [""] * self.spec.repeats

        for attempt_idx in range(self.spec.repeats):
            rng = self._rng_for_repeat(attempt_idx)
            repeat_rngs.append(rng)
            experiments.append(self._build_experiment(rng))
            repeat_start_times.append(time.perf_counter())

        history_dfs: list[pl.DataFrame] = []
        finalize_records: list[dict[str, Any]] = []

        for rid in range(self.spec.repeats):
            hist_df, finalize_record, stop_reason = self._run_single_repeat(
                rid=rid,
                locator_class=locator_class,
                locator_config=locator_config,
                rng=repeat_rngs[rid],
                experiment=experiments[rid],
            )
            stop_reasons[rid] = stop_reason
            if not hist_df.is_empty():
                history_dfs.append(hist_df)
            finalize_records.append(finalize_record)

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
        )

    def _build_repeat_outputs(self, artifacts: _RepeatArtifacts) -> TaskResults:
        """Generate metrics/plots and optional per-repeat cache entries."""
        all_results: TaskResults = []

        for attempt_idx in range(self.spec.repeats):
            entry_base, main_result_row, current_history_df = generate_attempt_metrics(
                n_repeats=self.spec.repeats,
                attempt_idx_in_combo=attempt_idx,
                gen_name=self.spec.generator_name,
                noise_name=self.spec.noise_name,
                strat_name=self.spec.strategy_name,
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
            )
            all_results.append((entries, main_result_row))
            self._save_repeat_cache(attempt_idx, entries, main_result_row)

            if self.task.progress_queue:
                self.task.progress_queue.put((self.task.task_id, 1))
            else:
                log.debug(
                    "Finished repeat %s for %s/%s/%s",
                    attempt_idx + 1,
                    self.spec.generator_name,
                    self.spec.noise_name,
                    self.spec.strategy_name,
                )

        return all_results

    def _save_repeat_cache(self, repeat_idx: int, entries: list[dict[str, Any]], result_row: dict[str, Any]) -> None:
        if not self.task.use_cache or self.spec.skip_cache:
            return
        part_cfg = {
            "kind": "locator_run",
            "schema_version": CACHE_SCHEMA_VERSION,
            "generator": self.spec.generator_name,
            "noise": self.spec.noise_name,
            "strategy": self.spec.strategy_name,
            "repeat": repeat_idx,
            "seed": self.task.seed,
            "max_steps": self.task.loc_max_steps,
            "timeout_s": self.task.loc_timeout_s,
        }
        self.cache.save_cached_repeat(part_cfg, embed_graph_content(entries, self.task.out_dir), result_row)

    def _save_full_cache(self, results: TaskResults) -> None:
        if not self.task.use_cache or self.spec.skip_cache or not results:
            return
        full_results = [(embed_graph_content(entries, self.task.out_dir), row) for entries, row in results]
        self.cache.save_cached_results(self.spec.combo_cache_key, full_results)

    def _rng_for_repeat(self, repeat_idx: int) -> random.Random:
        seed_str = (
            f"{self.task.seed}-{self.spec.generator_name}-{self.spec.strategy_name}-{self.spec.noise_name}-{repeat_idx}"
        )
        repeat_seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (10**8)
        return random.Random(repeat_seed)

    def _build_experiment(self, rng: random.Random) -> CoreExperiment:
        true_signal = self.task.generator.generate(rng)

        x_min = x_max = None
        for param in true_signal.parameters:
            if param.name == "x_min":
                x_min = param.value
            elif param.name == "x_max":
                x_max = param.value
            elif param.name == "frequency" and x_min is None:
                x_min, x_max = param.bounds

        if x_min is None or x_max is None:
            raise ValueError("TrueSignal must expose x_min/x_max parameters or frequency bounds")

        return CoreExperiment(true_signal=true_signal, noise=self.task.noise, x_min=x_min, x_max=x_max)

    def _run_single_repeat(
        self,
        *,
        rid: int,
        locator_class: type[Locator],
        locator_config: dict[str, Any],
        rng: random.Random,
        experiment: CoreExperiment,
    ) -> tuple[pl.DataFrame, dict[str, Any], str]:
        cfg = {**locator_config, "parameter_bounds": self._injected_parameter_bounds(experiment)}
        observer = Observer(experiment.true_signal, experiment.x_min, experiment.x_max)

        try:
            result = observer.watch(run_loop(locator_class, experiment, rng, **cfg))
            stop_reason = "locator_stop"
        except TimeoutError:
            result = RunResult(snapshots=observer.snapshots, true_signal=experiment.true_signal)
            stop_reason = "repeat_timeout"

        locator_instance = locator_class.create(**cfg)
        if result.snapshots:
            locator_instance.belief = result.snapshots[-1].belief
        locator_final_result = locator_instance.result()

        history_df = run_result_to_history_df(result, rid, experiment.x_min, experiment.x_max)
        finalize_record = run_result_to_finalize_record(
            result, locator_final_result, rid, experiment.x_min, experiment.x_max
        )
        return history_df, finalize_record, stop_reason

    @staticmethod
    def _injected_parameter_bounds(experiment: CoreExperiment) -> dict[str, tuple[float, float]]:
        """Extract valid `(lo, hi)` bounds from the true-signal parameters."""
        bounds: dict[str, tuple[float, float]] = {}
        for p in experiment.true_signal.parameters:
            if isinstance(p.bounds, tuple) and len(p.bounds) == 2:
                lo, hi = float(p.bounds[0]), float(p.bounds[1])
                if hi > lo:
                    bounds[p.name] = (lo, hi)
        return bounds


def run_task(task: LocatorTask) -> TaskResults:
    """Run a task (cache -> repeats -> outputs -> cache)."""
    spec = _TaskSpec.from_task(task)
    log.info(
        "Running task: %s/%s/%s (%s repeats)",
        spec.generator_name,
        spec.noise_name,
        spec.strategy_name,
        spec.repeats,
    )
    return _TaskRunner(task).run()
