"""Task executor — runs a LocatorTask end-to-end."""

from __future__ import annotations

import dataclasses
import datetime
import logging
import random
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import polars as pl

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
from nvision.runner.sweep_cache import (
    SweepCache,
    _create_sweep_belief,
    precompute_sweep,
)
from nvision.sim.combinations import CombinationGrid
from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
from nvision.sim.locs.coarse import SweepingLocator
from nvision.tools.log_context import reset_combination_log_initials, set_combination_log_initials
from nvision.viz import Viz

log = logging.getLogger(__name__)

MAX_PROCESS_MEMORY_GIB = 1.0


def _check_memory_limit() -> None:
    """Check current process memory usage and raise MemoryError if it exceeds limit."""
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        if mem_bytes > MAX_PROCESS_MEMORY_GIB * 1024 * 1024 * 1024:
            log.error(
                "Process %s memory limit exceeded: %.2f GiB > %.2f GiB",
                os.getpid(),
                mem_bytes / (1024**3),
                MAX_PROCESS_MEMORY_GIB,
            )
            raise MemoryError(
                f"Memory limit of {MAX_PROCESS_MEMORY_GIB} GiB exceeded (current: {mem_bytes / (1024**3):.2f} GiB)"
            )
    except ImportError:
        pass


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

    For Bayesian locators with initial sweeps and  locators, checks
    ``sweep_cache`` for pre-computed observations to avoid redundant measurements.
    """
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
        _check_memory_limit()
        step += 1
        x_current = locator.next()

        # Use cached observation if in sweep phase and cache available
        if cached_sweep is not None and step <= len(cached_sweep):
            cached_obs = cached_sweep[step - 1]
            obs = Observation(
                x=x_current,
                signal_value=cached_obs.signal_value,
                noise_std=cached_obs.noise_std,
                frequency_noise_model=cached_obs.frequency_noise_model,
            )
        else:
            obs = experiment.measure(x_current, rng)

        locator.observe(obs)
        yield locator


def run_task(
    task: LocatorTask,
    cache_bridge: CacheBridge | None = None,
    log_level: int = logging.INFO,
    shm_name: str | None = None,
    shm_lock: Any | None = None,
    sweep_shm_name: str | None = None,
    sweep_shm_lock: Any | None = None,
) -> tuple[list[dict[str, object]], list[dict]]:
    """Run a single combination task and return results."""
    from nvision.runner.signal_cache import initialize_shared_cache
    from nvision.runner.sweep_cache import initialize_shared_sweep_cache

    # Ensure this process is attached to the shared cache
    initialize_shared_cache(shm_name, shm_lock)
    initialize_shared_sweep_cache(sweep_shm_name, sweep_shm_lock)

    _check_memory_limit()
    token = set_combination_log_initials(task.generator_name, task.noise_name, task.strategy_name, task.seed)
    try:
        runner = _TaskRunner(task, cache_bridge=cache_bridge)
        return runner.run()
    finally:
        reset_combination_log_initials(token)


@dataclass(frozen=True, slots=True)
class _RepeatArtifacts:
    """Raw outputs produced by executing all repeats."""

    history_df: pl.DataFrame
    finalize_df: pl.DataFrame
    experiments: list[CoreExperiment]
    repeat_start_times: list[float]
    repeat_timestamps: list[str]
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
        self.skip_cache = (not task.use_cache) or (
            task.ignore_cache_strategy is not None and task.strategy_name == task.ignore_cache_strategy
        )
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
        # Background saver for streaming repeats
        self._saver_pool = ThreadPoolExecutor(max_workers=1)

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
        # Replicate the logic used to determine `max_steps` during `_run_single_repeat`.
        # Sweep locators use a dynamic step count if --sweep-max-steps is not set.
        locator_class = self.task.strategy_spec.locator_class
        uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)

        if uses_sweep_max_steps:
            # Reconstruct the experiment to compute the dynamic step count
            experiment = self._build_experiment(self._rng_for_measurement(0))
            effective_max_steps = self._resolve_sweep_max_steps(experiment)
        else:
            effective_max_steps = self.task.loc_max_steps
        return {
            "generator": self.generator_name,
            "noise": self.noise_name,
            "strategy": self.strategy_name,
            "repeats": self.task.repeat_total or self.repeats,
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

    def run(self) -> TaskResults:  # noqa: C901
        """Main pipeline: cache -> repeats -> outputs -> cache."""
        _check_memory_limit()
        try:
            cached, n_cached = self._restore_cached_results()

            # Resolve the total requested and total achieved in cache across the entire task (total level)
            total_requested = self.task.repeat_total or self.repeats
            total_achieved_in_cache = 0

            if not self.skip_cache:
                locator_class = self.task.strategy_spec.locator_class
                uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)
                if uses_sweep_max_steps:
                    try:
                        experiment = self._build_experiment(self._rng_for_measurement(0))
                        effective_max_steps = self._resolve_sweep_max_steps(experiment)
                    except Exception:
                        effective_max_steps = self.task.loc_max_steps
                else:
                    effective_max_steps = self.task.loc_max_steps

                from nvision.cache.hashing import stable_config_hash
                from nvision.cache.locator_keys import combination_base_cache_config

                ptr_config = combination_base_cache_config(
                    generator=self.generator_name,
                    noise=self.noise_name,
                    strategy=self.strategy_name,
                    seed=self.task.seed,
                    max_steps=effective_max_steps,
                    timeout_s=self.task.loc_timeout_s,
                    repeat_offset=0,  # Pointer totals are always at offset 0
                )
                ptr_key = stable_config_hash(ptr_config)
                try:
                    ptr_df = self.cache._store.load_df(ptr_key)
                    if ptr_df is None or ptr_df.is_empty():
                        ptr_config_v8 = dict(ptr_config)
                        ptr_config_v8["schema_version"] = 8
                        ptr_key_v8 = stable_config_hash(ptr_config_v8)
                        ptr_df_v8 = self.cache._store.load_df(ptr_key_v8)
                        if ptr_df_v8 is not None and not ptr_df_v8.is_empty():
                            ptr_key = ptr_key_v8
                            ptr_df = ptr_df_v8
                    if ptr_df is not None and not ptr_df.is_empty():
                        total_achieved_in_cache = self.cache._repeats.count_saved(ptr_key, max_expected=total_requested)
                except Exception:
                    pass

            if self.task.repeat_offset == 0:
                if total_achieved_in_cache >= total_requested:
                    log.info(
                        "Cache hit — skipped run: %s/%s/%s (%s repeats)",
                        self.generator_name,
                        self.noise_name,
                        self.strategy_name,
                        total_requested,
                    )
                else:
                    log.info(
                        "Running task: %s/%s/%s (%s total repeats, %s loaded from cache)",
                        self.generator_name,
                        self.noise_name,
                        self.strategy_name,
                        total_requested,
                        total_achieved_in_cache,
                    )
                    self._explain_cache_miss_total(total_achieved_in_cache, total_requested)

            if n_cached == self.repeats:
                self._flush_task_progress()
                return cached

            # Inform progress bar about cached repeats
            if n_cached > 0:
                pq = self.task.progress_queue
                tid = self.task.task_id
                if pq is not None and tid is not None:
                    pq.put((tid, n_cached))

            locator_class = self.task.strategy_spec.locator_class
            locator_config = dict(self.task.strategy_spec.locator_config)

            artifacts = self._run_repeats(locator_class, locator_config, start_idx=n_cached)
            if isinstance(artifacts, list):
                # Streaming case: artifacts are already TaskResults (lightweight)
                new_results = artifacts
            else:
                # Inline case: artifacts are raw _RepeatArtifacts
                new_results = self._build_repeat_outputs(artifacts, start_idx=n_cached)

            # Combine old and new results
            full_results = cached + new_results

            # Final save (handles both inline full save and streaming pointer updates)
            self._save_full_cache(full_results, n_cached=n_cached)
            return full_results
        finally:
            self._saver_pool.shutdown(wait=True)
            if self._owns_bridge:
                self.bridge.close()

    def _purge_cache_if_needed(self) -> None:
        """Purge the entire cached combination once across all partitions if skip_cache is active."""
        if not self.skip_cache:
            return

        import polars as pl

        from nvision.cache.hashing import stable_config_hash
        from nvision.cache.locator_keys import combination_base_cache_config

        # Resolve effective max_steps
        locator_class = self.task.strategy_spec.locator_class
        uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)
        if uses_sweep_max_steps:
            experiment = self._build_experiment(self._rng_for_measurement(0))
            effective_max_steps = self._resolve_sweep_max_steps(experiment)
        else:
            effective_max_steps = self.task.loc_max_steps

        # Use a standardized purged flag key independent of repeat_offset
        base_cfg = combination_base_cache_config(
            generator=self.generator_name,
            noise=self.noise_name,
            strategy=self.strategy_name,
            seed=self.task.seed,
            max_steps=effective_max_steps,
            timeout_s=self.task.loc_timeout_s,
            repeat_offset=0,
        )
        purged_key = f"purged:{stable_config_hash(base_cfg)}"

        if purged_key not in self.cache._store.backend:
            # Purge all partitions of this combination
            combo_kw = self._combination_cache_kwargs()
            self.cache.purge_cached_combination(**combo_kw, repeat_offset=self.task.repeat_offset)
            # Write standardized purged flag to cache DB so other parallel workers don't purge again
            self.cache._store.save_df(pl.DataFrame({"purged": [True]}), purged_key)

    def _restore_cached_results(self, allow_gaps: bool = False) -> tuple[TaskResults, int]:
        """Load results from cache. Handles full matches and partial streaming hits."""
        if self.skip_cache:
            if self.task.require_cache:
                log.warning(
                    "Cache miss for %s/%s/%s (seed=%s) with --require-cache + caching disabled. Skipping.",
                    self.generator_name,
                    self.noise_name,
                    self.strategy_name,
                    self.task.seed,
                )
                return [], self.repeats  # "Pretend" finished so it returns empty
            return [], 0

        combo_kw = self._combination_cache_kwargs()
        ro = self.task.repeat_offset

        # 1. Try exact full match (Inline or already complete Streaming)
        cached = self.cache.get_cached_combination(**combo_kw, repeat_offset=ro, allow_gaps=allow_gaps)
        if cached:
            restore_graphs(cached, self.task.out_dir)
            log.debug(
                "Full cache hit for %s/%s/%s (seed=%s); restoring.",
                self.generator_name,
                self.noise_name,
                self.strategy_name,
                self.task.seed,
            )
            # Strip heavy fields to keep manifest small
            results = [([strip_heavy_fields(e) for e in entries], row) for entries, row in cached]
            return results, len(results)

        # 2. Try partial streaming match (only if not using --require-cache, or if miss)
        # Pass chunk_size=self.repeats so sub-tasks only claim their own slice of cache
        partial, n = self.cache.get_cached_combination_partial(
            **combo_kw, repeat_offset=ro, chunk_size=self.repeats, allow_gaps=allow_gaps
        )
        if n > 0:
            restore_graphs(partial, self.task.out_dir)
            log.debug(
                "Partial cache hit for %s/%s/%s (seed=%s); restoring %s/%s repeats.",
                self.generator_name,
                self.noise_name,
                self.strategy_name,
                self.task.seed,
                n,
                self.repeats,
            )
            results = [([strip_heavy_fields(e) for e in entries], row) for entries, row in partial]
            return results, n

        if self.task.require_cache:
            log.warning(
                "Cache miss for %s/%s/%s (seed=%s) with --require-cache. Skipping.",
                self.generator_name,
                self.noise_name,
                self.strategy_name,
                self.task.seed,
            )
            return [], self.repeats  # "Pretend" finished so it returns empty

        return [], 0

    def _explain_cache_miss_total(self, total_achieved: int, total_requested: int) -> None:  # noqa: C901
        """Explain why there was a cache miss or partial cache hit at the global total level."""
        if self.skip_cache:
            log.info("Cache miss reason: Caching was bypassed/disabled via task config (skip_cache=True).")
            return

        if total_achieved > 0:
            log.info(
                "Cache miss reason: Partial cache hit. Only %s repeats exist in cache of the %s requested. "
                "The remaining %s repeats must be run.",
                total_achieved,
                total_requested,
                total_requested - total_achieved,
            )
            return

        self._combination_cache_kwargs()

        # Resolve effective max_steps
        locator_class = self.task.strategy_spec.locator_class
        uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)
        if uses_sweep_max_steps:
            try:
                experiment = self._build_experiment(self._rng_for_measurement(0))
                effective_max_steps = self._resolve_sweep_max_steps(experiment)
            except Exception:
                effective_max_steps = self.task.loc_max_steps
        else:
            effective_max_steps = self.task.loc_max_steps

        # Exact pointer doesn't exist. Let's look for similar combinations in the database
        similar_configs = []
        try:
            from nvision.cache.locator_keys import CACHE_SCHEMA_VERSION

            backend = self.cache.backend
            for k in backend:
                payload = backend.get(k)
                if isinstance(payload, dict) and "config" in payload:
                    config = payload["config"]
                    if (
                        config.get("generator") == self.generator_name
                        and config.get("noise") == self.noise_name
                        and config.get("strategy") == self.strategy_name
                    ):
                        similar_configs.append(config)
        except Exception as e:
            log.debug("Failed to scan cache for similar configs: %s", e)

        if not similar_configs:
            log.info(
                "Cache miss reason: No prior cache entries exist for combination: %s/%s/%s (first run).",
                self.generator_name,
                self.noise_name,
                self.strategy_name,
            )
            return

        # We have similar configs, let's explain the mismatch
        mismatch_reasons = []
        seen_seeds = set()
        seen_max_steps = set()
        seen_timeouts = set()
        seen_schemas = set()

        for cfg in similar_configs:
            if cfg.get("seed") != self.task.seed:
                seen_seeds.add(cfg.get("seed"))
            if cfg.get("max_steps") != effective_max_steps:
                seen_max_steps.add(cfg.get("max_steps"))
            if cfg.get("timeout_s") != self.task.loc_timeout_s:
                seen_timeouts.add(cfg.get("timeout_s"))
            if cfg.get("schema_version") not in (CACHE_SCHEMA_VERSION, 8):
                seen_schemas.add(cfg.get("schema_version"))

        if seen_seeds:
            mismatch_reasons.append(f"seed mismatch (target: {self.task.seed}, cached: {list(seen_seeds)})")
        if seen_max_steps:
            mismatch_reasons.append(
                f"max_steps mismatch (target: {effective_max_steps}, cached: {list(seen_max_steps)})"
            )
        if seen_timeouts:
            mismatch_reasons.append(
                f"timeout_s mismatch (target: {self.task.loc_timeout_s}, cached: {list(seen_timeouts)})"
            )
        if seen_schemas:
            mismatch_reasons.append(
                f"schema_version mismatch (target: {CACHE_SCHEMA_VERSION}, cached: {list(seen_schemas)})"
            )

        if mismatch_reasons:
            log.info(
                "Cache miss reason: Prior runs found for %s/%s/%s, but parameters differed: %s",
                self.generator_name,
                self.noise_name,
                self.strategy_name,
                ", ".join(mismatch_reasons),
            )
        else:
            log.info(
                "Cache miss reason: Prior runs found for %s/%s/%s, but no matching repeats were found.",
                self.generator_name,
                self.noise_name,
                self.strategy_name,
            )

    def _run_repeats(
        self, locator_class: type[Locator], locator_config: dict[str, Any], start_idx: int = 0
    ) -> _RepeatArtifacts | TaskResults:
        """Execute all repeats (or missing ones).

        Returns raw artifacts for small runs, or processed TaskResults for streaming runs
        to save memory (history is cleared after saving).
        """
        n_missing = self.repeats - start_idx
        if n_missing <= 0:
            return (
                []
                if self.repeats > 5
                else _RepeatArtifacts(
                    history_df=pl.DataFrame({"repeat_id": []}),
                    finalize_df=pl.DataFrame({"repeat_id": []}),
                    experiments=[],
                    repeat_start_times=[],
                    repeat_timestamps=[],
                    stop_reasons=[],
                    run_results=[],
                )
            )

        repeat_rngs: list[random.Random] = []
        experiments: list[CoreExperiment] = []
        repeat_start_times: list[float] = [0.0] * n_missing
        repeat_timestamps: list[str] = [""] * n_missing
        stop_reasons: list[str] = [""] * n_missing

        offset = self.task.repeat_offset
        for i in range(n_missing):
            rid = offset + start_idx + i
            measurement_rng = self._rng_for_measurement(rid)
            repeat_rngs.append(measurement_rng)
            experiments.append(get_shared_core_experiment(self.task, rid, self._build_experiment))

        # Pre-generate sweep for Bayesian locators
        if experiments:
            self._precompute_sweep_for_task(locator_class, locator_config, experiments[0], repeat_rngs[0])

        history_dfs: list[pl.DataFrame] = []
        finalize_records: list[dict[str, Any]] = []
        run_results: list[RunResult] = []
        streaming_results: TaskResults = []

        from nvision.cache.locator_repository import STREAMING_REPEAT_THRESHOLD

        total_repeats = self.task.repeat_total or self.repeats
        is_streaming = total_repeats > STREAMING_REPEAT_THRESHOLD

        # Determine effective max_steps for metrics/manifest keys
        locator_class = self.task.strategy_spec.locator_class
        uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)
        if uses_sweep_max_steps and experiments:
            self._resolve_sweep_max_steps(experiments[0])
        else:
            pass

        for i in range(n_missing):
            rid = offset + start_idx + i
            repeat_start_times[i] = time.perf_counter()
            repeat_timestamps[i] = datetime.datetime.now(datetime.UTC).isoformat()
            hist_df, finalize_record, stop_reason, run_result = self._run_single_repeat(
                rid=rid,
                locator_class=locator_class,
                locator_config=locator_config,
                rng=repeat_rngs[i],
                experiment=experiments[i],
                repeat_start_time=repeat_start_times[i],
            )
            stop_reasons[i] = stop_reason
            run_results.append(run_result)
            finalize_records.append(finalize_record)

            if self.skip_cache:
                self._purge_cache_if_needed()

            # Incremental streaming save for long runs
            if is_streaming:
                # Generate plots and metrics for JUST this repeat
                artifacts_single = _RepeatArtifacts(
                    history_df=hist_df,
                    finalize_df=pl.DataFrame([finalize_record]),
                    experiments=[experiments[i]],
                    repeat_start_times=[repeat_start_times[i]],
                    repeat_timestamps=[repeat_timestamps[i]],
                    stop_reasons=[stop_reason],
                    run_results=[run_result],
                )
                res_single = self._build_repeat_outputs(artifacts_single, start_idx=rid)
                entries, main_result_row = res_single[0]

                # Strip heavy fields for the in-memory results list
                light_entries = [strip_heavy_fields(e) for e in entries]
                streaming_results.append((light_entries, main_result_row))

                # Background save the full version (with base64 plots)
                entries_embedded = embed_graph_content(entries, self.task.out_dir)
                self._saver_pool.submit(
                    self._background_save_repeat,
                    rid=rid,
                    entries=entries_embedded,
                    main_result_row=main_result_row,
                )

                # FREE MEMORY: hist_df is no longer needed locally
                del hist_df
            else:
                if not hist_df.is_empty():
                    history_dfs.append(hist_df)

            self._notify_repeat_finished(rid)

        if is_streaming:
            return streaming_results

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
            finalize_df=pl.from_dicts(finalize_records, infer_schema_length=None)
            if finalize_records
            else pl.DataFrame({"repeat_id": []}),
            experiments=experiments,
            repeat_start_times=repeat_start_times,
            repeat_timestamps=repeat_timestamps,
            stop_reasons=stop_reasons,
            run_results=run_results,
        )

    def _background_save_repeat(self, rid: int, entries: list[dict[str, Any]], main_result_row: dict[str, Any]) -> None:
        """Worker function for background saving."""
        try:
            # save_repeat and append_cached_repeats do not accept 'repeats' keyword
            combo_kw = {k: v for k, v in self._combination_cache_kwargs().items() if k != "repeats"}
            self.cache.save_repeat(
                **combo_kw,
                repeat_offset=0,
                repeat_idx=rid,
                entries=entries,
                main_result_row=main_result_row,
            )
            # Update pointer
            self.cache.append_cached_repeats(
                **combo_kw,
                repeat_offset=0,
                new_results=[],
                start_idx=rid + 1,
            )
        except Exception:
            log.error("Failed to save repeat %s to cache in background", rid, exc_info=True)

    def _build_repeat_outputs(self, artifacts: _RepeatArtifacts, start_idx: int = 0) -> TaskResults:
        """Generate metrics/plots and optional per-repeat cache entries."""
        all_results: TaskResults = []
        n_repeats = len(artifacts.experiments)

        # Determine effective max_steps for metrics/manifest keys
        locator_class = self.task.strategy_spec.locator_class
        uses_sweep_max_steps = getattr(locator_class, "USES_SWEEP_MAX_STEPS", False)
        if uses_sweep_max_steps and artifacts.experiments:
            effective_max_steps = self._resolve_sweep_max_steps(artifacts.experiments[0])
        else:
            effective_max_steps = self.task.loc_max_steps

        # Pad list inputs for generate_attempt_metrics to align with attempt_idx (0 to self.repeats - 1)
        full_stop_reasons = [""] * start_idx + list(artifacts.stop_reasons)
        full_start_times = [0.0] * start_idx + list(artifacts.repeat_start_times)
        full_timestamps = [""] * start_idx + list(artifacts.repeat_timestamps)

        for i in range(n_repeats):
            attempt_idx = start_idx + i
            entry_base, main_result_row, current_history_df = generate_attempt_metrics(
                n_repeats=self.task.repeat_total or self.repeats,
                attempt_idx_in_combo=attempt_idx,
                gen_name=self.generator_name,
                noise_name=self.noise_name,
                strat_name=self.strategy_name,
                repeat_stop_reasons=full_stop_reasons,
                repeat_start_times=full_start_times,
                repeat_timestamps=full_timestamps,
                current_scan=artifacts.experiments[i],
                final_history_df=artifacts.history_df,
                finalize_results=artifacts.finalize_df,
                strat_obj=self.task.strategy,
                max_steps=effective_max_steps,
                seed=self.task.seed,
                run_result=artifacts.run_results[i] if i < len(artifacts.run_results) else None,
            )

            entries = generate_attempt_plots(
                viz=self.viz,
                entry_base=entry_base,
                attempt_idx_in_combo=attempt_idx,
                current_scan=artifacts.experiments[i],
                current_history_df=current_history_df,
                noise_obj=self.task.noise,
                strat_obj=self.task.strategy,
                slug_base=self.task.slug,
                out_dir=self.task.out_dir,
                scans_dir=self.task.scans_dir,
                bayes_dir=self.task.bayes_dir,
                run_result=artifacts.run_results[i] if i < len(artifacts.run_results) else None,
            )
            all_results.append((entries, main_result_row))

        return all_results

    def _save_full_cache(self, results: TaskResults, n_cached: int = 0) -> None:
        """Persist results to cache.

        Parameters
        ----------
        n_cached:
            Number of results that were already in cache (and thus already
            saved at the correct indices).  Only the *new* results (starting
            at index ``repeat_offset + n_cached``) need to be written.
        """
        if not results:
            return

        from nvision.cache.locator_repository import STREAMING_REPEAT_THRESHOLD

        ro = self.task.repeat_offset
        # The global index of the first NEW result
        first_new_idx = ro + n_cached

        if self.repeats > STREAMING_REPEAT_THRESHOLD:
            # Streaming: each repeat is already saved by _background_save_repeat.
            # Call save_cached_combination to ensure the pointer reflects the final
            # count (in case any background save raced).  Pass only the NEW results
            # so we don't overwrite already-saved entries at wrong indices.
            new_results = results[n_cached:]
            if new_results:
                final_idx = first_new_idx + len(new_results)
                self.cache.save_cached_combination(
                    **self._combination_cache_kwargs(),
                    repeat_offset=ro,
                    results=[],  # Do not overwrite repeats, just update the pointer
                    start_idx=final_idx,
                )
            return

        full_results = [(embed_graph_content(entries, self.task.out_dir), row) for entries, row in results]
        self.cache.save_cached_combination(
            **self._combination_cache_kwargs(),
            repeat_offset=ro,
            results=full_results,
            start_idx=ro,
        )

    def _rng_for_measurement(self, repeat_idx: int) -> random.Random:
        """RNG for measurement noise — still strategy-specific."""
        key = measurement_repeat_key(
            self.task.seed, self.generator_name, self.strategy_name, self.noise_name, repeat_idx
        )
        return random.Random(repeat_seed_int(key))

    def _rng_for_sobol_baseline(self, repeat_idx: int) -> random.Random:
        """RNG for Sobol baseline measurement noise — strategy-independent."""
        key = measurement_repeat_key(self.task.seed, self.generator_name, "sobol_baseline", self.noise_name, repeat_idx)
        return random.Random(repeat_seed_int(key))

    def _run_sobol_baseline(
        self,
        rid: int,
        experiment: CoreExperiment,
        locator_config: dict[str, Any],
        noise_std: float,
        noise_max_dev: float | None,
        signal_max_span: float | None,
    ) -> dict[str, Any]:
        """Simulate the SimpleSobolBayesianLocator until convergence and return detailed stats."""
        import math

        from nvision.runner.convert import belief_mode_estimates
        from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
        from nvision.sim.locs.bayesian.sobol_bayesian_locator import SimpleSobolBayesianLocator

        # Create a fresh RNG for the Sobol baseline measurements
        sobol_rng = self._rng_for_sobol_baseline(rid)

        parameter_bounds = self._injected_parameter_bounds(experiment)

        # Build belief directly
        belief = nv_center_smc_belief(parameter_bounds)

        locator = SimpleSobolBayesianLocator(
            belief=belief,
            max_steps=10000,
            noise_std=noise_std,
            **({} if noise_max_dev is None else {"noise_max_dev": noise_max_dev}),
            **({} if signal_max_span is None else {"signal_max_span": signal_max_span}),
        )

        step = 0
        sobol_xs = []
        sobol_ys = []
        sobol_freq_steps = None
        sobol_freq_uncert_at_conv = None
        sobol_freq_err_at_conv = None
        true_freq = experiment.true_signal.get_param_value("frequency")

        while not locator.done():
            _check_memory_limit()
            step += 1
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

        return {
            "sobol_baseline_steps": locator.step_count,
            "sobol_freq_steps": sobol_freq_steps,
            "sobol_freq_uncert_at_conv": sobol_freq_uncert_at_conv,
            "sobol_freq_err_at_conv": sobol_freq_err_at_conv,
            "sobol_xs": sobol_xs,
            "sobol_ys": sobol_ys,
            "sobol_mode_estimates": sobol_mode_estimates,
        }

    def _build_experiment(self, rng: random.Random) -> CoreExperiment:
        true_signal = self.task.generator.generate(rng)
        x_min, x_max = self._domain_from_signal_params(true_signal)
        if x_min is None or x_max is None:
            x_min, x_max = self._domain_from_generator(self.task.generator)
        if x_min is None or x_max is None:
            raise ValueError("TrueSignal must expose x_min/x_max parameters or frequency bounds")

        # Attach noise model (Bayesian inference dual of the physical noise)
        if self.task.noise is not None:
            noise_sig_model = self.task.noise.to_noise_signal_model()
            noise_bounds = noise_sig_model.spec.bounds if noise_sig_model else {}
            true_signal = dataclasses.replace(
                true_signal,
                noise_model=noise_sig_model,
                noise_bounds=noise_bounds,
            )

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
            max_steps=self.task.loc_max_steps,
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
        noise_std = 0.05
        noise_max_dev: float | None = None
        if experiment.noise is not None:
            noise_std = float(experiment.noise.estimated_noise_std())
            if hasattr(experiment.noise, "estimated_max_noise_deviation"):
                noise_max_dev = float(experiment.noise.estimated_max_noise_deviation(n_samples=6))

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
            **({"noise_model": experiment.true_signal.noise_model} if requires_belief else {}),
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
        noise_std = 0.05
        noise_max_dev: float | None = None
        if experiment.noise is not None:
            noise_std = float(experiment.noise.estimated_noise_std())
            if hasattr(experiment.noise, "estimated_max_noise_deviation"):
                # Use a default of 6 samples (previously mid-sweep) to account for mid-step sample size.
                noise_max_dev = float(experiment.noise.estimated_max_noise_deviation(n_samples=6))
        # Read signal spans from the model's declared methods.
        domain_width = float(experiment.x_max - experiment.x_min)
        signal_max_span: float | None = None
        model = experiment.true_signal.model
        if hasattr(model, "signal_min_span") and callable(model.signal_min_span):
            model.signal_min_span(domain_width)
        if hasattr(model, "signal_max_span") and callable(model.signal_max_span):
            signal_max_span = model.signal_max_span(domain_width)
        # Run Sobol Sweep baseline for this signal repeat — skip when we ARE the Sobol baseline
        sobol_baseline_steps: int | None = None
        sobol_freq_steps: int | None = None
        if self.strategy_name != "SimpleSobol":
            sobol_data = self._sweep_cache.get_sobol_baseline(
                experiment, self.task.seed, self.generator_name, self.noise_name, rid
            )
            if sobol_data is None:
                sobol_data = self._run_sobol_baseline(
                    rid, experiment, locator_config, noise_std, noise_max_dev, signal_max_span
                )
                self._sweep_cache.put_sobol_baseline(
                    experiment, self.task.seed, self.generator_name, self.noise_name, rid, sobol_data
                )
            if sobol_data is not None:
                sobol_baseline_steps = sobol_data.get("sobol_baseline_steps")
                sobol_freq_steps = sobol_data.get("sobol_freq_steps")

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
            if experiment.true_signal.noise_model is not None:
                cfg["noise_model"] = experiment.true_signal.noise_model

        observer = Observer(experiment.true_signal, experiment.x_min, experiment.x_max)
        token = set_combination_log_initials(self.generator_name, self.noise_name, self.strategy_name, repeat_idx=rid)

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
        finally:
            reset_combination_log_initials(token)

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
            getattr(last_loc, "initial_sweep_steps", 0)
            step_count = getattr(last_loc, "step_count", 0)
            inf_steps = getattr(last_loc, "inference_step_count", 0)
            max_steps = getattr(last_loc, "max_steps", 0)

            # For sweep-only runs (no inference phase), all steps are sweep steps.
            if inf_steps == 0 and step_count > (eff_sweep_steps or 0):
                eff_sweep_steps = step_count
            finalize_record["sweep_steps"] = int(eff_sweep_steps or 0)
            finalize_record["locator_steps"] = int(inf_steps or 0)
        finalize_record["sobol_baseline_steps"] = sobol_baseline_steps
        finalize_record["sobol_freq_steps"] = sobol_freq_steps
        if sobol_baseline_steps is not None and sobol_freq_steps is not None:
            finalize_record["sobol_conv_diff"] = sobol_baseline_steps - sobol_freq_steps
        else:
            finalize_record["sobol_conv_diff"] = None

        if self.strategy_name != "SimpleSobol" and sobol_data is not None:
            finalize_record["sobol_freq_uncert_at_conv"] = sobol_data.get("sobol_freq_uncert_at_conv")
            finalize_record["sobol_freq_err_at_conv"] = sobol_data.get("sobol_freq_err_at_conv")
        else:
            finalize_record["sobol_freq_uncert_at_conv"] = None
            finalize_record["sobol_freq_err_at_conv"] = None

        return history_df, finalize_record, stop_reason, result

    @staticmethod
    def _injected_parameter_bounds(experiment: CoreExperiment) -> dict[str, tuple[float, float]]:
        """Consolidate all parameter bounds for the locator.

        Any amplitude-like parameter is clamped so its lower bound is at least
        the estimated measurement-noise standard deviation (when feasible).
        """
        bounds: dict[str, tuple[float, float]] = {}
        noise_std = 0.05
        if experiment.noise is not None:
            noise_std = float(experiment.noise.estimated_noise_std())

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
                        lo = float(lo_raw)  # Fallback if noise too high
            bounds[name] = (lo, hi)

        # Merge noise bounds (no clamping needed for these)
        if experiment.true_signal.noise_bounds:
            bounds.update(experiment.true_signal.noise_bounds)

        return bounds
