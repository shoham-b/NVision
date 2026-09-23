# Runner Architecture & Orchestration

## Overview

The `nvision/runner/executor.py` module is responsible for orchestrating the execution of experiments. Because Bayesian inference is computationally intensive and experiments often involve thousands of simulated measurements across multiple strategies and noise levels, the runner employs a highly parallelized, streaming architecture.

## Execution Lifecycle

### 1. Repeat Artifact Generation
Instead of running a single long simulation, the runner breaks down tasks into **repeats** (independent runs with different random seeds) for statistical robustness.
- Each repeat generates its own independent random number generator (RNG).
- Measurements within a single repeat are deterministic given that repeat's specific seed.
- If a repeat is aborted, it can be resumed later without affecting others.

### 2. The Sobol Baseline (`_run_sobol_baseline`)
For every **Bayesian** strategy being tested (e.g., `SequentialBayesianExperimentDesignLocator`), the runner automatically performs a "Sobol Baseline" measurement first.

- **Purpose**: To provide a ground-truth benchmark of how a completely uniform, un-targeted random sequence (a van der Corput Sobol sequence) would converge given the same amount of time/noise. It calculates the expected uniform points needed.
- **Isolation**: The baseline is completely decoupled from the actual strategy execution. It instantiates a fresh `SimpleSobolBayesianLocator` and a fresh `UnitCubeSMCMarginalDistribution` (belief). 
- **Decoupling**: Non-Bayesian locators (like `StagedSobolSweepLocator` in `coarse/sobol_locator.py`) might still track a "belief" internally to observe data, but the executor explicitly avoids running the 10,000-step Bayesian baseline on them to prevent heavy inference algorithms from throttling coarse, fast sweep strategies.

### 3. The Main Locator Phase
After the baseline completes (or is skipped), a fresh belief is instantiated for the primary locator strategy.
- The `_run_single_repeat` method drives the primary locator's `.next()`, `.observe()`, and `.done()` hooks.
- A hard timeout is enforced via a threading monitor (e.g., stopping the loop if it exceeds `timeout_s`).
- At the end of the acquisition loop, if it's a Bayesian locator, the runner dumps the full posterior sample array into a parquet/feather artifact for downstream visualization.

### 3b. Stall diagnosis: the `--no-cache` purge (fixed 2026-09-23)

**Symptom.** On a long `nv run` / `nv groups` grid with `--no-cache`, the run sat idle for minutes,
then a whole batch of tasks started in the same second, and again a few minutes later. Workers were
using ~25% of one core each on a 12-core machine.

**Evidence.**
- Run log: after an initial burst, new "Running task" lines came in synchronized batches of six
  (two noise levels x three strategies), 4 min apart, growing to 6 min later in the run. A run
  whose combinations had all been purged before (flag present) had no gap at all.
- Live `py-spy dump` of the six runner processes and the parent during a gap: **all six** runner main
  threads were inside `_run_repeats -> _purge_cache_if_needed -> purge_cached_combination ->
  _ArchiveFallbackBackend.get -> ShardedSqliteCache.get / ComboArchive.get`; the parent was idle in
  `as_completed`. Nothing was blocked on a lock, the pool, graphs, or the parent.
- Cause: `purge_cached_combination` did `for k in backend: backend.get(k)` -- it iterated every key
  of the category cache (a run's cache had ~1M live keys: 880k `blob:`, 113k `repeat:`) and JSON-decoded
  each payload to compare `config` fields, once per combination. ~60-100 us/key measured on a synthetic
  cache with the same key formats, so >= 100 s single-process and minutes under 6-way disk/GIL
  contention. Every new combination also added ~35 keys, so the cost grew during the run. Because each
  batch of runners was doing the same fixed-length scan, they finished it together -- the synchronized
  batches.

**Fix.** The purge is now a keyed lookup: the combination's pointer/inline keys are hashes of its
identity, so they are computed directly (current schema, plus the legacy v8 form the read path still
accepts) and checked through the existing primary-key index (`keys_exist_batch`); only keys that exist
are removed, with a single batched `delete_many`. It no longer depends on cache size (3.0 s -> ~0 ms on a
46k-key synthetic cache) and it now also removes a combination's Parquet archive file, so stale archived
repeats cannot show through the archive fallback. Entries under other schema versions / physics
fingerprints (unreachable by current code) are not swept by the per-task purge; `nv cache clean` still
handles them.

**Does not need a re-run.** No result values change; only how fast a `--no-cache` task's purge is.

**If a run stalls again**, arm the stall watchdog (`NVISION_STALL_DUMP_S=60`, see
[graph_queue.md](graph_queue.md)) or attach `uvx py-spy dump --pid <pid>` to a runner and the parent:
the stacks say what each is blocked on.

### 4. Memory & Streaming Optimizations
When tasks request hundreds of combinations or repeats, storing all artifacts in memory before saving them would cause a memory exhaustion crash. 
The runner implements a streaming mode:
- If `repeats > STREAMING_REPEAT_THRESHOLD`, the runner yields results one-by-one as a generator.
- Results are saved to the cache on-the-fly (`save_cached_combination`).
- The in-memory history is cleared aggressively to keep the RAM footprint stable.
