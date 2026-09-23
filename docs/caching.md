# NVision Caching Guide

This document describes the caching architecture, execution control flags, and interruption salvaging mechanisms in the NVision simulation locator tool.

---

## 1. Caching Behavior Matrix

Caching is enabled by default to optimize repeat runs and avoid redundant computations. Developers and AI agents can control caching interactions using the `--no-cache` and `--dry-run` CLI options.

| Flag Scenario | Ignore Cache on Load? | Purges Old Cache? | Saves Results to Cache Database? | HARVESTS Partial Results on Ctrl-C? |
| :--- | :--- | :--- | :--- | :--- |
| **Default** (`--cache`) | ❌ No | ❌ No (Loads cache/resumes) |  Yes |  Yes (Resumable) |
| **`--no-cache`** |  Yes |  Yes (After 1st repeat) |  Yes |  Yes (Resumable) |
| **`--dry-run`** | ❌ No (Uses cache hits) | ❌ No | ❌ No | ❌ No |
| **`--no-cache --dry-run`** |  Yes | ❌ No | ❌ No | ❌ No |

---

## 2. Flags & Mechanics

### Default Caching Flow (`--cache`)
* **Behavior:** The runner queries the SQLite sharded cache database for any matching repeats. If matching repeats exist, they are restored instantly. Execution resumes exactly at the next missing repeat index.
* **Saving:** Every newly finished repeat is saved incrementally. When all repeats finish, the main database combination pointer is updated.

### Ignored Caching Flow (`--no-cache`)
* **Behavior:** The runner ignores any existing cache on load and starts calculations fresh from repeat `0`.
* **Purging & Saving:** Once the first repeat finishes successfully, the old cache database entries for the combination are purged/deleted. Fresh results are then actively saved to the database as they finish, allowing future cache-enabled runs to leverage them.

### Dry-Run Bypass Flow (`--dry-run`)
* **Behavior:** The runner completely bypasses all cache updates. 
* **Mechanics:** All cache database purges, background repeat writes, and final full saves are completely disabled. 

---

## 3. Harvester Recovery (Resiliency on Interruption)

When a simulation is interrupted by user command (`Ctrl-C` / `KeyboardInterrupt`):
1. Background worker threads flush any completed repeats to the SQLite database.
2. The parent process KeyboardInterrupt handler automatically invokes the harvester (`_harvest_partial_results_from_cache`).
3. The harvester bypasses `skip_cache` checks and queries the SQLite database to retrieve all completed repeats for the task.
4. Harvester appends the completed repeats directly into the final `locator_results.parquet` output file and static UI plots manifest.
5. The next cached run will automatically load the completed repeats and resume exactly where it was interrupted, preventing any lost progress.

---

## 4. Generations (`nv cache gen`)

The cache is a single mutable store keyed by `(generator, noise, strategy, repeats, ...)` --
a fresh `--no-cache` run purges and overwrites a combination's old entry once its first
repeat finishes. That's fine when the new run is just faster; it's lossy when the new code
also changes numerics (a "performance and convergence" change, not just performance), since
the old and new results can't coexist in the same key.

`nv cache gen` manages named snapshots of the whole `<out>/cache` directory so you can archive
before that kind of change, keep a rolling window of recent attempts, and roll back:

```bash
# Snapshot the current cache (label defaults to the current git commit short hash)
uv run nv cache gen save --label before-optimization

# List saved generations, most recent first
uv run nv cache gen list

# Replace the live cache with a saved generation (archives the current one first, unless --no-archive-current)
uv run nv cache gen restore before-optimization

# Keep only the 2 most recent generations, deleting the rest
uv run nv cache gen prune --keep 2
```

---

## 5. Resuming Interrupted Runs (`--resume`)

When a large parameter grid or group run (e.g. `nv groups both-sbed --repeats 50 --no-cache`) is interrupted midway, the cache is left with fresh results for the combinations reached so far, but stale entries for combinations that haven't started yet.

The `--resume` flag non-destructively bridges this gap:
1. **Scans the latest session log** (`logs/nvision-run-*.log`) to detect which combinations executed in that session.
2. **Completed combinations:** Preserved and skipped instantly via cache.
3. **Partial combinations:** Resumed from where they were interrupted (e.g. repeat 26 $\to$ 50).
4. **Unstarted combinations:** Executed fresh from repeat 0 to the target count, bypassing any stale pre-session cache without requiring manual purging or database deletions.

```bash
# Resume any interrupted group run safely
uv run nv groups both-sbed --repeats 50 --resume
```


`save`/`restore` only ever copy — the live cache is never modified by `save`, and `restore`
archives whatever is live before overwriting it. Best done while no `nv run`/`nv groups` is
actively writing to the cache, for a fully consistent snapshot.

## 6. Cache Invalidation (`CACHE_SCHEMA_VERSION` and the physics fingerprint)

Every combination key hashes two version markers (`nvision/cache/locator_keys.py`):

- `CACHE_SCHEMA_VERSION` — bumped by hand when algorithm or payload semantics change (currently 11: candidate-grid density mixture, Rao-Blackwellized noise likelihood, no particle rejuvenation, noise-floor fix, prior-mean widening).
- `PHYSICS_CONFIG_FINGERPRINT` — derived automatically from the physical constants and bounds generators draw from.

A change to either makes `nv run` miss the old entries and recompute, so stale results are never silently reused. Old entries are not deleted: `nv serve` rebuilds each entry's key from its *stored* schema version and fingerprint (`api_server._combo_key`), so results from earlier versions stay listed and openable. `nv render`, `nv cache` subcommands and other CLI readers build keys from the current versions and will not see them; delete `artifacts/cache` to reclaim the disk.
