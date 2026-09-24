# Deferred Graph Generation

Building a repeat's graphs is not needed to *have* its result, so `nv run` saves results first and
builds graphs afterwards in separate **graph-worker** processes. Code: `nvision/runner/graph_queue.py`,
launched from `nvision/cli/run.py` (`_GraphWorkers`), drained manually by `nv graphs`
(`nvision/cli/graphs_cmd.py`).

## What is (and is not) a graph

- The Bayesian extras (`bayesian_posterior_data`, `..._fisher_data`, `..._convergence_metrics_data`, ...)
  are *data payloads*; the browser renders them. Producing them is data reduction from particle
  snapshots, not drawing.
- The `scan` entry is the one Plotly figure the server builds.
- A repeat's metrics, per-step `series` and results row are **not** graphs: they are saved immediately
  and always exist.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `NVISION_GRAPH_REPEATS` | `10` | Build graphs only for the first N repeats of each combination (0 = all). Later repeats keep metrics/series but get no figures (`plot_skipped`). |
| `NVISION_GRAPH_WORKERS` | half of `--runners` (min 1) | Graph-worker processes `nv run` starts. `0` builds graphs inline in the runners (no deferral). Set explicitly to override the runner-scaled default -- see `graph_workers_for` in `nvision/cli/defaults.py`; a single worker against several runners lets the `graph_queue/` spool backlog grow far past what's actually archived. |
| `NVISION_PLOT_SWEEP_STRATEGIES` | `0` | Also build figures for `SimpleSweep`/`SimpleSobol` (off by default). |
| `NVISION_STALL_DUMP_S` | `600` | Stall watchdog interval, see below. `0` disables. |

## Flow

1. A runner finishes a repeat. `generate_attempt_plots(defer=True)` returns only the slim scan entry,
   flagged `plots_pending`. The repeat is saved to the cache.
2. Only *after* that save, the runner writes a **state file** with everything the graphs need
   (`GraphJob`: run result, experiment, history, SBED's own baselines) into `<out>/graph_queue/`.
   Repeats beyond `NVISION_GRAPH_REPEATS`, sweep strategies and cache hits queue nothing.
3. When every sub-task of a combination is done (the moment it becomes archivable), the parent marks
   it **ready**. Workers start lazily at the first ready combination.
4. A worker claims a ready combination (atomic file rename), builds the graphs of all its repeats,
   re-saves each repeat with them, and **only then archives** the combination. Claiming whole
   combinations means several workers never touch the same one.
5. At the end of `nv run`, the parent waits for the workers to drain the queue.

Spool files (`<out>/graph_queue/`): `<slug>__rNNNNNN.state`, `<slug>.ready`, `<slug>.claimed`,
`<slug>__rNNNNNN.failed`, `producers_done`.

## Interruption and recovery

The queue is plain files, so it survives Ctrl-C or a crash. `nv graphs [--dir D] [--workers N]
[--retry-failed]` builds whatever is left (combinations claimed by a killed worker are returned to the
queue first). Repeats whose graphs failed are kept as `.failed` and retried only with `--retry-failed`.

Until a repeat's graphs exist, its manifest entry has `plots_pending` and **no `path`**, so the results
UI shows the metrics but an empty scan area. Press `r` in the UI after the workers finish.

## Not deferred

Deferral needs the run state to be picklable and the repeat to be saved in streaming mode. A repeat
that cannot be deferred simply builds its graphs inline as before.

## Stall watchdog

`nvision/tools/stall_watch.py`: a worker that finishes no repeat, or a pool that completes no task,
for `NVISION_STALL_DUMP_S` seconds writes every thread's stack to `logs/stall-worker-<pid>.txt` /
`logs/stall-pool-<pid>.txt`, repeating while stuck. It uses `faulthandler`, so it works even if the
interpreter is blocked on a lock or in native code. A repeat whose metrics+graphs step takes 5 s or
more is also logged at INFO ("Slow repeat outputs").

## Cache and invalidation

No result values change: deferral only changes *when* graphs are built. Combinations run before this
feature keep the graphs they have; only newly run repeats follow the first-N rule. No re-run or
`nv render` is needed.
