---
name: nvision-cli
description: Guides correct use of the NVision Typer CLI (`uv run nv ...`) for running experiments, re-rendering plots without re-running, serving results, and managing the SQLite cache. Use this whenever the user asks to run an NVision experiment/simulation/scan, regenerate plots or the UI, start the results server, inspect or clean the cache, or verify that a code change didn't break the pipeline — even if they just say "run this", "check it still works", or "regenerate the report" without naming the CLI explicitly. Also covers the Windows `uv run` lock-file gotcha and picking between `run`, `run-single`, `groups`, `render`, and `cache` so you don't default to a slow full run when a cheap one would do.
---

# NVision CLI

The CLI is a Typer app with two equivalent entry points, `nv` and `nvision` (`pyproject.toml`
`[project.scripts]`). Always invoke through `uv run` — never call `python file.py` directly, and
never invoke `nvision`'s internals as a library script. Full reference: `docs/cli_reference.md`.

## Picking the right command for the job

Defaulting to a full `nv run` is almost never what you want mid-task — it's slow and
overwrites/reads the shared cache. Match the command to the intent:

| Intent | Command |
|---|---|
| Verify a code change didn't break the pipeline | `nv run-single` with `--dry-run` (see below) — **not** `nv run` |
| Run the full batch of scenarios | `nv run` |
| One specific (generator, noise, strategy) combo | `nv run-single <generator> <noise> <strategy>` |
| A preset scenario bundle | `nv groups list` / `nv groups run <name>` / `nv groups <name>` (shortcut) |
| Only changed plotting/viz code, sim results unchanged | `nv render` — rebuilds `plots_manifest.json` + UI from cache, no re-simulation |
| View results in the browser | `nv serve` (port 18080; press `r` in-browser to reload) |
| Inspect or prune the cache | `nv cache list` / `nv cache clean --filter-strategy <name>` |

## Verifying a change cheaply

Before claiming a fix works, run the lightweight dry-run rather than a full batch:

```bash
uv run nv run-single NVCenter-lorentzian "Gauss(0.01)" Bayesian-SBED --loc-max-steps 3 --repeats 1 --runners 1 --dry-run
```

This exercises the CLI, locator, and task orchestration end-to-end without heavy simulation
or touching the cache. Only escalate to a real `nv run` (or ask the user first) if the dry-run
alone can't exercise what changed.

## `--dry-run` vs `--no-cache` — don't confuse these

- `--dry-run`: does **not write** results to the cache. Use for quick verification.
- `--no-cache`: **bypasses reading** the cache (forces fresh computation) but still **writes**
  results back. Use to force a clean re-run whose output should stick around.

## After an edit: render vs re-run

- **Plotting/viz code only** (`nvision/viz/`, plot mixins) — the cached simulation results are
  still valid, so just rebuild the report:
  ```bash
  uv run nv render
  ```
- **Core algorithmic code** (SMC resampling, Bayesian likelihood, locator logic, generators) —
  the cache is now stale. Tell the user explicitly whether they need a full re-run of
  `artifacts/cache/` or whether `render` is enough; don't silently assume. See `docs/caching.md`.

## Windows: `uv run` lock error

`uv run nv <command>` can fail with `os error 32` because `nvision.exe` is locked by a prior
process still holding a handle. Skip the sync step to work around it:

```bash
uv run --no-sync nv <command>
```

`--no-sync` goes immediately after `uv run`, before the command. Reach for this whenever you're
re-running the CLI back-to-back in the same session (e.g. dry-run after a fix, then `render`).

## Other useful flags on `run` / `run-single`

- `--repeats`: number of repeat experiments (default 5).
- `--loc-max-steps`: cap on Bayesian locator steps (default 1200; use a small number like 3 for
  verification runs).
- `--filter-category`, `--filter-strategy`, `--filter-generator`, `--filter-noise`: narrow which
  scenarios run.
- `--runners`: parallel runner processes (default 8); use `--runners 1` for sequential execution
  when you need a clean traceback while debugging.
- `--open`: open the results browser after the run completes.

## Out of scope

Testing (`pytest`) and linting (`ruff`) are separate concerns from this skill — see AGENTS.md
for those commands. This skill is only about the `nv` CLI surface itself. Also, per AGENTS.md,
only run checks against files actually modified by the current task, not the whole repo.
