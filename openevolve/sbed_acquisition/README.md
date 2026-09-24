# OpenEvolve: Bayesian-SBED acquisition-point selection

Evolves the acquisition-point-selection logic of the `Bayesian-SBED` locator
(`nvision/sim/locs/bayesian/sbed_locator.py`), scored on how few measurements
it needs to reach a converged Zeeman-splitting estimate. See
`.claude/skills/locator-evaluation` for why `splitting_converged_step` is the
right metric here, not the locator's own stop reason.

## Files

- `initial_program.py` — standalone copies of `_acquire` and `_eig_acquire`, taken verbatim from
  `nvision/sim/locs/bayesian/sbed_locator.py`. Only the code between
  `EVOLVE-BLOCK-START`/`END` is mutated. **This never touches production
  code** — the evaluator monkeypatches these functions onto the real locator
  class for the duration of one evaluation, in a fresh subprocess/import each
  time, and nothing is written back to `nvision/`.
- `evaluator.py` — loads a candidate, monkeypatches it in, runs a small
  evaluation grid directly through `nvision.runner` (in-process, no CLI
  subprocess, `dry_run=True` so nothing touches `artifacts/cache/`), and
  scores it. Read the module docstring before changing the scoring — it
  explains the convergence-rate/catastrophic-error gates and why a plain
  speed metric would be exploitable (this codebase has a documented history
  of "improvements" that were actually gaming the benchmark's
  truth-centred prior — see project memory `sbed-voigt-fit-quality-fixes.md`).
- `config.yaml` — evolution parameters, tuned for a *small* local run (30
  iterations, population 30, 1 island) using the local `claude` CLI as the
  LLM backend (`provider: claude_code`) instead of a hosted API key.
- `claude_code_stdin.py` / `run.py` — a Windows-safe drop-in replacement for
  openevolve's built-in `claude_code` LLM backend, and the launcher needed to
  wire it in. See "Windows: command-line length limit" below — required on
  this platform, not optional polish.

## Running

Requires `claude login` to have been run once, and the `openevolve` dev
dependency installed (`uv sync`):

```bash
uv run python openevolve/sbed_acquisition/run.py --iterations 30
```

Override iteration count for a quick smoke test:

```bash
uv run python openevolve/sbed_acquisition/run.py --iterations 3
```

Use `run.py`, **not** the `openevolve-run` CLI directly — see
"Windows: command-line length limit" below for why.

### Troubleshooting

- **`Iteration N error: No valid diffs found in response` on every
  iteration**: the `claude` CLI's login session has expired. Run
  `claude login` (this is separate from whatever authenticates your
  interactive Claude Code session — confirmed independently, expiring on its
  own schedule).
- **`Generated code exceeds maximum length`**: `config.yaml`'s
  `max_code_length` is smaller than the file OpenEvolve is trying to save.
  `initial_program.py` itself is already ~12KB, well over the framework's
  10000-char default, so this needs real headroom above that (currently set
  to 25000) -- if you significantly grow the evolve target, raise it further.
- **`OSError: [WinError 206] The filename or extension is too long`**: you
  ran `openevolve-run` directly instead of `run.py`. See below.

### Windows: command-line length limit

The installed `openevolve` package's `provider: claude_code` backend
(`openevolve/llm/claude_code.py` in `.venv`) passes the *entire* prompt --
current program plus up to `num_top_programs + num_diverse_programs` full
previous-program bodies from `config.yaml` -- as a single trailing
command-line argument to `claude -p`. Once the program database has a few
entries, that reliably exceeds Windows' ~32K character command-line limit:
iterations early in a run (small database) succeed, then every iteration
after fails identically with `WinError 206`. Reproduced and confirmed while
setting this up.

The fix is `claude_code_stdin.py` (pipes the prompt via stdin instead --
confirmed working directly against the CLI first) plus `run.py` (a launcher
that wires it in via OpenEvolve's `init_client` config hook, since YAML can't
express a custom Python callable, and handles making the class picklable
across Windows' multiprocessing 'spawn' workers). **Always use `run.py`, not
`openevolve-run`**, on this project. Delete `claude_code_stdin.py`/`run.py`
and switch back to `openevolve-run` + `provider: claude_code` if upstream
ever fixes this.

### Cosmetic-only: emoji logging crash

You may see a `UnicodeEncodeError` on 🌟/✅ characters that openevolve's own
logging emits (`process_parallel.py`) -- Windows' terminal codepage (cp1252)
can't render them. Python's logging module catches this internally; it does
not affect the run (evolution completes and results save normally either
way). Safe to ignore.

Output (checkpoints, best program, logs) goes to `openevolve_output/` at the
repo root by default (gitignored) — pass `--output <dir>` to redirect it.

## Cost / timing

One evaluation of the unmodified `initial_program.py` (the full
`EVAL_GRID`: 4 repeats at `Gauss(0.0)` + 4 at `Gauss(0.05)`) takes ~35s on
this machine and scored `combined_score=0.30` (`convergence_rate=0.5`,
`catastrophic_rate=0.25`, `median_steps=40.5`) — those numbers came out right
at both gate boundaries, so treat them as a rough baseline, not a precise
one; the locator's own acquisition randomness isn't seeded (see below), so
re-running the same program gives a different number each time. At 30
iterations that's roughly 20-30 minutes of pure evaluation time, plus
whatever the `claude` CLI calls take per iteration.

Higher-noise combos (`Gauss(0.1)+`) were measured at **~70s/repeat and
frequently fail to converge within the step budget at all** (median error
several MHz — catastrophic by this evaluator's own bar). They're excluded
from `EVAL_GRID` deliberately: at that noise level the *evaluation* is both
too slow and too noisy for a search loop, not usefully "harder". If you want
to validate a winning candidate at realistic noise, do that afterward with
the normal pipeline (`nv groups run both-sbed`), not by editing
`EVAL_GRID`.

## Known limitation: acquisition randomness isn't seeded

`NVISION_RNG_SEED` fixes the *ground-truth* signal draw per repeat, but the
locator's own exploration randomness inside `_acquire`
(`np.random.rand()`/`uniform()`/`choice()`) uses the global, unseeded numpy
RNG. Two evaluations of the *identical* candidate program will therefore
score differently — this was confirmed empirically while building this
harness (see the fidelity-check scratch scripts referenced in the commit
that added this directory) and is a property of the existing simulation
code, not something this evaluator works around. `EVAL_GRID`'s repeat counts
(4+4) are a compromise between averaging out that noise and keeping one
evaluation fast; if OpenEvolve's search looks unstable, raising repeat
counts (at the cost of wall-clock time per iteration) is the first knob to
try, not chasing determinism that doesn't exist yet elsewhere in this
codebase.

## The pre-run CRLB feasibility gate

`nvision/runner/executor.py`'s pre-run oracle feasibility check
(`NVISION_CRLB_FEASIBILITY_MARGIN`, default `1.0`) skips a repeat entirely
(0 measurements) when the true parameters make the convergence threshold
provably unreachable within the step budget — computed from ground truth
before the locator runs, so it's identical no matter which acquisition
strategy is under evolution. At the default margin this skips a large
fraction of repeats for this generator, which is pure noise for a search
that only affects what happens *after* the gate. `evaluator.py` raises the
margin (`NVISION_CRLB_FEASIBILITY_MARGIN=50`) for evaluation purposes only,
via an env var set before `nvision.sim.defaults` is imported — this doesn't
touch the production default used by real experiment runs.

## After evolution finishes

`openevolve_output/best/best_program.py` (or the path OpenEvolve prints at
the end) holds the winning candidate. Review it like any other diff before
doing anything with it:

1. Read the evolved `EVOLVE-BLOCK` — check it doesn't violate the hard
   constraints in `config.yaml`'s system prompt (domain bounds, no silent
   fallbacks, doesn't touch stopping logic).
2. Validate it properly: copy the winning block into a scratch locator
   subclass (or temporarily monkeypatch it the same way `evaluator.py`
   does) and run it through the real evaluation pipeline
   (`nv groups run both-sbed` / the `nvision-convergence-check` and
   `nvision-plot-integrity-check` skills) at realistic noise levels and
   repeat counts — `EVAL_GRID` here is deliberately a cheap proxy, not a
   substitute for that.
3. Only then port the change into
   `nvision/sim/locs/bayesian/sbed_locator.py` by hand, preserving the
   original file's extensive inline documentation of *why* each branch
   exists (this evolve target's copy strips most of it for brevity — see
   the original for the full rationale of each piece you're replacing).
