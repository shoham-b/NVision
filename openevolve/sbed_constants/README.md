# `sbed_constants` — noise-aware numeric tuning of the SBED acquisition constants

A **no-LLM** optimizer over the 7 hardcoded numeric constants in
`nvision/sim/locs/bayesian/sbed_locator.py`'s `_acquire` / `_dual_window_acquire`.
It replaces the LLM-driven code search in `../sbed_acquisition/`, which was shown
not to work on this problem — for a statistical reason, not an implementation one.

Nothing under `nvision/` is modified, read or written by this package.

## Why this exists: the winner's curse

Three OpenEvolve searches over the acquisition *code* found nothing real. Two
"winning" candidates were later re-measured head-to-head against the unmodified
baseline and **both lost**:

| arm | combined_score |
|---|---|
| unmodified baseline | 0.5855 |
| search winner A | 0.5096 |
| search winner B | 0.4974 |

The cause is `max`. OpenEvolve promotes the best-scoring candidate, and the
maximum over a set of noisy estimates is a **biased** estimator of the underlying
value. At ~30 candidates and an evaluation sd of 0.06–0.08, the expected maximum
sits ~2 sd above the mean **even when every candidate is functionally identical
to the baseline**. Roughly +0.12–0.16 — which is exactly the size of every
apparent "win" the searches produced. They were measuring their own noise.

So the single design requirement for this package is that it must not be
fooled the same way.

## How the noise is handled

The optimizer is Gaussian-process Bayesian optimization with an **explicit
observation-noise model**. Five specific choices, all in `optimizer.py`:

1. **A `WhiteKernel` nugget in the kernel, fitted by marginal likelihood.** The
   run *measures* the evaluation noise rather than assuming it, and prints the
   fitted sd every iteration. If the fitted noise comes back at 0.07 while the
   spread of posterior means across the whole box is 0.02, the honest conclusion
   is "these constants do not matter" — and the run will say so instead of
   handing you a winner anyway.

2. **The recommendation is `argmax` of the posterior *mean*, never `argmax` of an
   observed score.** The posterior mean shrinks each observation toward what its
   neighbours say, which is precisely the correction the winner's curse needs.
   The smoke run below shows this working: a point that drew 0.8198 once is
   reported at 0.4686.

3. **Incumbents are re-evaluated.** Every `--reeval-every` iterations (default 4)
   the loop spends its evaluation re-measuring the current recommendation instead
   of proposing somewhere new. Replicates at identical coordinates are what let
   the GP separate the nugget from the signal at all — without them the marginal
   likelihood can explain every wiggle with a short length scale and near-zero
   noise.

4. **Expected improvement uses the *latent* standard deviation.** With a
   `WhiteKernel` in the kernel, sklearn's `predict(return_std=True)` returns a
   std that includes observation noise and therefore has a floor it can never go
   below. Left uncorrected, EI stays large everywhere forever and the search
   silently degenerates into random sampling. The nugget is subtracted first.

5. **EI is taken against the posterior-mean incumbent, not `max(y)`.** Chasing a
   lucky draw is the same mistake as reporting one.

**Why BO and not CMA-ES.** CMA-ES would also have been defensible — it is
rank-based and tolerates noise well. BO won on budget: at ~35 s per evaluation a
realistic run is 100–200 evaluations, and 7-dimensional CMA-ES with a population
of ~10 gets 10–20 generations out of that, too few for the covariance adaptation
to earn its keep. The GP also yields something CMA-ES does not: a directly
reportable estimate of the observation noise, which is the number that decides
whether any of this was worth doing.

## The parameter space

Defined in `space.py`. Bounds are deliberately generous — a box drawn snugly
around the current values can only ever conclude "the current values are fine".

| constant | baseline | bounds | scale |
|---|---|---|---|
| `decay_tau` — exploration decay time constant | 25.0 | 5 – 100 | log |
| `explore_p` — exploration probability scale | 0.1 | 0 – 0.5 | linear |
| `dip_p` — dip-branch probability boundary | 0.2 | 0.02 – 0.6 | linear |
| `min_obs` — min observations before dip detection | 5 | 2 – 30 | integer |
| `jitter_hz` — dip jitter half-width | 5 MHz | 0.5 – 20 MHz | log |
| `dual_trigger` — dual-window trigger multiplier | 3.0 | 0.5 – 6.0 | linear |
| `dual_halfwidth` — dual-window half-width multiplier | 3.0 | 0.5 – 6.0 | linear |

Two constraints are enforced by projection, so every proposed point is feasible
by construction:

- **`explore_p <= dip_p`.** The two probabilities gate consecutive branches of a
  single `rand_val` draw. Since `decay = exp(-step/tau) <= 1` for all
  non-negative steps, `explore_p <= dip_p` keeps the explore branch strictly
  inside the dip branch's interval at *every* step, not just at step 0.
- **`dual_halfwidth <= dual_trigger`.** The two flank windows are centred at
  `center ± split` with half-width `h·lw`, and they overlap — destroying the
  flank *balance* the branch exists to enforce — as soon as `h·lw >= split`. The
  branch only fires when `split >= trigger·lw`, so `h <= trigger` guarantees a
  gap. The rendered code additionally clamps to `min(h·lw, 0.9·split)` as a hard
  runtime backstop (the same guard a previous candidate used correctly), but with
  this constraint the clamp never binds at a proposed point — which is what keeps
  the rendered defaults behaviourally identical to production.

The surrogate is fitted on the **repaired** coordinates, not the requested ones.
Fitting on requested coordinates would show the GP a plateau of distinct inputs
producing identical outputs and charge the difference to observation noise — not
acceptable in a search whose whole difficulty is separating signal from noise.

## How the constants are overridden

They are inline numeric literals, not named parameters, so they cannot be set as
attributes — and `nvision/` must not be refactored (production carries unrelated
uncommitted work). This package reuses the mechanism `../sbed_acquisition/`
already relies on: render a standalone module containing the three acquisition
functions and monkeypatch it onto the real
`SequentialBayesianExperimentDesignLocator` for the duration of one evaluation.

The template is **derived at runtime** from
`../sbed_acquisition/initial_program.py` rather than kept as a second checked-in
copy. A checked-in copy rots the moment the acquisition source changes, and its
failure mode is the worst available: the optimizer keeps scoring a stale
algorithm while reporting numbers as if they came from the current one. Deriving
it means one source of truth, and each of the 8 substitution sites asserts it
matched **exactly once** — an edited line raises instead of silently substituting
nothing. Every run additionally diffs the rendered defaults against the source
and fails if any line outside those 8 sites changed. (Some sites change
textually without changing meaning — `5e6` renders as `5000000.0`, the same
float. At the defaults the only *semantic* change is the `half_width`
non-overlap backstop, which cannot bind.) This structural check is worth far
more than observing that the baseline and the rendered defaults *score*
similarly: at this evaluation noise, two genuinely different acquisition
functions score similarly all the time.

## Scoring

`objective.py` owns **no** scoring logic. The anytime log-error curve, the
calibration and catastrophic gates, and the `combined_score` composition are all
imported from `../sbed_acquisition/evaluator.py` (loaded by path — that directory
is not a package). Scores are therefore directly comparable with the previous
search's numbers and with anything that harness produces in future.
Re-implementing the score would fork the definition and make every historical
comparison meaningless.

The default grid is the harness's own `EVAL_GRID`: 27 combinations
(9 fixed physical variants × 3 noise levels) × 2 repeats = 54 repeats.

## Usage

```bash
# wiring check, ~30 s -- proves render -> monkeypatch -> run -> score works
uv run --no-sync python openevolve/sbed_constants/run.py smoke

# the real run
uv run --no-sync python openevolve/sbed_constants/run.py tune --n-init 20 --n-iter 130

# resume an interrupted run from its JSONL log
uv run --no-sync python openevolve/sbed_constants/run.py tune --resume

# MANDATORY before believing anything
uv run --no-sync python openevolve/sbed_constants/run.py validate \
    --params-json openevolve/sbed_constants/output/best.json --repeats-per-arm 12
```

`tune` writes `output/tune_log.jsonl` (one line per evaluation, resumable),
`output/best.json`, `output/summary.json` and `output/best_program.py` (the
rendered winner, ready to diff against `initial_program.py`).

### Optional: common random numbers

`--crn-seed N` re-seeds the global numpy RNG before each combination, so two
candidates evaluated at the same seed start each combination from the same
stream. It reduces between-candidate variance but does not eliminate it —
candidates consume different numbers of draws within a repeat, so the streams
desynchronise as the repeat proceeds. It is a **search-time** variance reduction
only. Validation is always unseeded; validating under CRN measures the candidate
on one particular random stream, which is the overfit being tested for.

## Smoke-run evidence

From an actual `smoke` run (tiny grid, so the scores rank nothing — what matters
is the machinery):

- The template and structural checks pass: all 8 substitution sites match, and
  rendering the defaults changes only those sites.
- One initial-design point scored **0.0000** — a gate failure — and the loop
  recorded it and carried on rather than crashing.
- The re-evaluation schedule fired twice, both times on the baseline point (the
  model's current recommendation). Three replicates of that *identical* point
  scored **0.4337 / 0.4330 / 0.4000**. That spread, at fixed constants, is the
  noise floor the old search mistook for improvement.
- An earlier smoke run showed the shrinkage doing its job explicitly: a point
  that drew **0.8198** was re-evaluated at **0.4094** and reported at posterior
  mean **0.4686**. A best-observed-score optimizer would have promoted 0.8198 as
  a 2× win.

## Cost

**Measured on this machine**, default 54-repeat grid, single-threaded:
**41 s per evaluation** warm. The first evaluation of a process costs ~106 s
(imports and numba JIT warmup) and is a one-off, not a per-evaluation cost.

Two full-grid baseline evaluations scored **0.5868** and **0.5408** — both inside
the documented baseline range of 0.521–0.585, confirming this wrapper reproduces
the harness's scoring. Note also that those two draws of the *same* program span
0.046, which is the problem this package is built around.

| run | evaluations | wall clock |
|---|---|---|
| `smoke` | 11 (tiny grid) | ~30 s |
| short `tune` (`--n-init 16 --n-iter 64`) | 80 | ~55 min |
| `tune` default (`--n-init 20 --n-iter 130`) | 150 | **~1 h 45 min** |
| `validate --repeats-per-arm 12` | 24 | ~17 min |

`tune` appends to `output/tune_log.jsonl` after every evaluation and `--resume`
picks up from it, so a long run can be interrupted and continued without losing
work.

`--repeats N` rescales the grid if you want cheaper, noisier evaluations; note
that evaluation sd scales as ~1/√N, and at this signal-to-noise cheaper
evaluations are usually a false economy.

## How a winner must be validated — read this before reporting anything

**A single best-observed score is the most biased number available.** It is the
number that produced two false winners already, and the optimizer prints it
labelled `<-- biased, do not report this` for that reason.

`validate` is the gate:

- **Both arms are re-measured from scratch, in the same run.** A baseline number
  remembered from an earlier session is not a control — machine load, repeat
  counts and the acquisition source may all have moved since.
- **The baseline arm is the unmodified `initial_program.py`, used verbatim** —
  not the renderer's output at default constants. If the renderer has a bug,
  running it on both arms cancels the bug out and hides it.
- **Arms alternate** A, B, A, B… rather than running as two blocks, so anything
  that drifts over the wall-clock of the comparison hits both arms equally.
- **Evaluations are unseeded** (no CRN).
- **The output is a confidence interval, not a winner.** At evaluation sd ~0.07,
  `n` repeats per arm resolve a difference of roughly `2.8 × 0.07 / √n`:

  | repeats per arm | smallest resolvable difference | total cost (both arms) |
  |---|---|---|
  | 8 | ~0.070 | ~11 min |
  | 12 | ~0.057 | ~17 min |
  | 20 | ~0.044 | ~27 min |
  | 50 | ~0.028 | ~68 min |
  | 190 | ~0.014 | ~4.3 h |

  If the 95% CI on the difference contains zero, the correct report is **"no
  measurable effect"**, not "slightly better". A tuned constant vector that
  cannot clear this bar must not be ported into `nvision/`.

Finally, note what `validate` does *not* establish. It is still the same
27-combination simulation grid the search optimized on. Per the cautionary
history in `../sbed_acquisition/evaluator.py` — where a candidate showing a ~2×
sample-efficiency gain on this grid produced *no measurable effect* on the real
pipeline at 450 repeats × 150 grid cells per arm — **real-pipeline A/B validation
before porting remains mandatory**. A clean `validate` result is evidence that
the tuning is not noise; it is not evidence that it helps in production.
