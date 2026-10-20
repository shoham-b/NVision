# Locator comparison: sweeps, Smart-Sweep, Smart-SBED

Every locator searches for the resonance (`zeeman_split`, the primary parameter); the standard run groups
compare them on the same signals and noise levels. The generators are the ordinary ones, whose priors on the
nuisance parameters are loose (std 10 % of the parameter range). The prior-based FWHM sizing below (sweep
spacing, Smart-* search grid) is therefore less accurate than it would be after a prior classification of the
spectrum with tight nuisance priors; it still uses the prior's width, never the true signal's.

## Run groups

| Group | Signals | Noise | Strategies |
|---|---|---|---|
| `voigt-grid` | Voigt width x contrast x sigma_inhom | SBED noise grid | the eight below |
| `lorentzian-drift` | Lorentzian width x contrast | no-drift control + drift scenarios | the eight below |
| `drive-control` | CW / pulsed drive-controlled NV | SBED noise grid | `Bayesian-SBED`, `Bayesian-SBED-Drive` |
| `all` | the three above | the three above | each scenario's own set (`nv groups all`, 19224 tasks) |

`nv groups list` shows every group with its task count. The former groups `both-sbed`, `both-sbed-only`,
`both-sweep-only`, `lorentzian-plain-drift` and `lorentzian-plain-drift-sbed-only` still work as aliases of
`voigt-grid` / `lorentzian-drift` restricted to `Bayesian-SBED` (+ `SimpleSweep-3perFWHM`). To run any other
subset, narrow a group: `--filter-signal` (generator substring) and `--filter-strategy` (comma-separated
regex/substring).

## Running the comparison

`voigt-grid` has 18000 tasks (75 signals x 6 noise levels x 8 strategies), `lorentzian-drift` 1200 (25 signals x
6 noise settings x 8 strategies); narrow them for a first look.

```bash
uv run nv run --run-group lorentzian-drift --filter-strategy "Sweep,SBED" --repeats 30 --runners 8
uv run nv compare            # stage-by-stage table from artifacts/ (or --out DIR)
uv run nv serve              # Speed tab -> "Stages"; Compare A/B pickers (Sweep -> spacing)
```

`nv compare` prints three tables (accuracy; stage boundaries and reach rates; per-stage duration and
uncertainty ratio) for every locator on the same footing; `--filter-signal`, `--filter-noise`,
`--by generator,noise` and `--csv` refine and export it. `nv groups count lorentzian-drift` shows the task count.
A quick look: `NVISION_SBED_NOISE_STEPS=1 uv run nv run --run-group lorentzian-drift --filter-signal "c0.25" --repeats 5`.

In the UI the plain sweeps appear as one "Sweep" entry; choosing it reveals a second dropdown for the spacing
(k points per FWHM, 1-5).

## Width and spacing

The width that sizes everything is the FWHM, `2 * effective_hwhm` of the **prior** (the belief's prior-mean
parameters, clipped to their bounds) -- never the true signal's width (`nvision/sim/locs/search_grid.py`).

* A dip is seen only if several measurements fall inside it. Three points per FWHM
  (`SWEEP_POINTS_PER_FWHM = 3`) cannot step over a dip and give `find_dips` the cluster it needs. It is derived,
  not a tuning knob, so it is a module constant and not an env var.
* Smart-Sweep and Smart-SBED start from the same FWHM/3 grid (`SBED_COARSE_POINTS_PER_FWHM = 3`, kept as its own
  constant so it can be tuned separately, e.g. to 4). A one-point-per-FWHM grid often leaves too few points in the
  dip for `find_dips`, so the grid had to refine before the focus appeared.

## Strategies

| Strategy | Acquisition |
|---|---|
| `SimpleSweep-<k>perFWHM`, k = 1..5 | uniform grid over the whole axis with k points per FWHM (spacing FWHM/k), then a least-squares fit. k = 3 is the derived default; the rest are the spacing study |
| `Smart-Sweep` | FWHM/3 sweep; once the data show a dip, sweep only inside the focus (finish FWHM/3 there, then halve the spacing) |
| `Bayesian-SBED` | unchanged SBED |
| `Smart-SBED` | EIG over the not-yet-measured points of a FWHM/3 grid until a dip is found, then ordinary SBED inside the focus |

`Smart-Sweep` and `Smart-SBED` are subclasses of the SBED locator: same belief, convergence/plateau/CRLB stop,
and focus rule (`next_focus_region`), but the focus is re-decided after every measurement (from
`belief.current_dips()`) instead of only at resamples. All sweeps update the same SMC belief every step, so an
uncertainty trajectory exists for every locator. A plain sweep's *reported* estimate is still its final fit.

## Stages

`nvision/metrics/stages.py` splits every run, whatever the locator, into three stages with one detector over the
per-step belief snapshots:

1. **search** -- until `belief.current_dips()` (the deterministic `find_dips`) first reports a dip = *focus found*;
2. **focus** -- until the primary parameter's uncertainty is below its convergence threshold = *max information*;
3. **refine** -- the rest, measuring at maximum sensitivity.

Per run (results-row columns): `focus_found_step`, `max_info_step`, `stage{1,2,3}_steps`,
`stage{1,2,3}_unc_start|unc_end|unc_ratio`. A stage a run never reached is `None` -- a plain sweep that
ends above the threshold has no refine stage, which is a result, not missing data. The Speed tab's "Stages"
section and the `model_comparison` box plots show them.

## Invalidation

Strategy names, cache keys, and the sweep belief all changed: a **full re-run** is required (`nv render` is not
enough). The Sobol locators, the Sobol baseline inside SBED runs, and the "Sobol savings" metric are removed.
