# Resonance search after a field change

## Scenario

The common ODMR situation: the spectrum was already classified (linewidth, contrast, hyperfine), then the
magnetic field changed, so the resonance moved. The nuisance parameters are expected to change little; the
unknown is where the dip is now (`zeeman_split`, the primary parameter).

`NVCenterCoreGenerator(resonance_search=True)` builds that case: every prior except `zeeman_split` is tight
(std scaled by `RESONANCE_SEARCH_NUISANCE_STD_SCALE`, 0.25, in `nvision/sim/gen/nv_center_generator.py`) and
`zeeman_split` has no prior (flat over its bounds). `presets.resonance_search_generators()` is the width x
contrast grid (names end in `-rs`); the run group is `resonance-search`.

`--filter-signal` (generator substring) and `--filter-strategy` (comma-separated regex/substring) narrow a run group.

## Running the study

Full grid: 25 signals (5 widths x 5 contrasts) x 6 noise levels x 8 strategies = 1200 tasks.

```bash
uv run nv run --run-group resonance-search --repeats 30 --runners 8
uv run nv compare            # stage-by-stage table from artifacts/ (or --out DIR)
uv run nv serve              # Speed tab -> "Stages"; Compare A/B pickers (Sweep -> spacing)
```

`nv compare` prints three tables (accuracy; stage boundaries and reach rates; per-stage duration and
uncertainty ratio) for every locator on the same footing; `--filter-signal`, `--filter-noise`,
`--by generator,noise` and `--csv` refine and export it. `nv groups count resonance-search` shows the task count.
A quick look: `NVISION_SBED_NOISE_STEPS=1 uv run nv run --run-group resonance-search --filter-signal "c0.25-rs" --repeats 5`.

In the UI the plain sweeps appear as one "Sweep" entry; choosing it reveals a second dropdown for the spacing
(k points per FWHM, 1-5).

## Width and spacing

The width that sizes everything is the FWHM, `2 * effective_hwhm` of the **prior** (the belief's prior-mean
parameters, clipped to their bounds) -- never the true signal's width (`nvision/sim/locs/search_grid.py`).

* A dip is seen only if several measurements fall inside it. Three points per FWHM
  (`SWEEP_POINTS_PER_FWHM = 3`) cannot step over a dip and give `find_dips` the cluster it needs. It is derived,
  not a tuning knob, so it is a module constant and not an env var.
* SBED chooses *which* point to measure, so its coarse search grid needs one candidate per FWHM
  (`SBED_COARSE_POINTS_PER_FWHM = 1`).

## Strategies

| Strategy | Acquisition |
|---|---|
| `SimpleSweep-<k>perFWHM`, k = 1..5 | uniform grid over the whole axis with k points per FWHM (spacing FWHM/k), then a least-squares fit. k = 3 is the derived default; the rest are the spacing study |
| `Smart-Sweep` | FWHM/3 sweep; once the data show a dip, sweep only inside the focus (finish FWHM/3 there, then halve the spacing) |
| `Bayesian-SBED` | unchanged SBED |
| `Smart-SBED` | EIG over the not-yet-measured points of a FWHM grid until a dip is found, then ordinary SBED inside the focus |

`Smart-Sweep` and `Smart-SBED` are subclasses of the SBED locator: same belief, convergence/plateau/CRLB stop,
and focus rule (`next_focus_region`), but the focus is re-decided after every measurement (from
`belief.current_dips()`) instead of only at resamples. All sweeps update the same SMC belief every step, so an
uncertainty trajectory exists for every locator. A plain sweep's *reported* estimate is still its final fit.

The pre-run CRLB feasibility gate uses the SBED step budget for every strategy so the same repeats are skipped
for all of them.

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
