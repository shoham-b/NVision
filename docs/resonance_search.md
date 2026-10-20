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

The pre-run CRLB feasibility gate uses the SBED step budget for every strategy so the same repeats are skipped
for all of them.

## Stages

`nvision/metrics/stages.py` splits every run, whatever the locator, into three stages with one detector over the
per-step belief snapshots:

1. **search** -- up to the measurement after which `belief.current_dips()` (the deterministic `find_dips`) first
   reports a dip = *focus found*;
2. **focus** -- until the measurements gain (nearly) the Cramer-Rao-optimal information every step = *max information*;
3. **refine** -- the rest.

**Fisher-information efficiency.** A measurement at drive frequency $x$ (drive $d$, $k$ shots) carries
$I = k\,g(x,d)^2/\sigma^2$ about the primary parameter $\theta$, with $g = \partial\,\text{signal}/\partial\theta$ on
the **true** signal. An oracle that knows $\theta$ spends the same shots at the best point, $I^* = k \max g^2/\sigma^2$.
The per-step efficiency is $\eta = g(x,d)^2/\max g^2 \in [0, 1]$: 1 at the steepest point of the dip, ~0 at the dip
centre or on the flat baseline.

**Rule.** Refine starts at the first measurement $n$ after the focus from which

$$\sum_{t\ge n} k_t\,\eta_t \;\ge\; (1-\delta) \sum_{t\ge n} k_t, \qquad \delta = \texttt{REFINE\_INFO\_LOSS} = 0.1 .$$

Measurements are independent given $\theta$, so Fisher information adds and the CRLB on $\theta$ from the refine
stage is $1/\sum_{t\ge n} I_t$. The rule therefore **guarantees** the refine stage's CRLB is within $1/(1-\delta)$
(about 1.11x) of the best any design could reach with the same measurements. A steady information rate alone is not
enough: a locator stuck at a weak point is steady too, at low $\eta$. The tail mean tolerates an occasional
exploratory probe instead of resetting on it.

**"Near" the optimum is enough.** The locator aims at the optimum of its *estimate*, off by its error $\epsilon$.
The optimum is a maximum of $\eta$, so the loss is second order. For a Lorentzian of HWHM $\gamma$ the optimum is
$|x-x_0| = \gamma/\sqrt3$ and $\eta \approx 1 - 4.5(\epsilon/\gamma)^2$, so $\delta = 0.1$ accepts measuring within
about $\pm 0.15\,\gamma$ of it -- the locally linear regime.

Assumptions: additive noise with $\sigma$ independent of $x$ and the drive; nuisance parameters known (the
resonance-search scenario -- with free nuisances the diagonal term is an upper bound on the efficient information).
For drive-controlled models the oracle also picks the drive amplitude (and the true pi-pulse when pulsed). $\eta$
uses the true signal, not the belief (whose own precision is overconfident), so it is simulation-only ground truth.

Per run (results-row columns): `focus_found_step`, `max_info_step`, `stage{1,2,3}_steps`,
`stage{1,2,3}_unc_start|unc_end|unc_ratio|sens_eff` (shot-weighted mean $\eta$ per stage). A stage a run never reached is `None` -- a plain sweep, which
keeps moving across the dip, has no refine stage, which is a result, not missing data. The Speed tab's "Stages"
section and the `model_comparison` box plots show them.

## Invalidation

Strategy names, cache keys, and the sweep belief all changed: a **full re-run** is required (`nv render` is not
enough). The Sobol locators, the Sobol baseline inside SBED runs, and the "Sobol savings" metric are removed.

## The comparison report in the web UI

`nv compare` has a counterpart in the results UI: Speed tab → *Stages* lists the same three tables (accuracy,
stage boundaries, per-stage measurements and uncertainty ratio) for the selected generator(s) and noise levels,
one row per strategy (each `SimpleSweep-<k>perFWHM` spacing is its own row). The per-run fields come from
`/api/scan-fields` (`compare`), and runs are filtered as in `metrics/compare.py` `usable_runs`.

Manual check that UI and CLI agree on a tiny run:

```bash
NVISION_SBED_NOISE_STEPS=1 uv run nv run --run-group resonance-search --filter-signal c0.25 --repeats 3 --runners 2 --out scratch/x
uv run nv compare --out scratch/x --filter-signal w2.75MHz --filter-noise "Gauss(0.01)"
uv run nv serve --dir scratch/x   # Speed tab, generator w2.75MHz-c0.25, noise Gauss(0.01), compare the tables
```
