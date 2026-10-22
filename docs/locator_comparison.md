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
| `drive-control` | CW / pulsed drive-controlled NV | SBED noise grid | `Bayesian-SBED`, `Bayesian-SBED-Drive` |
| `all` | the two above | the two above | each scenario's own set (`nv groups count all`) |

`nv groups list` shows every group with its task count. The former groups `both-sbed`, `both-sbed-only`,
and `both-sweep-only` still work as aliases of
`voigt-grid` restricted to `Bayesian-SBED` (+ `SimpleSweep-3perFWHM`). To run any other
subset, narrow a group: `--filter-signal` (generator substring) and `--filter-strategy` (comma-separated
regex/substring). Only the sweeps, on every scenario: `--run-group all --filter-strategy Sweep`.

## Running the comparison

`voigt-grid` has 18000 tasks (75 signals x 6 noise levels x 8 strategies); narrow it for a first look.

```bash
uv run nv run --run-group voigt-grid --filter-strategy "Sweep,SBED" --repeats 30 --runners 8
uv run nv compare            # stage-by-stage table from artifacts/ (or --out DIR)
uv run nv serve              # View: "Step size" (sweep spacing), "Stages" (stage by stage)
```

`nv compare` prints three tables (accuracy; stage boundaries and reach rates; per-stage duration and
uncertainty ratio) for every locator on the same footing; `--filter-signal`, `--filter-noise`,
`--by generator,noise` and `--csv` refine and export it. `nv groups count voigt-grid` shows the task count.
A quick look: `NVISION_SBED_NOISE_STEPS=1 uv run nv run --run-group voigt-grid --filter-signal "c0.25" --repeats 5`.

In the UI the plain sweeps appear as one "Sweep" entry; choosing it reveals a second dropdown for the spacing
(k points per FWHM, 2-4).

## Width and spacing

The width that sizes everything is the FWHM, `2 * effective_hwhm` of the **prior** (the belief's prior-mean
parameters, clipped to their bounds) -- never the true signal's width (`nvision/sim/locs/search_grid.py`).

* A dip is seen only if several measurements fall inside it. Three points per FWHM
  (`SWEEP_POINTS_PER_FWHM = 3`) cannot step over a dip and give `find_dips` the cluster it needs. It is derived,
  not a tuning knob, so it is a module constant and not an env var.
* Smart-Sweep and Smart-SBED start from one shared coarse grid, FWHM/2 (`COARSE_SEARCH_POINTS_PER_FWHM = 2`),
  so they differ only in the acquisition after the dip is seen. 2 was chosen from the plain-sweep spacing study
  (k = 2 had the lowest median and p75 cost). A one-point-per-FWHM grid often leaves too few points in the dip for
  `find_dips`; a grid that misses the dip refines by halving.
* Smart-Sweep's focus sweep uses a finer grid, FWHM/7 (`FOCUS_SWEEP_POINTS_PER_FWHM = 7`). Derived: a measurement
  eps from the max-information point keeps eta ~ 1 - 4.5 (eps/HWHM)^2 of the optimal information, so eta >= 0.9
  (the refine switch's `SMART_SWEEP_REFINE_INFO_LOSS`) needs eps <= 0.15 HWHM, i.e. spacing 2 eps = FWHM/6.7,
  rounded up to 7 points per FWHM. A grid point is then always near-optimal until the refine rule fires.

## Strategies

| Strategy | Acquisition |
|---|---|
| `SimpleSweep-<k>perFWHM`, k = 2..4 | uniform grid over the whole axis with k points per FWHM (spacing FWHM/k), then a least-squares fit. k = 3 is the derived default; the rest are the spacing study |
| `Smart-Sweep` | FWHM/2 sweep; once the data show a dip, sweep only inside the focus on a FWHM/7 grid, then measure at the max-information points once the refine rule fires |
| `Bayesian-SBED` | unchanged SBED |
| `Smart-SBED` | EIG over the not-yet-measured points of a FWHM/2 grid until a dip is found, then ordinary SBED inside the focus |

`Smart-Sweep` and `Smart-SBED` are subclasses of the SBED locator: same belief, uncertainty-convergence stop,
and focus rule (`next_focus_region`), but the focus is re-decided after every measurement (from
`belief.current_dips()`) instead of only at resamples. All sweeps update the same SMC belief every step, so an
uncertainty trajectory exists for every locator. A plain sweep's *reported* estimate is still its final fit.

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
known-nuisance scenario -- with free nuisances the diagonal term is an upper bound on the efficient information).
For drive-controlled models the oracle also picks the drive amplitude (and the true pi-pulse when pulsed). $\eta$
uses the true signal, not the belief (whose own precision is overconfident), so it is simulation-only ground truth.

Per run (results-row columns): `focus_found_step`, `max_info_step`, `stage{1,2,3}_steps`,
`stage{1,2,3}_unc_start|unc_end|unc_ratio|sens_eff` (shot-weighted mean $\eta$ per stage). A stage a run never reached is `None` -- a plain sweep, which
keeps moving across the dip, has no refine stage, which is a result, not missing data. The UI's "Stages" view
shows them.

## Invalidation

Strategy names, cache keys, and the sweep belief all changed: a **full re-run** is required (`nv render` is not
enough). The Sobol locators, the Sobol baseline inside SBED runs, and the "Sobol savings" metric are removed.

## The comparison report in the web UI

`nv compare` has a counterpart in the results UI: the *Stages* view lists the same three tables (accuracy,
stage boundaries, per-stage measurements and uncertainty ratio) for the selected generator(s) and noise levels,
one row per strategy (each `SimpleSweep-<k>perFWHM` spacing is its own row). The per-run fields come from
`/api/scan-fields` (`compare`), and runs are filtered as in `metrics/compare.py` `usable_runs`.

Manual check that UI and CLI agree on a tiny run:

```bash
NVISION_SBED_NOISE_STEPS=1 uv run nv run --run-group voigt-grid --filter-signal c0.25 --repeats 3 --runners 2 --out scratch/x
uv run nv compare --out scratch/x --filter-signal w2.75MHz --filter-noise "Gauss(0.01)"
uv run nv serve --dir scratch/x   # Stages view, generator w2.75MHz-c0.25, noise Gauss(0.01), compare the tables
```
