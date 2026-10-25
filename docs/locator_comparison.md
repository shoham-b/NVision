# Locator comparison: sweeps, Smart-Sweep, Smart-SBED

Every locator searches for the resonance (`zeeman_split`, the primary parameter); the standard run groups
compare them on the same signals and noise levels. The generators are the ordinary ones, whose priors on the
nuisance parameters are loose (std 10 % of the parameter range). The prior-based FWHM sizing below (sweep
spacing, Smart-* search grid) is therefore less accurate than it would be after a prior classification of the
spectrum with tight nuisance priors; it still uses the prior's width, never the true signal's.

## Run groups

| Group | Signals | Noise | Strategies |
|---|---|---|---|
| `voigt-grid` | Voigt window/FWHM x Gaussian share (fixed total FWHM, isotope) | SNR grid | the six below |
| `drive-control` | CW / pulsed drive-controlled NV | SNR grid (nominal) | `Bayesian-SBED`, `Bayesian-SBED-Drive` |
| `all` | the two above | the two above | each scenario's own set (`nv groups count all`) |

`nv groups list` shows every group with its task count. The former groups `both-sbed`, `both-sbed-only`,
and `both-sweep-only` still work as aliases of
`voigt-grid` restricted to `Bayesian-SBED` (+ `SimpleSweep-3perFWHM`). To run any other
subset, narrow a group: `--filter-signal` (generator substring) and `--filter-strategy` (comma-separated
regex/substring). Only the sweeps, on every scenario: `--run-group all --filter-strategy Sweep`.

## The grid: SNR, window and line shape, one thing per axis

The `voigt-grid` is posed in the dimensionless numbers that decide how hard a search is, not in contrast, noise
sigma and linewidth separately, and each axis changes one thing:

- **SNR = `contrast_total` / noise sigma.** Every signal carries the same `contrast_total`
  (`NVISION_SBED_SNR_CONTRAST`, 0.25) and a run is named by its SNR, `SNR(10)`, for the levels in
  `NVISION_SBED_SNR_VALUES` (default 1, 2, 5, 10, 20). The runs are not a grid of noise levels and contrasts: the name
  carries only the SNR, and the Gaussian sigma (`contrast_total / SNR`) is derived for each signal when the run is
  resolved, so the name keeps its meaning if the contrast constant is changed. Results record it as `snr`. The levels
  are spaced geometrically because the behaviour is: in a 12-repeat scan at a 10-FWHM window, everything converges
  from SNR 20 up (10 vs 50 barely differs), at SNR 10 most runs still converge, at 5 a quarter to three quarters hit
  the budget, and at 2 and below nearly every run does (the sweep fit finds no dip at SNR 1). 1 is the smallest SNR
  worth running.
- **Window / FWHM.** The window is `N` times ONE total FWHM shared by every signal (`NVISION_SBED_FWHM`, 2 MHz), for
  `N` in `NVISION_SBED_WINDOW_FWHM_VALUES` (default 5, 10, 20; `fw<N>` in the generator name, `grid_window_fwhm` in
  the results). So a window is the same absolute 10, 20 or 40 MHz whatever the line shape, and the window changes the
  window only. Wider windows add nothing the study needs: the search is hard when the dips are a sizeable part of the
  window, not only when it is orders of magnitude wider than the line. Everything with a physical scale (hyperfine
  offsets, the absolute convergence thresholds, the candidate step) keeps its meaning across the axis. The window is
  `[D, D + N * FWHM]` and the Zeeman range follows it (`0.4 * window`; the default 150 MHz window has 60 MHz).
- **Gaussian share** (`NVISION_SBED_GAUSSIAN_SHARES`, default 0, 0.15, 0.3; `gs<g>` in the name,
  `grid_gaussian_share`): how much of that total FWHM is inhomogeneous (Gaussian) broadening, 0 = pure Lorentzian.
  The Lorentzian HWHM and `sigma_inhom` are set so the total FWHM stays `NVISION_SBED_FWHM`
  (`fwhm = 2 hwhm_lorentz + 2 sqrt(2 ln 2) sigma_inhom`), so the share changes the line shape and not the width.
  Window mostly decides the search stage and the share mostly the refine stage; the stage metrics
  (`nv compare`, the Stages view) attribute a cost to the stage it came from.
- **Isotope** is held fixed (`NVISION_SBED_STUDY_ISOTOPE`, `unresolved`): `contrast_total` is shared among the Zeeman
  groups and, for a resolved isotope, the hyperfine lines, so one dip is half of it for a merged line and roughly a
  quarter to a third for N-14 or N-15, and a changing isotope would change the effective SNR.
- **Budget.** The same for every window and locator: `NVISION_MAX_MEASUREMENTS` (2500). A run that hits it is
  censored at that count, so censored costs of different windows are on one scale.

3 windows x 3 shares x 5 SNR x 6 locators = 270 tasks.

Only the positions follow the window (`zero_field_freq` range, `zeeman_shift` range and its prior std). The width
bounds (`hwhm_lorentz`, `sigma_inhom`, `hyperfine_shift`) are normally sized as a fraction of the window; for these
signals they are sized off the default window instead (`NVCenterCoreGenerator.width_bounds_reference_phys`), so
how well the linewidth is known does not change with `N`. The search grids of the sweeps and Smart-* locators are
still sized from the prior FWHM, which is much wider than the true one (a prior of about 11 MHz against 2 MHz here),
so at 5 FWHM the coarse stage has only a handful of points.

The previous width x contrast x sigma_inhom grid (`NVCenter-voigt-w<..>MHz-c<..>-si<..>MHz-hf<..>`), the earlier
`fw<N>-si<..>-hf<..>` signals and the old `Gauss(sigma)` noise names remain resolvable by name, so cached results
still open.

## Selecting a point

`nv run` with no selection runs the default group (`voigt-grid`). Pick points of it by their numbers:

```bash
uv run nv run --snr 10 --snr 20 --window-fwhm 5 --filter-strategy "Smart-Sweep" --repeats 20
```

`--snr` and `--window-fwhm` (repeatable) narrow any run group, together with `--filter-signal` and
`--filter-strategy`; `nv groups run <group>` takes them too. A signal with no stated window (`drive-control`) matches
no `--window-fwhm`. The plain path (random width and contrast per repeat, a Gauss-sigma noise grid) now runs only
when asked for with `--all`, `--filter-category/-generator/-noise`, `--width`, `--contrast`, `--sweep-param`,
`--max-noise-std` or `--noise-steps`.

## Running the comparison

`voigt-grid` has 270 tasks (9 signals x 5 SNR levels x 6 strategies); narrow it for a first look.

```bash
uv run nv run --run-group voigt-grid --filter-strategy "Sweep,SBED" --repeats 30 --runners 8
uv run nv compare            # stage-by-stage table from artifacts/ (or --out DIR)
uv run nv serve              # View: "Step size" (sweep spacing), "Stages" (stage by stage)
```

`nv compare` prints three tables (accuracy; stage boundaries and reach rates; per-stage duration and
uncertainty ratio) for every locator on the same footing; `--filter-signal`, `--filter-noise`,
`--by generator,noise` and `--csv` refine and export it. `nv groups count voigt-grid` shows the task count.
A quick look: `NVISION_SBED_SNR_VALUES=5 uv run nv run --run-group voigt-grid --filter-signal "fw10-gs0.15" --repeats 5`.

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
  `find_dips`; a grid that misses the dip is swept again.
* Smart-Sweep's focus sweep uses a finer grid, FWHM/10 (`FOCUS_SWEEP_POINTS_PER_FWHM = 10`), laid fresh inside each
  focus section from its lower edge. Lower bound, derived: a measurement eps from the max-information point keeps
  eta ~ 1 - 4.5 (eps/HWHM)^2 of the optimal information, so eta >= 0.9 (the refine switch's
  `SMART_SWEEP_REFINE_INFO_LOSS`) needs eps <= 0.15 HWHM, i.e. spacing 2 eps = FWHM/6.7, so 7 points per FWHM.
  At 7, one pass over the focus occasionally ended before the posterior allowed the refine switch (1 of 8 test
  runs), so 10 was chosen empirically (0 of 8). A grid that is fully measured is swept again at the same spacing, never refined and never an error (the stop rules end the run).

## Strategies

| Strategy | Acquisition |
|---|---|
| `SimpleSweep-<k>perFWHM`, k = 2..4 | uniform grid over the whole axis with k points per FWHM (spacing FWHM/k), then a least-squares fit. k = 3 is the derived default; the rest are the spacing study |
| `Smart-Sweep` | FWHM/2 sweep; once the data show a dip, sweep only inside the focus on a FWHM/10 grid, then measure at the max-information points once the refine rule fires |
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
NVISION_SBED_SNR_VALUES=5 uv run nv run --run-group voigt-grid --filter-signal fw10 --repeats 3 --runners 2 --out scratch/x
uv run nv compare --out scratch/x --filter-signal fw10 --filter-noise "SNR(5)"
uv run nv serve --dir scratch/x   # Stages view, a fw10 generator, noise SNR(5), compare the tables
```
