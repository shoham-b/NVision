# Locator comparison: sweeps and SBED

Terms follow the article (`articles/sections/methods.tex`, `sm_algorithm.tex`): the **sweep family** (SimpleSweep)
is what an experimentalist would do without adaptive design, and the question is whether its best member matches
Bayesian SBED. All strategies infer with the same particle filter (so the **claimed uncertainty** $\hat\sigma_N$ means the
same for all), run under the same stop rule (claimed uncertainty below its ceiling for 8 consecutive measurements *and* a
dip found by the **dip test**) and the same 2500-measurement budget; a run that hits the budget is censored and counted
at the budget. Symbol/code-name table: [glossary](glossary.md). In SNR labels, `SNR(x)` is `contrast_total / noise_sigma`,
i.e. the article's $C/\sigma = x/2$.

Every locator searches for the resonance (`zeeman_shift`, the primary parameter); the standard run groups
compare them on the same signals and noise levels. The generators are the ordinary ones, whose priors on the
nuisance parameters are loose (std 10 % of the parameter range). The prior-based FWHM sizing below (sweep
spacing) is therefore less accurate than it would be after a prior classification of the
spectrum with tight nuisance priors; it still uses the prior's width, never the true signal's.

## Run groups

| Group | Signals | Noise | Strategies |
|---|---|---|---|
| `voigt-grid` | Voigt window/FWHM x Gaussian share x isotope (fixed total FWHM) | SNR grid | the six below |
| `drive-control` | CW / pulsed drive-controlled NV x isotope | SNR grid (nominal) | `Bayesian-SBED`, `Bayesian-SBED-Drive` |
| `all` | the two above | the two above | each scenario's own set (`nv groups count all`) |

`nv groups list` shows every group with its task count. The former groups `both-sbed`, `both-sbed-only`,
and `both-sweep-only` still work as aliases of
`voigt-grid` restricted to `Bayesian-SBED` (+ `SimpleSweep-optimal`). To run any other
subset, narrow a group: `--filter-signal` (generator substring) and `--filter-strategy` (comma-separated
regex/substring). Only the sweeps, on every scenario: `--run-group all --filter-strategy Sweep`.

## The grid: SNR, window and line shape, one thing per axis

The `voigt-grid` is posed in the dimensionless numbers that decide how hard a search is, not in contrast, noise
sigma and linewidth separately, and each axis changes one thing:

- **SNR = C / noise sigma** (the article's Eq. snr), C = `contrast_total` / 2 the per-line contrast (each Zeeman group
  carries half of `contrast_total`; the factor is `LINE_CONTRAST_FRACTION` in `sim/presets.py`). Every signal carries the same `contrast_total`
  (`NVISION_SBED_SNR_CONTRAST`, 0.25) and a run is named by its SNR, `SNR(10)`, for the levels in
  `NVISION_SBED_SNR_VALUES` (default 1, 2, 5, 10, 20). The runs are not a grid of noise levels and contrasts: the name
  carries only the SNR, and the Gaussian sigma (`contrast_total / 2 / SNR`) is derived for each signal when the run is
  resolved, so the name keeps its meaning if the contrast constant is changed. Results record it as `snr`. The levels
  are spaced geometrically because the behaviour is: in a 12-repeat scan at a 10-FWHM window, everything converges
  from SNR 20 up (10 vs 50 barely differs), at SNR 10 most runs still converge, at 5 a quarter to three quarters hit
  the budget, and at 2 and below nearly every run does (the sweep fit finds no dip at SNR 1). 1 is the smallest SNR
  worth running.
- **Window span.** The window is a multiple of the width one dip group covers: `window = x * covered`, for `x` in
  `NVISION_SBED_WINDOW_SPANS` (default 1.6, 2, 3, 5, 10, 20, 50; `span<x>` in the generator name). The results and UI
  report the inverse, the signal occupancy `1 / x` (`grid_signal_occupancy`). The narrow windows (up to 5) compare
  the locators once the dip is in view; the wide ones (10-50), where finding the dip takes a sizeable part of the
  budget, carry the search / localize / speedup results. 1.6 is about the tightest window every isotope fits (N-15 needs `x` > 1.572). `covered` is the union of the dip group's hyperfine lines' FWHM intervals
  (ONE total FWHM shared by every signal, `NVISION_SBED_FWHM`, 2 MHz): one line for the unresolved isotope, two for N-15
  (`2 FWHM - overlap`), three for N-14 (`3 FWHM - 2 overlaps`), where lines closer than one FWHM overlap by
  `FWHM - spacing` (none at the default FWHM). So the same span is the same share of signal in view for every
  isotope, but a different absolute window: at span 10 the window is 20 MHz unresolved, 40 MHz N-15, 60 MHz N-14.
  The window changes the window only: the search is hard when the dips are a sizeable part of the window, not only when
  it is orders of magnitude wider than the line. Everything with a physical scale (hyperfine offsets, the absolute
  convergence thresholds, the candidate step) keeps its meaning across the axis. The window is `[D, D + x * covered]`
  and the Zeeman range follows it (`0.4 * window`; the default 150 MHz window has 60 MHz). The generator also records
  `grid_window_fwhm` (window / FWHM) as a derived number. The dip must clear the lower edge by the edge pad
  (`NVISION_NV_ZEEMAN_EDGE_PAD_FWHM`, 0.5 FWHM plus the outermost hyperfine offset), which is what limits the largest
  occupancy per isotope (the unresolved line at 0.5 needs the 0.5 pad).
- **Gaussian share** (`NVISION_SBED_GAUSSIAN_SHARES`, default 0, 0.15, 0.3; `gs<g>` in the name,
  `grid_gaussian_share`): how much of that total FWHM is inhomogeneous (Gaussian) broadening, 0 = pure Lorentzian.
  The Lorentzian HWHM and `sigma_inhom` are set so the total FWHM stays `NVISION_SBED_FWHM`
  (`fwhm = 2 hwhm_lorentz + 2 sqrt(2 ln 2) sigma_inhom`), so the share changes the line shape and not the width.
  Window mostly decides the search and the share mostly the refine regime.
- **Isotope** (`NVISION_SBED_HYPERFINE_VALUES`, default `unresolved`, `n14`, `n15`): `contrast_total` is shared among the Zeeman
  groups and, for a resolved isotope, the hyperfine lines, so one dip is half of it for a merged line and roughly a
  quarter to a third for N-14 or N-15. Windows that cannot fit the isotope's Zeeman padding (e.g. 5 FWHM for 14N) are omitted.
- **Budget.** The same for every window and locator: `NVISION_MAX_MEASUREMENTS` (2500). A run that hits it is
  censored at that count, so censored costs of different windows are on one scale.

4 occupancies x 3 shares x 3 isotopes (36 signals) x 5 SNR x 6 locators = 1080 tasks.

Only the positions follow the window (`zero_field_freq` range, `zeeman_shift` range and its prior std). The width
bounds (`hwhm_lorentz`, `sigma_inhom`, `hyperfine_shift`) are normally sized as a fraction of the window; for these
signals they are sized off the default window instead (`NVCenterCoreGenerator.width_bounds_reference_phys`), so
how well the linewidth is known does not change with `N`. The search grids of the sweeps are
still sized from the prior FWHM, which is much wider than the true one (a prior of about 11 MHz against 2 MHz here),
so at 5 FWHM the coarse stage has only a handful of points.

The previous width x contrast x sigma_inhom grid (`NVCenter-voigt-w<..>MHz-c<..>-si<..>MHz-hf<..>`), the earlier
`fw<N>-` / `occ<x>-` `si<..>-hf<..>` signals and the old `Gauss(sigma)` noise names remain resolvable by name, so cached results
still open.

## Selecting a point

`nv run` with no selection runs the default group (`voigt-grid`). Pick points of it by their numbers:

```bash
uv run nv run --snr 10 --snr 20 --signal-occupancy 0.2 --filter-strategy "SimpleSweep-optimal" --repeats 20
```

`--snr` and `--signal-occupancy` (repeatable) narrow any run group, together with `--filter-signal` and
`--filter-strategy`; `nv groups run <group>` takes them too. A signal with no stated window (`drive-control`) matches
no `--signal-occupancy`. The plain path (random width and contrast per repeat, a Gauss-sigma noise grid) now runs only
when asked for with `--all`, `--filter-category/-generator/-noise`, `--width`, `--contrast`, `--sweep-param`,
`--max-noise-std` or `--noise-steps`.

## Running the comparison

`voigt-grid` has 1080 tasks (36 signals x 5 SNR levels x 6 strategies); narrow it for a first look.

```bash
uv run nv run --run-group voigt-grid --filter-strategy "Sweep,SBED" --repeats 30 --runners 8
uv run nv compare            # stage-by-stage table from artifacts/ (or --out DIR)
uv run nv serve              # View: "Step size" (sweep spacing), "Stages" (stage by stage)
```

`nv compare` prints three tables (accuracy; stage boundaries and reach rates; per-stage duration and
uncertainty ratio) for every locator on the same footing; `--filter-signal`, `--filter-noise`,
`--by generator,noise` and `--csv` refine and export it. `nv groups count voigt-grid` shows the task count.
A quick look: `NVISION_SBED_SNR_VALUES=5 uv run nv run --run-group voigt-grid --filter-signal "span5-gs0.15" --repeats 5`.

In the UI the plain sweeps appear as one "Sweep" entry; choosing it reveals a second dropdown for the spacing
(`optimal` = k*, and its multiples 0.5×k* and 2×k*).

## Width and spacing

The width that sizes everything is the FWHM, `2 * effective_hwhm` of the **prior** (the belief's prior-mean
parameters, clipped to their bounds) -- never the true signal's width (`nvision/sim/locs/search_grid.py`).

* A dip is seen only if several measurements fall inside it. Three points per FWHM
  (`SWEEP_POINTS_PER_FWHM = 3`) cannot step over a dip and keep the dip's position information within 4 % of a
  dense grid (table below); `find_dips` is a boxcar GLRT scan and needs no minimum cluster. 3 is the **reference**
  sweep's grid, not the plain-sweep baseline's step: that is the optimal k*
  (next section). It is derived, not a tuning knob, so it is a module constant and not an env var.


### The optimal sweep step k*

A plain sweep measures a uniform grid of spacing $\Gamma_0/k$ ($\Gamma_0$ the prior FWHM) and repeats whole passes
until the stop rules fire, stopping only at the end of a pass. Two things depend on $k$.

*Ripple (Poisson sum).* The Fisher information about the dip position from grid points $x_i = x_0 + i\Delta$ is
$\sum_i g(x_i)^2$ with $g = \partial\mu/\partial x_0$. By Poisson summation
$\Delta \sum_i g(x_i)^2 = \int g^2\,dx + 2\sum_{m\ge1}\hat{G}(m/\Delta)\cos(2\pi m\,\varphi/\Delta)$, with $\varphi$ the dip's
offset from the grid and $\hat G$ the Fourier transform of $g^2$. For a Lorentzian $\hat G$ decays like
$e^{-\pi\Gamma_0 m/\Delta}$ times a power, so the relative harmonic is small once $\Delta \lesssim \Gamma_0/3$.
Information per measurement relative to a dense grid, worst / best dip offset:

| k | 1 | 1.5 | 2 | 2.5 | 3 | 4 |
|---|---|---|---|---|---|---|
| worst | 0.13 | 0.43 | 0.71 | 0.88 | 0.96 | 1.00 |
| best | 1.37 | 1.48 | 1.28 | 1.12 | 1.04 | 1.00 |

Detection energy per measurement is flat (>= 0.97) from $k \approx 2$. Above $k \approx 2.5$, $k$ therefore changes
the cost only through pass rounding.

*Pass rounding.* One pass is $n(k) \approx k W/\Gamma_0$ measurements and the sweep stops after a whole number of
passes, so it spends $P\,n(k)$ with $P$ an integer. A fractional $k$ lets a whole number of passes land on the
measurement count the stop rules need; at high SNR one pass already overshoots, so a smaller $k$ (accepting some
ripple) is cheaper.

`SimpleSweep-optimal` (`nvision/sim/locs/sweep_optimal.py`) takes the minimiser over a scan of $k \in [0.5, 8]$
(step 0.05) of

$$\mathrm{cost}(k) = \frac1{N_\varphi}\sum_{\varphi} P(k,\varphi)\,n(k),\qquad P = \max(P_\mathrm{prec}, P_\mathrm{det}),$$

$\varphi$ over 8 evenly spaced offsets within one step. $P_\mathrm{prec}$ is the smallest $P$ for which the
Cramer-Rao standard deviations from $P\,F_\mathrm{pass}$ (the belief's own signal model, numerical parameter
derivatives, the actual grid, the true noise $\sigma$) satisfy `target_params_converged` (the stop rule's ceilings and
RMS condition). $P_\mathrm{det}$ is the smallest $P$ for which the expected boxcar score $\sqrt P\,S_1$ of the best
`find_dips` window clears $\Phi^{-1}(1 - p_\mathrm{FA}/N_\mathrm{windows})$ (normal quantile: $\sigma$ is known).
A $k$ with cost above `NVISION_MAX_MEASUREMENTS` $M$ is predicted not to converge within the budget; any reachable $k$
beats it. If no $k$ is reachable, the sweep will stop at its cap of $P_\mathrm{cap} = \max(1, \lfloor M/n\rfloor)$ passes,
so the objective becomes the expected final error there: the $k$ minimising the mean over offsets of the binding
relative uncertainty $s^*(P_\mathrm{cap} F_\mathrm{pass})$ (the factor by which the CRLB stds must shrink for
`target_params_converged` to accept them; precision only). Signal quantities are the belief's prior means, $\sigma$ is the true task noise (an oracle yardstick), and
the realised $k^*$ is recorded per repeat (`sweep_k_opt`, `sweep_k` = m k*, `sweep_passes_pred`, `sweep_cost_pred`).

## Strategies

| Strategy | Acquisition |
|---|---|
| `SimpleSweep-optimal` | uniform grid over the whole axis at the fractional step FWHM/k* that minimises the predicted measurements of its task (previous section), then a least-squares fit. The reference sweep: its repeats are persisted for the scan-figure overlay |
| `SimpleSweep-optimal-x0.5`, `-x2` | the same sweep at m k* points per FWHM, m = 0.5 or 2: a check that k* really is the cheapest step (the cost should rise on both sides). A fixed number of points per FWHM proves nothing about the sweep, because the best step depends on the task |
| `Bayesian-SBED` | unchanged SBED |
All sweeps update the same SMC belief every step, so an uncertainty trajectory exists for every locator. A plain sweep's *reported* estimate is still its final fit.

## Regimes

A run passes the article's regimes: search + localize until its own dip test fires (`dip_detected_step`), converge until
the posterior is a single Gaussian peak, then refine. They are read from each run's own history and judged against the
article's bounds on the Simulation page; [evaluation metrics](evaluation_metrics.md) says how. The former stage model
(boundaries from a 0.7 drop of the claimed σ and an ε tolerance, `focus_found_step`, `refine_start_step`,
`stage{1,2,3}_*`, the ε oracles) was removed on 2026-11-22: it contradicted the article.

## Known deviations (docs/code vs. article)

- **Bound.** The code and several docs say CRLB where the article says van Trees bound; van Trees is used only from each
  run's own detection, with the posterior's J_τ ([evaluation metrics](evaluation_metrics.md)).
- **SimpleSweep spacing.** The article lists $p = 2, 3, 4$; the code now runs the reference $k = 3$ and the optimal
  fractional $k^*$ (`SimpleSweep-optimal`) instead of the $2, 3, 4$ ladder.
- **Sweep reported estimate.** A plain sweep's reported estimate is its final fit while the claimed uncertainty comes from the
  shared belief. A known open bug (the fit returning the domain midpoint) makes sweep accuracy numbers invalid; sweep cost is unaffected.

## Invalidation

Strategy names, cache keys, and the sweep belief all changed: a **full re-run** is required (`nv render` is not
enough). The Sobol locators, the Sobol baseline inside SBED runs, and the "Sobol savings" metric are removed.

## The comparison report

`nv compare` prints, per strategy, the calibration (share within 2 claimed σ), the final error and the dip detection
(share of runs whose dip test fired, median step). The article's claims are on the UI's Simulation page.

```bash
uv run nv compare --filter-signal span5 --filter-noise "SNR(5)"
```
