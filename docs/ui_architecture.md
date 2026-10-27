# UI Architecture — `static/`

The UI is a single-page application with no build step. All files are served directly from `static/` (dev server) or `artifacts/` (static export). Both directories are kept in sync.

Its colours, type, spacing and component markup are documented as a design system in [design_system/README.md](design_system/README.md). Read it before adding or restyling UI.

## File map

| File | Lines | Responsibility |
|---|---|---|
| `index.html` | — | HTML shell and `<script>` load order |
| `styles.css` | — | All CSS |
| `bootstrap.js` | ~100 | Loads `manifest.js` / `settings.js` from disk; populates `window.MANIFEST`, `window.SETTINGS`, `window.NVISION_ASSET_PREFIX` |
| `format-utils.js` | ~70 | Pure formatting helpers (`formatHz`, `escapeHtml`, `formatMetricValue`, etc.) — no DOM or Plotly deps |
| `plotly-utils.js` | ~80 | Plotly CDN loading, JSON/gz fetch, array decode (`_fetchJson`, `_decodePlotlyFigure`, `resolveAssetPath`, etc.) |
| `run-status.js` | ~115 | Run status banner polling + help-toggle accordion buttons |
| `app.js` | ~3950 | `main()` — all scan/Bayesian/comparison UI logic |
| `reload.js` | ~95 | Recalculate button, `r` keyboard shortcut, `init()` entry point that awaits bootstrap then calls `main()` |

## Script load order

```html
<!-- In <head> — run before body renders -->
<script src="bootstrap.js"></script>   <!-- sets window.NVISION_BOOTSTRAP (a Promise) -->
<script src="format-utils.js"></script>
<script src="plotly-utils.js"></script>
<script src="run-status.js"></script>

<!-- At end of <body> — DOM is ready -->
<script src="app.js"></script>         <!-- defines main() -->
<script src="reload.js"></script>      <!-- awaits NVISION_BOOTSTRAP, then calls main() -->
```

`bootstrap.js` must be first because it sets `window.NVISION_BOOTSTRAP`.  
`app.js` must precede `reload.js` because `reload.js` calls `main()`.  
`format-utils.js` and `plotly-utils.js` must precede `app.js` because `main()` calls their globals.

## Global namespace

All files write to `window` globals (no modules, no bundler).

| Global | Set by | Used by |
|---|---|---|
| `window.NVISION_BOOTSTRAP` | `bootstrap.js` | `reload.js` |
| `window.NVISION_ASSET_PREFIX` | `bootstrap.js` | `plotly-utils.js` (`resolveAssetPath`) |
| `window.MANIFEST` | `bootstrap.js` | `app.js` (`main()`) |
| `window.SETTINGS` | `bootstrap.js` | `app.js` (`main()`) |
| `window.RUN_STATUS` | inlined by server | `run-status.js` |
| `resolveAssetPath` | `plotly-utils.js` | `app.js` |
| `ensurePlotly` | `plotly-utils.js` | `app.js` |
| `_fetchJson` | `plotly-utils.js` | `app.js` |
| `_decodePlotlyFigure` | `plotly-utils.js` | `app.js` |
| `formatHz` etc. | `format-utils.js` | `app.js` |
| `renderRunStatusBanner` etc. | `run-status.js` | `app.js` (`initRunStatusBanner`) |
| `main` | `app.js` | `reload.js` |
| `window.NVISION_ALL_GENERATORS` | `bootstrap.js` (live mode only) | `app.js` (generator/study picker) |
| `window.NVISION_LOADED_GENERATORS` | `bootstrap.js` (live mode only) | `app.js` (`nvisionNeedsWiderLoad`) |
| `window.NVISION_GENERATOR_GRID_INFO` | `bootstrap.js` (`/api/generator-grid-info`) | `app.js` (`parseGeneratorFacets`) |
| `window.nvisionGeneratorAxes`, `NVISION_GRID_AXIS_ORDER`/`_INFO` | `bootstrap.js` | `bootstrap.js` and `app.js` (facet/family grouping) |
| `nvisionNeedsWiderLoad`, `nvisionNavigateToGenerator` | `bootstrap.js` | `app.js` (`updateScanSignalControls`) |

## Live manifest fetch is scoped to one generator family

In live mode (`nv serve`), `bootstrap.js` does NOT fetch the whole cache's `/api/manifest` up
front — on a large shared cache (many unrelated generators/noises/strategies/repeats) that full
scan is what made every page load slow, even though any one view only ever looks at one
parameter-study "family" at a time (a family being either a single ungrouped generator, every
generator sharing the same swept-parameter grid coordinates, or — for `nv matlab-run` real-data
generators — the flat "MATLAB (real data)" study bucket). Instead:

1. `bootstrap.js` fetches two cheap indexes in parallel: `/api/combos` (generator/noise/strategy/
   repeat count, no per-repeat scan) to learn every generator name in the cache, and
   `/api/generator-grid-info` (one repeat-0 read per *distinct generator*, not per repeat/combo —
   still cheap even on a huge cache) for each generator's recorded `grid_*` swept-parameter
   coordinates (`nvision/runner/metrics.py`; see `api_server.py`'s `_get_generator_grid_info`).
2. Both `bootstrap.js` (to decide what to fetch) and `app.js` (to build the Study/facet pickers)
   group generators into families via the single shared `window.nvisionGeneratorAxes` function
   (defined in `bootstrap.js`, called by both), which reads a generator's `grid_*` coordinates —
   never the generator *name* string. There used to be two independent name-regex parsers here
   (one per file, required to be kept in sync by hand) before this was fixed to read the actual
   recorded data instead.
3. It picks the initial family to load — the URL hash's `generator` if present and known,
   otherwise the first generator alphabetically — via `_nvisionFamilyScope`, then fetches only
   `/api/manifest?generators=<family members>`, sets `window.MANIFEST` to that, and records the
   loaded scope on `window.NVISION_LOADED_GENERATORS`.

Families are physics-level, not generator-class-level (`window.nvisionGeneratorAxes`): a Lorentzian
width × contrast grid generator joins the Voigt family as its σ_inhom = 0 slice, and `cw_drive` /
`pulsed_drive` generators form one "drive" family with a CW/pulsed axis (`drive_mode`, derived from
`grid_variant`). Because of the merge, the facet selection σ_inhom = 0 + isotope `unresolved` matches both
the `voigt-grid` generator and the `lorentzian-drift` generator, so their repeats are pooled when both exist.

The pickers follow the SNR x window study: the window axis (`window`, labelled "Window (× signal span)": 1 / `grid_signal_occupancy`, so occupancy 10 % shows as window 10 × signal span; older rows without it fall back to `grid_window_fwhm`, "Window (FWHM)") sits first, an
axis with a single value in the family (the study's fixed HWHM and contrast) is listed as "fixed: ..." instead of
getting a one-button picker. A run of the SNR studies is picked by SNR alone: the noise buttons are its `SNR(x)`
names (`SNR x`, ascending), and the study's fixed linewidth and contrast are not axes at all
(`nvisionGeneratorAxes` drops them for a generator with a `grid_signal_occupancy` (or legacy `grid_window_fwhm`), which varies by occupancy and
`grid_gaussian_share` only; `sigma_inhom` just restates the share). An older `Gauss(sigma)` run of a fixed
contrast is labelled by its SNR as well (`nvisionNoiseSnrLabel`).

`app.js`'s generator/study picker (`updateScanSignalControls`) still builds its options from
*every* generator in the cache (`window.NVISION_ALL_GENERATORS`), not just the loaded ones, so
every generator remains selectable. Picking one outside the loaded scope calls
`window.nvisionNavigateToGenerator(name)`, which updates the URL hash and does a full
`location.reload()` — not an in-place re-fetch — because `main()` owns `plots`/`scanPlots`/etc.
as plain `const`s computed once at call time and is only ever invoked once per page load (the
existing recalculate/`r` flow already reloads the whole page for the same reason). A
`sessionStorage`-backed safety valve in `nvisionNavigateToGenerator` still guards against a reload
loop (defense-in-depth against something like a stale `/api/generator-grid-info` cache mid-reload)
even though the single shared grouping function means the two call sites can no longer disagree
the way two independent parsers previously could.

The static-export path (`nv render`, `nv matlab`) is unaffected: `window.MANIFEST` is inlined (or
fetched from `plots_manifest.json.gz`) as a single already-complete array, so `NVISION_ALL_
GENERATORS`/`NVISION_LOADED_GENERATORS` are never set and the picker falls back to deriving its
generator list from `scanPlots` the way it always did.

## `app.js` internal structure

`main()` is a large closure that owns all shared UI state (`currentPlot`, `plots`, `scanIframe`, etc.). Internal functions that depend on this shared state are intentionally kept inside `main()` rather than extracted — they are hoisted function declarations visible throughout the closure.

Key sections inside `main()` (in order of appearance):

1. **Manifest filtering** — splits `plots` array by `type` into typed arrays (`scanPlots`, `bayesPosteriorDataPlots`, etc.)
2. **Global timeline controls** — Bayesian animation sync (play/pause/step slider)
3. **Bayesian tab system** — tab bar build and section switching
4. **Scan controls** — generator, noise, strategy, repeat pickers; Gauss std slider
5. **Scan plot rendering** — `renderPlotFromJson` (calls `ensurePlotly` + `_fetchJson`); measurement distribution legend sync
6. **Comparison cards** — head-to-head scan comparison; trace building for sampled-measurement plot
7. **Bayesian rendering** — posterior heatmap, convergence, metrics, Fisher, covariance ellipses, jitter
8. **Aggregate views** — the `#scan-view-mode` control's modes besides *Run*, each reading per-repeat fields
   for many repeats at once:
   * **Cost & search** (the `locators` mode; `renderStepSizeTab`, `ss*` helpers) — every locator's cost and search metrics side by side, each
     plain-sweep spacing k its own locator, with toggle chips to show or hide each one; nothing is ranked. Reads
     `/api/run-rows` (per-repeat scalars: measurements, final error and claimed σ, focus-found step, contrast,
     noise σ, grid axes) over the loaded family; `/api/sweep-spacings` lists the sweep
     spacings the current code studies, so older spacings left in the cache are hidden. A **use case** is one signal
     at one noise level (window x line shape x SNR in the SNR x window study; generator x noise elsewhere). The
     **table** is per use case: one block per use case of the current Signal / Noise selection (the pickers above
     the view). With several use cases selected the first block is the **median over the use cases** of each use
     case's own number (never read off pooled runs; "runs" is then the per-use-case count), followed by one block per
     use case.
     Per locator it shows the median and p75 cost, the capped share, how often a dip was found, the search cost, and
     whether the claimed σ is right (share of runs with |error| ≤ 1σ / 2σ / 3σ, nominal 68% / 95% / 99.7%). Selecting
     several noise levels or windows gives several blocks. The **charts** ("Across use cases") span the whole family:
     cost and search cost (and, in the **Accuracy** view, the 1σ/2σ/3σ share: "Is the claimed σ right?") against SNR = contrast / noise σ, each point one use case, with a dotted
     vertical line at the SNR of the selected use case in the panel of its window. They follow the SNR x window study:
     x = SNR (log), one panel per signal occupancy (`grid_signal_occupancy`, titled "window 10 × signal span", occupancy 10 % in the tooltip; the absolute window in MHz differs per isotope at one occupancy, so the tooltip lists it per isotope instead of keying the panel on it), lines = locators at ONE
     line shape chosen by a button group (default the selected use case's shape; the shape is named by its Lorentzian
     share of the FWHM, `1 - grid_gaussian_share`, with `hwhm_lorentz` / `sigma_inhom` in the tooltip). "Compare
     shapes" turns the lines into the shapes of one chosen locator. Runs recorded before the occupancy grid
     (only `grid_window_fwhm`) keep one panel per window in FWHM. Runs without either window field /
     `grid_gaussian_share` (older caches) get one panel and no shape controls. Runs stop once σ is below the target,
     so the error at the end is not shown: only whether σ is honest. The locator chips, the line-shape buttons and the
     run counts live in `#ss-controls`, shown above both the Cost & search and the Accuracy view (the σ chart is
     rendered by `renderStepSizeContent`, which `showScanViewPanels` also calls for Accuracy; a chart whose view is
     hidden is skipped).
   * **Stages** (`renderStagesTab`, `renderHlStages`) — the search / focus / refine comparison
     (`docs/locator_comparison.md`), one section per stage, each headed by what drives it and its metric (as in
     `docs/evaluation_metrics.md`), shown one at a time with the **Show:** selector (`hlStageShown`; the other sections are
     collapsed, not `display:none`, so Plotly keeps their width): *Whole run* (over-time curves, measurements per stage), *1. Search* (measurements,
     dip-found share), *2. Focus* (measurements, sensitivity), *3. Refine* (measurements, Fisher efficiency η, claimed
     σ / true error vs the Fisher reference). Each stage has `hl-stage-compare-<n>` (`renderHlStageCompare(..., n)`),
     `hl-stages-steps-<n>` and the claimed-uncertainty / true-error curves over that stage only (`hl-stage-unc-<n>`,
     `hl-stage-err-<n>`; `renderHlStageCurve(..., stage)` draws each run at its own measurement numbers, only between the start and end of the stage). The
     single-repeat panels (`#scan-run-content`) are shown only in the Run view, collapsed the same way elsewhere.
   * **Accuracy** (`renderAccuracyTab`) — whether the stop is reliable: true error at the stop in units of τ,
     claimed σ against the true error over measurements, and claimed σ against the true error.

   Stages and Accuracy have a **scope** control (`#uc-scope-stages` / `#uc-scope-accuracy`, one setting, `ucScope`):
   *Median over selection* (default) or *Each use case*. The markup inside `#stages-body` / `#accuracy-body` is a
   template that `ucRenderBlocks` clones once per block, titled and with every element id suffixed `--uc<i>`; the
   render functions take that suffix (`renderHlStages(gen, noises, sfx)`, `renderHlStopQuality(..., sfx)`). In the
   median scope the selection is ONE block whose generator / noise arguments are arrays. Its numbers are medians over
   the use cases of each use case's own value (`hlUnits`, `hlCaseShares`): table cells, stage bars and the stage
   comparison, and the over-time curves (median of per-use-case medians at each step, shown while at least half the
   use cases are still running; the Cost & search median covers only the use cases every shown locator has runs on).
   The Accuracy stop table and σ-vs-error chart and the Stages refine curves pool the runs of the selection instead
   (they are in units of τ / σ or relative to the refine start); the Accuracy pace chart (absolute Hz against
   measurements) is shown only per use case. The controls that are one setting for the view (Stages "Show:", the
   Accuracy deviation-band toggle) redraw every block. New elements inside either body must therefore be looked up
   by `id + sfx`.
   * **Failed Runs** (`renderFailedRunsTab`).

   Stages and Accuracy read `series`/`true_params`/stage fields from `/api/scan-fields` (`ensureScanFieldsFor`),
   scoped to the current generator/noise selection.
9. **Narrowed bounds panel** — reads `layout.meta.narrowed_param_bounds` from the scan figure
10. **Tabs setup** — top-level tab bar (just "Scan measurements")
11. **Event listeners** — wires all control-change and click events

## Data formats

### `plots_manifest.json`

Array of plot-entry objects. Each entry has at minimum:

```json
{ "type": "scan", "generator": "NVCenter-lorentzian", "noise": "Gauss(0.0067)",
  "strategy": "Bayesian-SMC", "repeat": 0, "path": "artifacts/scan_...json.gz" }
```

Plot files are `.json.gz` (gzip-compressed JSON). Numeric arrays are encoded as `{"__f32__": "<base64>"}` (our custom format) or `{"bdata": "<base64>", "dtype": "float64"}` (Plotly Python 5.x).

Both formats are decoded by `_decodePlotlyFigure` in `plotly-utils.js`.

### Bayesian data files

| `type` | Schema key | Content |
|---|---|---|
| `bayesian_posterior_data` | `posterior_v1` | Per-step 2-D posterior heatmap arrays |
| `bayesian_parameter_convergence_data` | `param_convergence_v1` | Per-step parameter uncertainty time-series |
| `bayesian_convergence_metrics_data` | `convergence_metrics_v1` | Convergence streak, patience, per-parameter status |
| `bayesian_fisher_data` | `fisher_v1` | CRLB bounds and uncertainty comparison |
| `bayesian_covariance_ellipses_data` | `covariance_ellipses_v1` | 2-D ellipse parameters per step |

## Artifacts sync

After every change to `static/`, mirror to `artifacts/`:

```powershell
@("bootstrap.js","run-status.js","format-utils.js","plotly-utils.js","app.js","reload.js","index.html","styles.css") |
    ForEach-Object { Copy-Item "static\$_" "artifacts\$_" -Force }
```
