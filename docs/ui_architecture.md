# UI Architecture — `static/`

The UI is a single-page application with no build step. All files are served directly from `static/` (dev server) or `artifacts/` (static export). Both directories are kept in sync.

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
   * **Step size** (`renderStepSizeTab`, `ss*` helpers) — chooses the plain sweep's spacing. Reads
     `/api/run-rows` (per-repeat scalars: measurements, final error, stop target `tau`, focus-found step, grid
     axes) for the spacings `/api/sweep-spacings` lists, over every signal and noise level of the loaded family or
     only the current selection: per-spacing cost, cost relative to the best spacing in the same cell, error at
     the stop in units of τ, and the search stage.
   * **Stages** (`renderStagesTab`, `renderHlStages`) — the search / focus / refine comparison
     (`docs/locator_comparison.md`).
   * **Accuracy** (`renderAccuracyTab`) — whether the stop is reliable: true error at the stop in units of τ,
     claimed σ against the true error over measurements, and claimed σ against the true error.
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
