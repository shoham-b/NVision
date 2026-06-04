# UI Architecture — `static/`

The UI is a single-page application with no build step. All files are served directly from `static/` (dev server) or `artifacts/` (static export). Both directories are kept in sync.

## File map

| File | Lines | Responsibility |
|---|---|---|
| `index.html` | — | HTML shell and `<script>` load order |
| `styles.css` | — | All CSS |
| `bootstrap.js` | ~100 | Loads `manifest.js` / `settings.js` from disk; populates `window.MANIFEST`, `window.SETTINGS`, `window.NVISION_ASSET_PREFIX` |
| `format-utils.js` | ~70 | Pure formatting helpers (`formatFrequency`, `escapeHtml`, `formatMetricValue`, etc.) — no DOM or Plotly deps |
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
| `formatFrequency` etc. | `format-utils.js` | `app.js` |
| `renderRunStatusBanner` etc. | `run-status.js` | `app.js` (`initRunStatusBanner`) |
| `main` | `app.js` | `reload.js` |

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
8. **Summary view** — repeats-summary charts, noise-metrics iframes, milestone plots
9. **Narrowed bounds panel** — reads `layout.meta.narrowed_param_bounds` from the scan figure
10. **Tabs setup** — top-level tab bar (Scan / Noise Metrics)
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
