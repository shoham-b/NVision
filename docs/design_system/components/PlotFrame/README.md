# PlotFrame

The bordered container a Plotly chart renders into, and the trace colour conventions every NVision chart follows.

**Markup.**
- A Plotly JSON plot: `<div class="plotly-json-plot" id="…-json-div">`, min-height 500px.
- An embedded HTML plot: the source styles a bare `iframe` (height 600px). Here use `.plot-frame .plot-frame-html`.
- The comparison view uses `.head-to-head-plot` (480px, `radius-md`).
- Per-parameter posteriors sit in `.posterior-card-grid > .posterior-card`, two-up, each min 480px.

**Frame tokens.** 1px `border-control`, `radius-xs` and a `surface` fill. The frame sits `space-6` below the controls.

**Plotly layout** (from `static/graphs/*.json`):
- `paper_bgcolor`/`plot_bgcolor` white, axis `gridcolor` `chart-grid`, `hovermode: "closest"`.
- A horizontal legend above the plot (`y: 1.02`, `yanchor: "bottom"`).
- `drive_freq` axes use `tickformat: "~s"`. Axis titles are the code's own names, e.g. `drive_freq (Hz)`, `pl_normalized`.
- In-plot annotation boxes are `chart-legend-bg`.

**Trace meanings are fixed.**

| Trace | Colour | Style |
|---|---|---|
| true signal | `trace-true-signal` | solid, width 2 |
| simulated noisy signal | `trace-noisy` | dotted; ±1σ/±2σ fills `trace-noisy-band-1/-2` |
| locator most likely signal | `trace-mode` | dashed, width 2 |
| sweep most likely signal | `trace-sweep` | dotted, width 2 |
| measurements by phase | `trace-coarse`, `trace-secondary`, `trace-tertiary` (edges `-edge`) | markers, size 7 |
| measurements by step | Viridis, edge `trace-marker-edge` | markers, size 8 |
| SMC resample | `orange-500` | dotted vertical line |

**Series.**
- Colour per-parameter panels by `series-1…7` in parameter order.
- Colour Highlights strategies by `highlight-1…8` in strategy order.
- Colour failed-run reasons by `reason-*`.
- Never reassign a slot between charts on the same page.

The preview is a static SVG rendition of a scan plot; the real charts are Plotly. Hand-written from `static/styles.css` `iframe`, `.plotly-json-plot`, `.posterior-card` and `static/graphs/scan.json`.
