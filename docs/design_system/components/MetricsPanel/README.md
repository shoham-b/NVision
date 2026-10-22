# MetricsPanel

A grid of small metric cards summarising a run: an uppercase label, a big monospace value and an optional formula. A 2px coloured rule on top of each card says which family it belongs to.

**Markup.** The panel is `.sweep-metrics-panel` (sweep, blue), `.scan-metrics-panel` (scan, violet) or `.bayes-jitter-panel .jitter-cards` (jitter, violet). Each card is a `.metric-item` holding:
- `.metric-header`: a `.metric-label` and an optional `.help-icon`
- `.metric-value`
- an optional `.metric-formula`, or a `table.metric-mini-table` with `th` labels and `td` values

**Variants.**
- `efficiency-good`, `efficiency-medium`, `efficiency-bad` grade a sweep efficiency.
- `expected-card` is the ground-truth value, on green.
- `err-medium-card` and `err-high-card` grade an error.
- Mini-table rows take `err-medium-row` or `err-high-row`.
- `.scan-metrics-panel.summary-grid` is a fixed 3-column grid (1 column under 850px).

**Tokens.**
- Panel: `slate-50` ground, 1px `slate-200`, `radius-xl`, 0.8em base size (`type-small`), `space-2` gap, `minmax(120px, 1fr)` columns.
- Card: `surface`, `radius-lg`, `shadow-card`. On hover it lifts 1px with `shadow-card-hover` (or `-violet`) and a `blue-200`/`violet-200` border.
- Label: `type-metric-label` in `slate-500`.
- Value: `type-metric-value` in `blue-800` or `violet-800`.
- Formula: `type-metric-formula` in `slate-400`, decorative only (2.56:1).
- Variant values: `green-700`, `orange-700` and `red-700` for efficiency; `red-600` and `red-900` for error.

**Consumer provides.** Label, value with unit, the variant class chosen from a threshold, and an optional help tooltip.

- **Do** put the unit in the value ("84 kHz").
- **Do** let the word carry the meaning as well as the colour.
- **Don't** mix sweep and scan cards in one panel.

Hand-written from `static/styles.css` `.sweep-metrics-panel`, `.scan-metrics-panel`, `.bayes-jitter-panel`. The source repeats the card rules per panel; `bundle.css` shares them with `:is()`.
