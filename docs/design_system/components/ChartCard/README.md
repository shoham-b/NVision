# ChartCard

The white card that holds one analysis chart in the Speed, Stages and Accuracy views. Cards usually come in side-by-side pairs, under uppercase group labels.

**Markup.**
- Wrap the view in `<div class="view-panel">`, with an `h2` title ("Speed — Measurement Efficiency").
- Each group starts with `<h3 class="group-label">Over time</h3>`.
- Two charts side by side: `<div class="chart-pair">` holding two `<div class="chart-card">`.
- One full-width chart: `<div class="chart-card chart-card-full">`.
- Inside a card: an `h3` title ending in a `HelpIcon`, then the chart `div` with an explicit height (340–420px). An optional `h4` sub-chart heading goes under it.
- Diagnostic mini-charts (SMC weights, ESS) take an `h4.diagnostic-label` above a 180px chart `div`.

**Tokens.**
- Card: `surface`, 1px `slate-100`, `radius-lg`, 10px padding. Its `h3` is indented 10px.
- Pair: `flex: 1 1 46%`, min-width 420px, gap `space-6`. It wraps to one column under about 900px.
- Group label: `type-group-label` in `slate-600`, uppercase.
- View panel: 1px `divider-soft` top rule, 1em padding.

**Consumer provides.** The chart title (a question or "Y vs X"), a help tooltip saying how to read the chart, and the Plotly target `div` with its height.

- **Do** title charts as the question they answer ("Was the stop right?", "Did σ fall at the right pace?") or as "Y vs X".
- **Don't** put two unrelated charts in one pair. A pair compares two views of the same question.

Named from inline styles in `static/index.html` (`#speed-view`, `#stages-view`, `#accuracy-view`); the source has no class for it.
