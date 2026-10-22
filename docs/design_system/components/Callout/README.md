# Callout

Blue-ruled blocks of explanatory text placed next to the chart they explain.

**Variants.**
- `.explanation-panel`: a titled "how to read this" box with an `h4` and an optional `.explanation-grid` of `.explanation-item`s (a `<strong>` term, then the explanation).
- `.covariance-explainer`: a one-paragraph blurb, hidden until its `?` toggle is pressed.
- `.resampling-summary`: a frosted summary row prefixed with `↺`.
- `.scan-view-at-caption`: a quiet caption under the scan plot naming the step being viewed.

**Tokens.**
- Explanation panel: `slate-50`, 1px `slate-200`, a 4px `blue-500` left rule, `radius-lg`, `shadow-callout`, 0.9em `slate-700`, `h4` in `blue-900`.
- Covariance explainer: `blue-tint`, a 3px `blue-500` rule, `radius-md`.
- Resampling summary: `glass-slate` with a 12px blur, 1px `glass-border`, a 4px `blue-500` rule, `radius-2xl`, `slate-600`.
- Caption: `slate-50`, 1px `slate-200`, `radius-lg`, 0.8em `slate-600`.

**Consumer provides.** Short explanatory copy in the system's voice: what the chart shows and what is expected. For the explainer, also the `HelpIcon` toggle (`.help-toggle[aria-expanded]`).

- **Do** explain surprising-but-expected behaviour.
- **Don't** use callouts for errors. Use `RunStatusBanner` or error metric cards.

Hand-written from `static/styles.css` `.explanation-*`, `.covariance-explainer`, `.resampling-summary`, `.scan-view-at-caption`.
