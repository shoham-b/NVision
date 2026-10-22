# EmptyState

A dashed, centred box shown in place of a chart or panel when the current selection has no data.

**Markup.** `<div class="empty-state" role="status">No Bayesian analysis data available for this selection.</div>`. A compact variant (Grid Stats) uses the same words with 1.5em padding and no box.

**Tokens.** `surface-sunken` fill, a 1px dashed `border-control`, `radius-lg`, 2em padding, `ink-muted` text (5.50:1).

**Consumer provides.** One sentence saying what is missing and for what: "No data for this combination."

- **Do** keep the space the chart would take, so the layout doesn't jump.
- **Don't** use it for errors. Errors go in `RunStatusBanner` or `Notice`.

Named from the inline style on `#bayes-no-data-message` in `static/index.html`.
