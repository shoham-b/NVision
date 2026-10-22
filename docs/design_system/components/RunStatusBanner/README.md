# RunStatusBanner

The full-width banner under the page title that says whether the experiment behind the page is scheduled, running, partial, failed or complete. The running state has a spinner and a refresh button.

**Markup.** `<div class="run-status-banner run-status-<state>" role="status">`. `<state>` is one of `scheduled`, `running`, `partial`, `error` or `complete`; use `role="alert"` for `error`.
- Running adds a leading `<span class="run-status-spinner" aria-hidden="true">`.
- A `<button class="run-status-refresh-btn">` (outlined in `currentColor`, filled on hover) reloads results.

**Tokens.**
- Each state uses `status-<state>-bg`, `-border` and `-ink`.
- 0.95em, weight 500, `radius-md`, 0.75em × 1em padding.
- The spinner is a 2px `spinner-track` ring with a `status-running-ink` arc, spinning at 0.8s linear.
- `status-partial-ink` on its fill is 3.46:1, below 4.5:1. Kept from the source; the words carry the state.

**Consumer provides.** The state and a short message: "<State> — <detail>".

- **Do** lead with the state word so it reads without colour.
- **Don't** show more than one banner.

Hand-written from `static/styles.css` `.run-status-*`, wired by `static/run-status.js`.
