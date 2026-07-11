---
name: nvision-ui
description: Guides working with NVision's `static/` results frontend (index.html, app.js, plotly-utils.js, etc.) — a build-step-free single-page app served from `static/` in dev and mirrored to `artifacts/` for the static export. Use whenever the user asks to change the results UI, fix a plot or chart that's rendering wrong, add a UI control, or debug something visible in the browser at localhost:18080. Covers the mandatory `static/` → `artifacts/` sync step after any edit (easy to forget — it's why "I edited the file but nothing changed" happens), script load order constraints, and why editing the files isn't enough — a UI change must be verified by actually running the server and looking at it in a browser.
---

# NVision UI (`static/`)

Full architecture reference: `docs/ui_architecture.md` — read it before any non-trivial change,
especially the "Data formats" and "app.js internal structure" sections. This skill covers the
things that are easy to get wrong or forget.

## No build step, but two copies must stay in sync

`static/` is what the dev server serves live; `artifacts/` holds the static export produced by
`nv render`. **After every edit to a file in `static/`, copy it to `artifacts/` too** — otherwise
the change is invisible when viewing already-rendered results, and it looks like the edit "didn't
work." From `docs/ui_architecture.md`:

```powershell
@("bootstrap.js","run-status.js","format-utils.js","plotly-utils.js","app.js","reload.js","index.html","styles.css") |
    ForEach-Object { Copy-Item "static\$_" "artifacts\$_" -Force }
```

(Bash equivalent: `for f in bootstrap.js run-status.js format-utils.js plotly-utils.js app.js reload.js index.html styles.css; do cp "static/$f" "artifacts/$f"; done`)

## File map

| File | Responsibility |
|---|---|
| `index.html` | HTML shell and `<script>` load order |
| `styles.css` | All CSS |
| `bootstrap.js` | Loads `manifest.js`/`settings.js`; sets `window.MANIFEST`, `window.SETTINGS`, `window.NVISION_ASSET_PREFIX` |
| `format-utils.js` | Pure formatting helpers, no DOM/Plotly deps |
| `plotly-utils.js` | Plotly CDN loading, JSON/gz fetch, array decode |
| `run-status.js` | Run status banner polling + accordion buttons |
| `app.js` | `main()` — all scan/Bayesian/comparison UI logic (~4000 lines) |
| `reload.js` | Recalculate button, `r` shortcut, calls `main()` after bootstrap |

## Script load order is load-bearing — don't reorder

`bootstrap.js` must load first (sets `window.NVISION_BOOTSTRAP`). `format-utils.js` and
`plotly-utils.js` must precede `app.js` (it calls their globals at call time). `app.js` must
precede `reload.js` (which calls `main()`). Everything communicates through `window` globals —
no modules, no bundler — so load order is the only thing enforcing dependency order.

## `app.js` structure

`main()` is one large closure owning all shared UI state (`currentPlot`, `plots`, `scanIframe`,
etc.); helper functions live inside it deliberately so they can close over that state. It's
organized into ~13 sections in a fixed order (manifest filtering → global timeline controls →
Bayesian tabs → scan controls → scan rendering → comparison cards → Bayesian rendering → summary
view → Highlights view → narrowed-bounds panel → Dashboard tab → tab wiring → event listeners).
For the full section-by-section breakdown and what each renders, read
`docs/ui_architecture.md` rather than re-deriving it from the ~4000-line file — grep for the
section's distinguishing function name (e.g. `renderHighlights`, `renderDashboard`) to jump
straight there.

## Verify in the browser — don't stop at editing the source

Per this repo's AGENTS.md, `static/` is explicitly "do not analyze directly, run the server
instead." A UI change isn't done until you've actually seen it render:

1. Start the server: `uv run nv serve` (port 18080; use `--dir artifacts` if that's what you
   need to check) — or use the `preview_start`/`preview_*` tools if available in this session.
2. Reload the page (or press `r` in-browser to trigger a recalculate).
3. Check the browser console for JS errors and confirm the change actually rendered.
4. Remember the sync step above if you're checking against `artifacts/`.

## Data format gotchas when a plot renders blank

Plot files are gzip-compressed JSON (`.json.gz`). Numeric arrays show up in one of two encodings
— `{"__f32__": "<base64>"}` (this project's compact format) or `{"bdata": "<base64>", "dtype":
"float64"}` (Plotly Python 5.x's native format). Both are decoded by `_decodePlotlyFigure` in
`plotly-utils.js` — if a plot renders blank or throws in the console, check whether a new payload
shape slipped past that decoder rather than assuming the data itself is bad.
