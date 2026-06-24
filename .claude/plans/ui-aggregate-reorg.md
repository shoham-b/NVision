# Plan — Reorganize the aggregate/comparison UI

## Context

The single-run **debug** view is the primary use of the dashboard and is fine. The
pain is the **aggregate side** — the "all runs" summaries and noise comparison. Two
concrete problems, both layout/IA only (no metric-math changes):

1. **Redundancy + scattered controls.** Four overlapping views (Highlights, Dashboard,
   Repeats Summary, Noise Metrics) show the same three quantities — error, cost
   (measurements/convergence), savings-vs-baseline — sliced four ways. "Savings vs
   span" is duplicated **three times** (`hl-span`, `dash-span-plot`,
   `summary-span-container`), and there are **three independent baseline selectors**
   (`hl-span-baseline`, `dash-span-baseline`, `noise-baseline-select`), reached from
   two different nav levels (Dashboard is a top tab; Highlights/Summary/Noise are
   sub-toggles inside Scan).

2. **Repeats Summary pairwise explosion.** It uses a two-dropdown, one-pair-at-a-time
   selector (`buildTwoDropdownSelector` -> `onPairChange(eA, eB)`) over a **flat list
   of 3 x (#strategies)** entities. With S strategies that's ~(3S)²/2 pairs viewed one
   at a time. It also can't show a single locator's stopping criteria **side by side**
   (you can only hold two entities), even though those entities already exist.

Key existing fact: `_makeStrategyEntities` (`static/app.js:3673`) already builds three
entities per locator — `X`, `X freq converged` (`_freq`), `X converged` (`_conv`)
(lines 3686-3699). So the convergence-criteria comparison is **already in the data**;
it's just never surfaced as a set. Comparing a locator across its own stopping
criteria is also the *fairest* comparison available (same run truncated at different
points, paired by identical ground truth) — worth promoting.

## Goal
- Replace the pairwise default in Repeats Summary with an all-at-once table grouped by
  locator (criteria nested), surfacing the convergence-criteria comparison and killing
  the pair explosion.
- Consolidate the four aggregate views into one **Compare** area: one control bar, pick
  the slice axis once, one savings-vs-span panel.
- Do NOT touch the single-run debug view or any metric computation.

## Relevant code
- Page shell / the four aggregate containers: `static/index.html`
  - `#highlights-view` (lines ~163-199), `#scan-summary-view` (~201-208),
    `#noise-metrics-view` (~210-248), `#dashboard-section` (~251-262).
  - View-mode toggle `#scan-view-mode` (~50-55); three baseline selectors noted above.
- Renderers in `static/app.js`:
  - `renderRepeatsSummary` (4117), `buildSummaryEntities` (3647),
    `_makeStrategyEntities` (3673), `buildTwoDropdownSelector` (3861),
    `buildPairwiseRows` (3815), `renderPairwiseCards` (3921),
    `renderSavingsVsSpanChart` (4167).
  - `renderHighlights` (4438) + `renderHl*` panels (4520-4876).
  - `renderDashboard` (4877), `renderDashGeneratorSection` (4919),
    `renderDashPooledSpan` (5072).
  - Noise metrics render path (controls in index.html `#noise-*`; plotted into
    `#comp-div-summary-*`).

## Phase 1 — Unify single-repeat + summary into entity-major cards (ship first)

Insight: single-repeat and summary are the **same card with N=1 vs N=many repeats** —
`buildPairwiseRows` already passes each metric as `data` that is a **scalar
(single-repeat) or an array (summary histogram)**, and one renderer handles both
(`isArr` branch, `static/app.js:3815`). The problem is the layout is *metric-major*
(each row = one metric split into A / B / Δ cards), which structurally forces exactly
two entities and causes the pair explosion. Also, cross-criterion pairing is
**deliberately blocked** in `buildTwoDropdownSelector.refreshBOptions`
(`static/app.js:3891`) — which is exactly why a locator's criteria can't be compared.

Goal: flip to **entity-major** — one card per entity holding ALL its metric rows
(cells scalar-or-histogram), so any number of entities sit side by side and a
locator's `full` / `freq converged` / `converged` are just adjacent cards.

1. New `buildEntityCard(entity)` in `app.js`:
   - One card per entity. Rows = the metric set already defined in `buildPairwiseRows`
     (Steps to completion, Final freq uncertainty/error, Steps to freq conv,
     Uncertainty/Error @ freq conv). Reuse that metric list + the existing scalar/array
     cell renderer so numbers and formatting match today's cards.
   - Cell renders a point value when `data` is scalar, a histogram/distribution when
     `data` is an array (the renderer already branches on this).
2. New `renderEntityCards(entities, container, {groupByLocator})`:
   - Group cards by base locator id (strip `_freq`/`_conv` via `entityCriterion`,
     `static/app.js:3853`); render a group header per locator with its up-to-three
     criterion cards adjacent.
3. Make this the renderer for BOTH:
   - **Single-repeat** (`#scan-metrics`): the per-strategy entity card(s) for the
     current repeat (scalar cells) — replaces the ad-hoc scan-metrics list.
   - **Repeats Summary** (`#summary-subjects-container`): all entity cards (histogram
     cells) — replaces the two-dropdown pairwise default.
4. Preserve the paired delta as an **opt-in**: a "Δ vs <reference card>" toggle that
   re-renders deltas against a chosen entity, reusing the paired per-repeat diff logic
   in `renderSavingsVsSpanChart` (`static/app.js:4167`) factored into a shared helper.
   Default view is plain entity cards (no pairing, no cross-criterion block).

Verify: single-repeat shows the same numbers as before in card form; Repeats Summary
shows every locator + its 3 criteria at once as histograms, grouped, no pair selector
needed; the optional Δ mode reproduces the old pairwise delta for a chosen reference.
No console errors.

## Phase 2 — Consolidate into one "Compare" area

Goal: one control bar + axis selector replaces four views and three baseline pickers.

1. In `index.html`, introduce a single `#compare-section` with one control bar
   (locators multi-select · stopping criterion · baseline) and an **axis selector**:
   `vs measurements · vs noise · vs signal span · pooled heatmap`.
2. Route existing renderers by axis instead of by view:
   - vs measurements -> `renderHlErrorCurves` / `renderHlSurvival`.
   - vs noise -> the noise-metrics plots (`#comp-div-summary-*`).
   - vs span -> ONE savings-vs-span panel (delete the other two duplicates); reuse the
     pooled logic in `renderDashPooledSpan` (5072) / `renderSavingsVsSpanChart`.
   - pooled heatmap -> `renderDashGeneratorSection` (4919).
   - coverage (`renderHlCoverage`, 4637) shown alongside as the "trust" panel.
3. Collapse the three baseline selectors into the single control bar; remove
   `#scan-view-mode`'s aggregate options and the now-dead `#highlights-view`,
   `#scan-summary-view` (table moves under Compare), `#noise-metrics-view`,
   `#dashboard-section` once their renderers are re-hosted.
4. Keep the Repeats Summary table (Phase 1) as one axis/section within Compare.

Verify: each axis shows the same three quantities consistently; baseline set once
applies everywhere; no duplicated savings-vs-span; the debug (single-run) view is
unchanged.

## Out of scope
The single-run **debug diagnostics** (the Bayesian sub-tabs: posterior evolution, SMC
weights/ESS, Fisher, covariance, jitter) stay untouched. NOTE: Phase 1 *does* touch the
single-repeat **scalar metrics panel** (`#scan-metrics`) — it gets unified into the
entity-card component — but nothing below it. No metric computation (median/CI, MLE,
convergence criteria — all previously aborted); no new metrics. Layout/IA only.

## Open question for Phase 1
Card row order: lead with the outcome rows (final error, final uncertainty, steps to
completion) and put the "@ freq convergence" rows below — confirm that ordering, since
the card now carries all rows for every entity at once.
