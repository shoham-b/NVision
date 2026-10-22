# ResultsPage

The whole NVision results page in its Run view, assembled from the system's components in the order `static/index.html` uses. Start every new page or view from this skeleton.

**Structure, top to bottom.**
1. **Header.**
   - `h1` "NVision results"
   - `RunStatusBanner` (`#run-status-banner`)
   - `.last-run` stamp
   - `.header-actions` with link-buttons
2. **`TabBar`.** One pill per top-level section; today only "Scan measurements".
3. **`Section`** (`.section.tab-panel`), containing:
   - a. **Controls.** `h2` + `HelpIcon`, then `.scan-controls-split` holding the Signal `Panel` (generator `SegmentedControl`, `SignalEquation`, noise range buttons) and the Locator `Panel` (stopping point).
   - b. **View switch.** A "View:" `SegmentedControl` (Run, Speed, Accuracy, Stages, Pooled). Each non-Run view is a `.view-panel` of `ChartCard`s that replaces 3c–3d.
   - c. **Run view.**
     - a `.repeat-row` with the repeat picker (`Button` + `Select`)
     - the `.metrics` summary line
     - a `MetricsPanel`
     - `BoundsChips`
     - the scan `PlotFrame` (`#scan-plot-div`)
     - the "view at" `Callout` caption
   - d. **Bayesian block.** A `divider-soft` top rule, then:
     - an `h2`
     - the sticky `TimelineControls`
     - `SubTabs` with one `.bayes-tab-panel` per sub-view; each holds an `h3` + `HelpIcon` and a `PlotFrame`

**Consumer provides.**
- Data for every slot.
- Tab and sub-tab switching.
- The repeat picker and timeline behaviour.
- Plotly rendering into the plot `div`s. The source lazy-loads `https://cdn.plot.ly/plotly-2.27.0.min.js`.

**What is not real in the preview.**
- The scan plot is a static SVG stand-in, and the posterior heatmap is a gradient stand-in.
- The source hides the Locator panel and the `SignalEquation` until data selects them.
- The body's 2em margin is the source's own.

- **Do** keep this order and these IDs when an agent rebuilds or extends the page. `app.js` looks elements up by ID.
- **Don't** add a second top-level layout, a sidebar or a nav bar. NVision is one long page of sections.

Hand-assembled from `static/index.html`.
