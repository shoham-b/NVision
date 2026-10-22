# NVision Design System

> This folder mirrors the published NVision design-system artifact. `tokens.json` is the source of every value; `tokens.css` is generated from it. Each `components/<Name>/preview.html` links `tokens.css` and `components/bundle.css`, so it opens standalone in a browser.

NVision is the results UI for NV-centre resonance locators: a single, dense, data-first page of Plotly charts and the controls that pick which run they show. It is a scientific instrument, not a marketing surface. Every rule below serves one goal: let a researcher compare locators quickly and trust what the numbers say.

## Building a page (read this first)

This system is written for agents that build or extend NVision UI. The UI is plain HTML and CSS; `static/app.js` builds the DOM, and there is no JS component library. Every component here is its real markup and class names, and every value comes from a token.

1. **Load the styles in this order.** Nothing else is needed for the chrome.
   ```html
   <link rel="stylesheet" href="tokens.css">
   <link rel="stylesheet" href="components/bundle.css">
   ```
   Charts use Plotly, which the source lazy-loads from `https://cdn.plot.ly/plotly-2.27.0.min.js`.
2. **Start from the page skeleton** in `components/ResultsPage/preview.html`. It is the real page in the real order, with the IDs `app.js` looks up. Keep those IDs.
3. **Copy markup from a component's `preview.html`.** Its `README.md` lists the classes, states and what you must supply.
4. **Pick components by job:**

| You need | Use | Root class |
|---|---|---|
| A top-level section with a title | `Section` | `.section.tab-panel` |
| A group of related controls | `Panel` | `.panel.panel-signal` / `.panel-locator` |
| A labelled control row | `Panel` (row) | `.control-row` + `.control-label` |
| One choice from 2–8 options | `SegmentedControl` | `.segmented-control` |
| A range over noise levels or facet values | `SegmentedControl` (compact) | `.noise-btn` |
| One choice from a long list | `Select` | `.control-select` |
| An on/off view option | `Switch` | `.ios-switch` |
| A page action | `Button` | `.link-button` |
| Step through repeats | `Button` + `Select` | `.repeat-picker` |
| Switch top-level sections / sub-views | `TabBar` / `SubTabs` | `.tab-bar` / `.bayes-tab-bar` |
| Scrub every plot to one step | `TimelineControls` | `.global-timeline-controls` |
| Step one plot | `SliderWithArrows` | `.slider-with-arrows` |
| A run-view plot | `PlotFrame` | `.plotly-json-plot` |
| An analysis chart, single or paired | `ChartCard` | `.chart-card`, `.chart-pair`, `.view-panel` |
| Headline numbers | `MetricsPanel` | `.sweep-metrics-panel` / `.scan-metrics-panel` |
| Strategy × statistic summary | `EntityTable` | `.entity-group` |
| A parameter matrix | `CorrelationTable` | `.correlation-table` |
| Narrowed bounds, priors, a value in a range | `BoundsChips` / `ParamSlider` | `.narrowed-bounds-panel` / `.param-bounds-grid` |
| A signal formula | `SignalEquation` | `.signal-equation-panel` |
| How to read a chart | `HelpIcon`, then `Callout` | `.help-icon-visible`, `.explanation-panel` |
| Run state | `RunStatusBanner` | `.run-status-banner` |
| A caveat on the data | `Notice` | `.notice-amber` |
| Nothing to show | `EmptyState` | `.empty-state` |

5. **Rules to check before you finish:**
   - **Values.** Never write a literal colour, radius or shadow. Use `var(--token)`. If no token fits, ask before adding one.
   - **Titles.** Every chart has a title and a `HelpIcon` whose tooltip says how to read it.
   - **Chart colours.** Use `series-*`, `highlight-*` and `trace-*` in slot order, and keep an entity's slot the same on every chart.
   - **Labels.** Use the code's vocabulary (`drive_freq`, `zeeman_split`), never a bare "frequency". Put units on every value. Set machine values in mono.
   - **Restraint.** Keep one brand blue (`primary`) and one sticky element (the timeline). Add no icons, emoji, sidebars or extra layouts.
   - **Contrast.** Don't use the flagged failing-contrast tokens for new essential text (see Colour, below).

## Principles

- **Data first.** The plot is the hero. Chrome is light grey, white cards and one blue. Colour is spent on data and on state, never on decoration.
- **Say exactly what is measured.** Labels use the code's own vocabulary verbatim (`drive_freq (Hz)`, `pl_normalized`, `zeeman_split`), so a label can be searched in the codebase.
- **Explain in place.** Any non-obvious chart gets a `?` help icon carrying a one- or two-sentence tooltip, or a callout opened by it. Never a separate docs page.
- **Fair comparisons only.** Colour encodes identity (strategy, parameter, phase) consistently across every chart on the page.

## Content fundamentals

- **Voice.** Plain and technical, third person, no "I/we". Describe what a chart shows and how to read it: "Individual scan showing measured points (colored by phase) and the inferred signal location."
- **Casing.** The source mixes cases; match the neighbours of whatever you add.
  - Section and view titles (`type-h2`) are mostly Title Case: "Bayesian Analysis", "Failed Runs", "Grid Stats". A few are sentence case: "Scan measurements", "Stages — search, focus, refine".
  - A view title may add an em-dash subtitle: "Speed — Measurement Efficiency", "Accuracy — Uncertainty Quality".
  - Sub-tab labels are Title Case: "Posterior Evolution", "Fisher Bounds", "CRLB Pairs".
  - Buttons and segmented options are sentence case verbs or nouns: "Recalculate results", "Previous", "Next", "Primary converged".
- **Control labels** end in a colon and are bold (`type-control-label`): "Generator:", "Study:", "Repeat:", "View at:".
- **Vocabulary.** Never write a bare "frequency". Write `drive_freq` (the swept axis), `candidate_drive_freq` (the next point) or `zero_field_freq` (the dip-centre parameter). `zeeman_shift` is never a frequency.
- **Numbers.** Plot axes use SI-prefix ticks (`tickformat: "~s"`). Metric values are monospace, with units: "2.8701 GHz", "1.2 kHz". Separate secondary facts with " · ".
- **Emoji.** None. The only glyphs are the `?` in help icons, the `↺` on the resampling summary, and arrow glyphs on step buttons.
- **Help text** explains surprising behaviour honestly: "a sharp uncertainty jump or ESS dip lined up with one of these is expected (fighting particle impoverishment), not an anomaly."

## Visual foundations

### Colour

- **Page and cards.**
  - The page is `page`. Each tab panel is a `Section` on `surface`, with a 1px `border`, `radius-lg` and `shadow-section`.
  - Inside a section, group controls in a `Panel` on `surface-sunken`.
  - Set text in `ink`; use `ink-muted` only for the last-run stamp.
- **One brand blue.** `primary` carries:
  - h1/h2 text and their 2px underline
  - links
  - the active state of every tab and segmented button
  - the focus ring

  Never introduce a second chrome blue. `blue-*` belongs to explanatory callouts and sweep metrics; `timeline` belongs to the timeline only.
- **Controls.** Rest on `control` or `tab`, bordered by `border-control`. On hover the border turns `ink-disabled` (#888). Tabs, sub-tabs and control buttons also darken their fill (`tab-hover`, `control-hover`); segmented and compact range buttons change only the border. The active state is white text on `primary`.
- **Slate panels.** The newer analytic panels (metrics, equations, tables, explanations) use the `slate-*` scale:
  - ground: `slate-50`
  - borders: `slate-200`
  - labels: `slate-500`
  - text: `slate-600` to `slate-800`

  Keep the two neutral families apart. Grey (`ink`, `border`, `control`) is for chrome; slate is for data panels.
- **Metric families are told apart by hue.**
  - Sweep metrics are blue: a `blue-500`→`blue-400` rule, values in `blue-800`.
  - Scan and jitter metrics are violet: a `violet-500`→`violet-400` rule, values in `violet-800`.
  - Expected-value cards are green: `green-50` ground and a `emerald-500`→`emerald-400` rule.
  - Error severity escalates through `red-*`. Medium is a `red-25` ground with a `red-600` value; high is a `red-50` ground with a `red-900` value.
- **Efficiency.** Good is `green-700`, medium is `orange-700` and bad is `red-700`. The word or number always carries the meaning too; colour is never the only signal.
- **Run status** uses its own Material-derived set, `status-<state>-bg/-border/-ink`, for five states: scheduled (purple), running (blue, with spinner), partial (orange), error (red) and complete (green).
- **Charts.**
  - Plot background is white; gridlines are `chart-grid`.
  - Colour per-parameter traces by `series-1…7` in order, and Highlights strategies by `highlight-1…8` in order. Keep an entity's slot fixed across every chart.
  - Scan traces have fixed meanings:
    - `trace-true-signal`: truth, solid
    - `trace-noisy`: noisy simulation, dotted, with `trace-noisy-band-1/-2` σ bands
    - `trace-mode`: locator belief, dashed red
    - `trace-sweep`: sweep belief, dotted brown
  - Measurement markers are coloured by phase: `trace-coarse`, `trace-secondary`, `trace-tertiary`, or a Viridis scale by step.
  - SMC resamples are always `orange-500` dotted lines or ticks.
- **Contrast.** `primary`, `ink`, the `slate-600+` text and every metric value clear 4.5:1 on their grounds. The source has some failing pairs, kept exact and flagged in their token notes:
  - `slate-400` formula and bound text (2.56:1)
  - `slate-300` N/A text
  - `red-500` row text
  - `status-partial-ink`
  - white on `primary-soft`
  - the `border-control` outline (1.61:1)

  Don't use any of these for new essential text.

### Type

- **No brand typeface.** Body is the browser's generic `sans-serif` (`--font-sans`). The sticky timeline and the correlation table use the `system` stack. Numbers use `mono` (ui-monospace / SF Mono / Cascadia Code). Equations use `math` (Cambria Math).
- **Scale.** The scale is browser-default headings plus em fractions:
  - `type-h1` is 2em, `type-h2` 1.5em; both are `primary` with a 2px `primary` rule.
  - `type-panel-title` is 1.1rem.
  - `type-body` is 16px with a 1.6 line height.
  - Metric panels set a 0.8em base (`type-small`) and scale their contents from it: label 0.72em uppercase (`type-metric-label`), value 1.15em mono bold (`type-metric-value`), formula 0.7em (`type-metric-formula`).
  - Every px value in the type tokens assumes a 16px root.
- **Monospace.** Use it for every machine value: metric values, bounds, step counters and parameter chips. Never use it for prose.
- **Small labels.** Set uppercase with 0.02–0.05em tracking.

### Spacing and layout

- **Spacing is em-based**, so it scales with the local font size: `space-1` through `space-8` (0.25em to 2em).
  - The body has a `space-8` margin.
  - Sections have `space-6` padding and are `space-8` apart.
  - Control rows wrap with a `space-3` gap.
  - The only px spacing is the sticky timeline bar's (`space-timeline-*`).
- **Layout is flex-wrap throughout.**
  - Panels are `flex: 1 1 280px` (min 260px).
  - Comparison columns are `1 1 400px`.
  - Metric grids are `repeat(auto-fit, minmax(120px, 1fr))`.
  - Posterior cards sit two-up (min 480px).
- **Plots** fill the width at a minimum height of 500px (Plotly JSON) or 600px (an embedded HTML plot). They sit in a 1px `border-control` frame with `radius-xs`.

### Shape, borders and elevation

- **Radius scale.** Use `radius-xs` (4px) for plots, `radius-md` (6px) for controls, `radius-lg` (8px) for cards, `radius-xl` (10px) for metric panels, `radius-2xl` (12px) for the floating timeline, and `radius-pill` for tabs, segmented buttons, chips and the switch.
- **Borders carry structure.** Shadows stay faint: `shadow-section` at rest, and a 1px lift with a hue-tinted `shadow-card-hover` on interactive metric cards. Only the sticky timeline floats (`shadow-floating`, plus a 12px backdrop blur on `glass`).
- **Accent rules.** The source uses three kinds:
  - a 4px coloured left rule on the Signal and Locator panels and on callouts
  - a 2px gradient top rule on metric cards
  - a 2px `primary` underline on headings

  Use them only in those places.

### States and motion

- **Focus.** Every focusable control gets `outline: 2px solid primary; outline-offset: 2px` on `:focus-visible`. It is 7:1 on surface, so it is solid and visible on every ground.
- **Hover.** Controls turn their border `ink-disabled`. Tabs, sub-tabs and control buttons also take `tab-hover`/`control-hover`. Metric and slider cards lift 1–1.5px.
- **Disabled.** Opacity .55–.6, `disabled` fill, `ink-disabled` text, `cursor: not-allowed`.
- **Transitions.** Use 0.15–0.2s on background, border, colour and transform. Nothing else moves except the run-status spinner (0.8s linear spin).

## Iconography

NVision has no icon set, no icon font and no logo. Icons are typographic:

- `?` in a `radius-circle` `slate-200` disc (`HelpIcon`) is the only icon control.
- `↺` marks the resampling summary.
- Arrow glyphs mark the step buttons of `SliderWithArrows`.
- Coloured dots mark timeline milestones.

When you need a new affordance, use a word on a `Button` rather than inventing an icon. If an icon is unavoidable, set it as a glyph in `ink`, at least 3:1 on its ground.

## Not synced

- **Fonts.** The source ships no font files; the body face is the generic `sans-serif`, so it varies by OS.
- **Logos.** The repository has none, and no icon set.
- **Inline colours in `static/app.js`.** About 50 one-off hex values set through inline `style.cssText` were not tokenized. Only the `COLORS`, `HL_COLORS` and `REASON_COLORS` palettes and the `styles.css` values were.
- **Plotly templates.** Layout fields such as margins, legend placement and heights in `static/graphs/*.json` are described under `PlotFrame`, not as tokens.
- **Route.** Components are static renditions: markup plus `bundle.css`, with no bundle.js, because the source has no component library to build.
- **Inline styles in `static/index.html`.** Recurring ones were named as classes: `ChartCard` (`.chart-card`, `.chart-pair`, `.view-panel`, `.group-label`, `.diagnostic-label`), `EmptyState` and `Notice`. The source still writes them inline, so their class names exist only in this system.
