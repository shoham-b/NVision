# SubTabs

Squared, smaller tabs for the sub-views inside one section, such as the Bayesian analysis panes. They sit on a hairline.

**Markup.** `<div class="bayes-tab-bar" role="tablist">` holding `<button class="bayes-tab-button" role="tab">`; the active one adds `is-active`. The panes are `.bayes-tab-panel`, shown by `is-active`. An embedded plot inside a pane is 85vh (min 600px).

**Tokens.**
- Rest: `tab` fill, 1px `border-control`, `radius-md`, 0.9em text.
- Hover: `ink-disabled` border, `tab-hover` fill.
- Active: `primary` fill, white text.
- The bar has a 1px `border` underline and a `space-2` gap.

**Consumer provides.** Labels and pane switching.

- **Do** use it for alternative views of the same data.
- **Don't** use it for filters. Filters are `SegmentedControl`.

Hand-written from `static/styles.css` `.bayes-tab-bar`, `.bayes-tab-button`, `.bayes-tab-panel`.
