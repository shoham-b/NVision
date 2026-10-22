# Section

The white card that holds one tab panel's content: a `type-h2` title with its `primary` rule, controls, plots and summary lines.

**Markup.** `<div class="section tab-panel" role="tabpanel" aria-labelledby="tab-…">`. Add `is-hidden` to hide an inactive panel. The page above it is a `type-h1` title, a `.last-run` stamp and `.header-actions` holding link-buttons.

**Tokens.**
- `surface` fill, 1px `border`, `radius-lg` and `shadow-section`.
- `space-6` padding; sections are `space-8` apart.
- The `h2` is `primary` with a 2px `primary` bottom rule and 0.5em padding.
- `.metrics` summary lines are `type-metrics` in `ink-soft`.

**Consumer provides.** The title, an optional `HelpIcon` at the end of it, and the panel body.

- **Do** use one section per top-level tab.
- **Don't** nest sections. Group controls inside one with `Panel`.

Hand-written from `static/styles.css` `.section`, `h1, h2`, `.last-run`, `.header-actions`, `.metrics`.
