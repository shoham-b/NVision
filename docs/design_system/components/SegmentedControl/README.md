# SegmentedControl

A wrapping row of pill buttons that acts as a radio group. It picks generator, strategy, view or stopping point. Its compact variant, `noise-btn`, also supports click-to-anchor, shift-click range selection over noise levels and facet values.

**Markup.** `<div class="segmented-control" role="radiogroup" aria-label="…">` with `<button role="radio" aria-checked>` children. Give the active button `is-active` and `tabindex="0"`, and the others `tabindex="-1"`.

**Compact range buttons.** Add `class="noise-btn"` to each button. The source scopes these by ID (`#scan-noise`, `#scan-generator-facets`); here they are scoped under `.segmented-control`. The states are:
- `is-active`: selected; `primary` fill
- `is-range-anchor`: the first end of a range; `primary-strong` fill with an inset white ring
- `is-range-preview`: inside the pending range; `primary-soft` fill (white on it is 2.23:1, per the source)

**Tokens.**
- Pill: `control` fill, 1px `border-control`, `radius-pill`, 0.4em × 0.9em padding.
- Compact: `radius-sm`, 0.78em text.
- Hover: border `ink-disabled`.
- Disabled: `disabled` fill, `ink-disabled` text, opacity .55.
- Focus: 2px `primary` ring, 2px offset.

**Consumer provides.** Options, the selection state and roving-tabindex keyboard handling. Lead the row with a `.control-label`.

- **Do** use it for 2–8 mutually exclusive options.
- **Don't** use it for long lists. Use `Select` there.

Hand-written from `static/styles.css` `.segmented-control`, `#scan-noise .noise-btn`, `.facet-select-*`.
