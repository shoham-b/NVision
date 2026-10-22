# HelpIcon

A small round `?` placed after a heading or metric label. It carries a one- or two-sentence explanation as its `title` tooltip, or toggles a `Callout` open. It is the UI's only icon.

**Markup.**
- After a heading: `<span tabindex="0" class="help-icon-visible" title="…">?</span>`.
- Inside a metric card: `class="help-icon"`.
- As a toggle: a `<button class="help-icon-visible help-toggle" aria-expanded>`. While expanded it shows `blue-100` / `blue-700` / `blue-300`.

**Tokens.**
- 1.2em (1.1em in metric cards) `radius-circle` disc, 0.65em bold.
- Rest: `slate-500` on `slate-200` (3.86:1, below 4.5 for its size; per the source).
- Hover and focus: white on `blue-500`, plus the 2px `primary` focus ring.

**Consumer provides.** The tooltip text: what the chart shows and what is expected, in plain technical sentences.

- **Do** make it focusable (`tabindex="0"`) so keyboard users can reach the tooltip.
- **Don't** use any glyph other than `?`.

Hand-written from `static/styles.css` `.help-icon-visible`, `.help-icon`, `.help-toggle`.
