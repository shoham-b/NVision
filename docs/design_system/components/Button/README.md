# Button

NVision's buttons are deliberately quiet. A grey bordered `control-button` steps through repeats. A text-only `link-button` handles page actions. A square `arrow-btn` steps a slider.

**Markup.**
- `<button type="button" class="control-button">Previous</button>`, grouped in `.repeat-picker` around a `Select`.
- `class="link-button"` works on `<button>` or `<a>` and looks like a link.
- Add `arrow-btn` to a `control-button` to make a 24px square glyph button. Give it an `aria-label`.

**Tokens.**
- `control-button`: `control` fill, 1px `border-control`, `ink` text, `radius-md`, 0.4em × 0.9em padding. Hover: `ink-disabled` border, `control-hover` fill. Disabled: opacity .6, `not-allowed`.
- `link-button`: `primary` text, underlined on hover.
- Focus: a 2px `primary` ring on all of them.

**Consumer provides.** The label (a sentence-case verb or noun), `aria-label` when the label is a glyph, and the click handler.

- **Do** keep page-level actions ("Recalculate results") as link-buttons in `.header-actions`.
- **Don't** add a filled primary button. The UI has none; selection is shown by `SegmentedControl`/`TabBar` `is-active`.

Hand-written from `static/styles.css` `.control-button`, `.link-button`, `.arrow-btn`, `.repeat-picker`.
