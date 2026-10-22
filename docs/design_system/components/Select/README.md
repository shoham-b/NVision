# Select

A native `<select>` for choices too long for a `SegmentedControl`: parameter study, repeat number, a facet value, playback speed.

**Markup.**
- `<select class="control-select" aria-label="…">`.
- A facet value is `.facet-select-wrapper > label.facet-select-label` wrapping the parameter name and its select.
- The timeline's speed picker is `class="speed-select"`.

**Tokens.**
- `control-select`: `control` fill, 1px `border-control`, `radius-md`, 0.6em × 0.8em padding, min-width 6em. Hover: `ink-disabled` border.
- `speed-select`: `surface` fill, 2px `slate-300` border (hover `slate-400`), `radius-lg`, 13px bold `slate-700`.

**Consumer provides.** Options, `aria-label`, and the change handler.

- **Do** lead with a `.control-label` ("Study:").
- **Don't** restyle the native dropdown list.

Hand-written from `static/styles.css` `.control-select`, `.speed-select`, `.facet-select-*`.
