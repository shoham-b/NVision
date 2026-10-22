# SliderWithArrows

A step slider for scrubbing one plot through measurement steps. Arrow buttons step by one; a monospace readout shows the current step.

**Markup.** `<div class="slider-with-arrows">` holding an `arrow-btn`, an `<input type="range">`, a second `arrow-btn` and a `<span class="slider-value-display">`.

**Tokens.**
- Container: `glass-soft` fill, 1px `slate-200`, `radius-lg`, 4px × 8px padding.
- Track: 6px `slate-300`.
- Thumb: 14px `primary` disc, `primary-hover` and scaled 1.25 on hover.
- Readout: 0.9em, weight 600, `font-mono`, `slate-700`.

**Consumer provides.** min, max and value, the input handler, and `aria-label`s on the arrows.

- **Do** use it for a plot-local step.
- **Don't** use it for the page-wide step. That is `TimelineControls`.

Hand-written from `static/styles.css` `.slider-with-arrows`, `.arrow-btn`, `.slider-value-display`.
