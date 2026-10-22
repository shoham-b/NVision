# ParamSlider

Read-only slider visuals for parameter ranges.
- The green slider card shows where the sweep narrowed a prior inside its full bounds, and the current value.
- The thin violet range track marks a true value within its bounds.

**Markup.**
- **Narrowed priors:** a `.param-bounds-header`, then `.param-bounds-grid > .param-slider-card`. Each card holds a `.param-slider-label` and a `.param-slider-wrapper`, then `.param-slider-bounds`. The wrapper contains `.param-slider-track-bg`, `.param-slider-track-fill`, `.param-slider-handle` and `.param-slider-value`. Position the fill, handle and value with inline `left`/`width` %.
- **Range marker:** `.param-range-container > .param-range-track > .param-range-marker`, then `.param-range-bounds`.

**Tokens.**
- Card: `surface`, 1px `slider-card-border` (`-hover` on hover, with a 1.5px lift), `radius-lg`.
- Label: bold `green-700`, capitalised.
- Track: `slate-200`; fill: an `emerald-500`→`emerald-400` gradient; handle: a white disc ringed in `emerald-500`.
- Value: mono bold `emerald-700`. Bounds: mono `slate-500`.
- Range marker: a `violet-500` disc on a `radius-track` `slate-200` track. Its bounds are `slate-400` (decorative, 2.56:1).

**Consumer provides.** Bounds, narrowed interval and value as percentages of the full range, plus labelled physical values.

- **Do** treat these as displays, not inputs.
- **Don't** give them `<input>` semantics.

Hand-written from `static/styles.css` `.param-bounds-*`, `.param-slider-*`, `.param-range-*`.
