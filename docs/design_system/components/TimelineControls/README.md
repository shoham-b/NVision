# TimelineControls

The sticky, frosted bar that scrubs every plot on the page to one measurement step. It marks SMC resamples as orange ticks and convergence milestones as dots you can click to jump to.

**Markup.** `<div class="global-timeline-controls">` holding:
- a `.timeline-btn` (play/pause)
- a `.timeline-slider-container` with an `input.timeline-slider` and a `.timeline-ticks-container`
- a `.timeline-label` readout and a `.speed-select`

Inside the ticks container:
- `span.timeline-tick` is one per resample.
- `button.timeline-milestone` is one per milestone, with `milestone-primary` (primary converged), `milestone-all` (all converged) or `milestone-full` (end of run).
- Position each with an inline `left: %`.

**Tokens.**
- Bar: `glass` fill with a 12px backdrop blur, 1px `glass-border`, `radius-2xl`, `shadow-floating`, `space-timeline-*` padding and gap, `font-system`.
- Thumb: 18px `timeline` disc ringed in white (`timeline-strong` on hover).
- Ticks: `orange-500`.
- Milestones: `timeline`, `milestone-all`, `slate-500`, each with `shadow-milestone`.
- Play button: `timeline-tint` fill, 2px `timeline` border, `timeline-strong` text.

**Consumer provides.** The step range and value, resample and milestone positions, and playback.

- **Do** keep it the only sticky element (`position: sticky; top: 0; z-index: 1000`).
- **Don't** reuse the `timeline` blue anywhere outside this bar.

Hand-written from `static/styles.css` `.global-timeline-controls`, `.timeline-*`, `.speed-select`.
