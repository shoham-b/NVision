# Switch

An iOS-style on/off switch for a single chart option, such as the shared "Log scale" toggle on uncertainty charts.

**Markup.** `<label class="ios-switch"><input type="checkbox"><span class="ios-switch-track"></span><span class="ios-switch-label">Log scale</span></label>`. The checkbox stays in the DOM, visually hidden rather than `display:none`, so click, keyboard and focus work natively. The track and its `::before` knob are drawn in CSS.

**Tokens.**
- Track: 38 × 22px, `radius-pill`, `gray-300` off and `switch-on` on.
- Knob: 18px `surface` disc with `shadow-knob`; it slides 16px.
- Label: 0.8em `slate-600`.
- Focus: 2px `primary` ring on the track.

**Consumer provides.** The label and the `change` handler. In the source it is built by `_logScaleToggle` in `app.js`.

- **Do** use it for view options that apply at once.
- **Don't** use it to pick between two named modes. Use `SegmentedControl` for that.

Hand-written from `static/styles.css` `.ios-switch`.
