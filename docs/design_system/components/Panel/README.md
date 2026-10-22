# Panel

A sunken group of related controls inside a `Section`. The Signal panel and the Locator panel sit side by side and wrap on narrow screens.

**Markup.**
- Wrap the panels in `.scan-controls-split`.
- Each is a `.panel` plus `.panel-signal` or `.panel-locator`.
- Inside: a `h3.panel-title`, then `.control-row`s. Each row is a bold `.control-label` ("Generator:") followed by its control.

**Tokens.**
- `surface-sunken` fill, 1px `border` and `radius-lg`.
- `space-4`/`space-5` padding.
- The 4px left rule is `signal-accent` or `locator-accent`.
- The title is `type-panel-title` in `primary`.

**Layout.** `flex: 1 1 280px`, min 260px, with a 1.5rem gap. Comparison views use `.comparison-outer` / `.comparison-column` (`1 1 400px`, 2rem gap) with `.comparison-subheading` (`type-subheading`). `.repeat-row` and `.comparison-shared-signal` separate blocks with a 1px `divider`.

**Consumer provides.** The title and the control rows. Hide a row with the `hidden` attribute.

- **Do** end every control label with a colon.
- **Don't** use the coloured left rule anywhere except these two panels and the `Callout`s.

Hand-written from `static/styles.css` `.panel`, `.control-row`, `.comparison-*`, `.repeat-row`.
