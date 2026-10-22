# TabBar

Pill tabs that switch between the page's top-level `Section`s.

**Markup.** `<div class="tab-bar" role="tablist">` holding `<button class="tab-button" role="tab" aria-selected>`. The active tab adds `is-active`. Each controls a `.section.tab-panel`; hide the others with `is-hidden`.

**Tokens.**
- Rest: `tab` fill, 1px `border-control`, `ink` text, `radius-pill`, 0.6em × 1.2em padding.
- Hover: border `ink-disabled`, fill `tab-hover`.
- Active: `primary` fill and border, white text.
- Gap `space-3`; vertical margin `space-6`.

**Consumer provides.** The labels, the tab/panel wiring (`aria-controls` / `aria-labelledby`), and arrow-key handling. Today `app.js` builds a single tab, "Scan measurements", when scan plots exist. Its label repeats the section's `h2`, so new tabs should do the same.

- **Do** keep a single row that wraps.
- **Don't** use pill tabs inside a section. Use `SubTabs` or `SegmentedControl` there.

Hand-written from `static/styles.css` `.tab-bar`, `.tab-button`.
