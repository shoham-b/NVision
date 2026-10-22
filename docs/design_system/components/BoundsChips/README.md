# BoundsChips

A green strip of monospace chips listing each parameter's narrowed bounds once the search stage has focused the prior.

**Markup.** `<div class="narrowed-bounds-panel">` holding a `.param-bound-label`, then one `<span class="param-bound-chip"><strong>name</strong> lo – hi unit</span>` per parameter.

**Tokens.**
- Panel: `bounds-tint` fill, 1px `bounds-border`, `radius-lg`, 0.88em `bounds-ink` text.
- Chip: `bounds-chip` fill (`bounds-chip-hover` on hover), 1px `bounds-chip-border`, `radius-pill`, `type-chip` in `bounds-label`.

**Consumer provides.** Parameter names (code vocabulary), physical bounds with units, and an en dash between them.

- **Do** show physical (`_phys`) values, never unit-cube ones.
- **Don't** use the bounds green for anything except narrowing.

Hand-written from `static/styles.css` `.narrowed-bounds-panel`, `.param-bound-*`.
