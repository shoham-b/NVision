# EntityTable

The Repeats summary card: one bordered group per strategy (the entity), with a CSS-grid table of metrics (rows) by statistic (columns). A lime "delta" variant shows the difference between two entities.

**Markup.**
- The group is `.entity-group` (add `entity-group-delta` for differences), holding a `.entity-group-header` and a `.entity-table`.
- Set `grid-template-columns` inline (label column plus N stat columns).
- The first row is a `.entity-table-blank` followed by `.entity-table-col-header`s.
- Each following row is a `.entity-table-row-label` and `.entity-table-cell`s.
- A cell holds a `.entity-metric-scalar`, or `.entity-metric-na` ("—") when the value is missing.

**Tokens.**
- Group: `surface`, 1px `slate-200`, `radius-xl`.
- Header: `type-group-header` in `slate-800` on `slate-50`, with a 2px `slate-200` rule.
- Column headers: `type-col-header` in `slate-600`, uppercase.
- Row labels: `slate-500` on `surface-sunken`.
- Values: mono bold `violet-800`.
- N/A: `slate-300` (1.48:1, per the source; the dash is decorative).
- Delta: 2px `lime-400` border, with a `lime-900` header on `lime-50`.

**Consumer provides.** The entity name, column and row labels, values, and the column template.

- **Do** use the same column order in every group so the cards can be scanned down.
- **Don't** colour individual cells.

Hand-written from `static/styles.css` `.entity-group`, `.entity-table-*`.
