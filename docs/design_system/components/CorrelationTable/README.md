# CorrelationTable

A bordered matrix of posterior parameter correlations. The first column is sticky for wide matrices.

**Markup.** `<div class="correlation-table-wrapper"><table class="correlation-table">`. The first row and first column hold parameter names. Value cells are `td.corr-cell`; they scale up 1.05 on hover.

**Tokens.**
- Wrapper: 1px `slate-200`, `radius-lg`, `shadow-table`; scrolls horizontally.
- Table: `font-system` at 0.9em, `surface` body, 12px × 16px cells with 1px `slate-200` rules.
- Headers: weight 600, `slate-600` on `slate-50`.
- First column: `slate-700`, sticky.
- Values: mono bold.

**Consumer provides.** Parameter names in code vocabulary, and values to two decimals with a true minus sign (−). The source can tint `corr-cell` backgrounds by strength inline.

- **Do** keep the matrix symmetric, with 1.00 on the diagonal.
- **Don't** drop the sticky first column.

Hand-written from `static/styles.css` `.correlation-table*`, `.corr-cell`.
