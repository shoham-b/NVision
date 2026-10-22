# SignalEquation

A small panel that typesets the selected generator's signal model in HTML, with stacked fractions and tall brackets and no math library. Notes and parameter values follow the formula.

**Markup.** `<div class="signal-equation-panel">` holding:
- `.signal-equation-title`
- `.signal-equation-formula`, which uses:
  - `<b>` for model symbols
  - `.eq-frac > .eq-num + .eq-den` for a stacked fraction
  - `.eq-brackets` for square brackets that stretch to their content
- `.signal-equation-note`, holding `.signal-eq-note-block`, `.signal-eq-note-label` and `ul.signal-eq-note-list`
- `.signal-equation-params > .signal-eq-param`

**Tokens.**
- Panel: `slate-50`, 1px `slate-200`, `radius-lg`, 0.8em.
- Title: weight 600, `slate-700`.
- Formula: `type-equation` (`font-math`) in `slate-800`; symbols in `blue-800`.
- Note: italic `slate-500`.
- Params: `slate-600`.
- Fraction bars and brackets use `currentColor`.

**Consumer provides.** The formula markup, symbol definitions and current parameter values.

- **Do** use real minus signs (−) and Greek letters.
- **Don't** write `a/b` inline. Use `.eq-frac`.

Hand-written from `static/styles.css` `.signal-equation-*`, `.eq-frac`, `.eq-brackets`.
