# Notice

An amber summary box at the top of a view. It calls out something the reader must weigh before reading the charts below, such as how many runs failed.

**Markup.** `<div class="notice-amber">`, with the count in `<strong>` and then the breakdown.

**Tokens.** `amber-50` fill, 1px `amber-200`, `radius-md`, 0.9em `amber-800` text (6.37:1).

**Consumer provides.** One line: the headline number, then " — " and the breakdown.

- **Do** use it for caveats that change how the data should be read.
- **Don't** use it for run state. That is `RunStatusBanner`.

Named from the inline style on `#failed-runs-summary` in `static/index.html`.
