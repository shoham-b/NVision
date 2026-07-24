---
name: nvision-run-monitoring
description: Guides checking progress on an in-progress (still-running) `nv run`/`nv groups` job and actually verifying its graphs render correctly, as opposed to just trusting the process is "probably fine." Use whenever the user asks to check on a run, see mid-run/in-progress results, find out how many repeats are done, or wants proof that a long-running job's output is actually good — not just that it hasn't crashed. Covers the `nv cache progress` command, running `nv serve` live alongside `nv run`/`nv groups` (no `nv render` step needed — press 'r'/POST `/api/reload` to invalidate the server's in-process manifest cache and pick up new repeats), and gotchas that look like bugs but aren't: a slow first `/api/manifest` build on a big cache, and huge manifests freezing the browser's screenshot/computer tools while the page is actually fine.
---

> This skill's step 4 verification recipe only spot-checks one combo at a time. For bulk,
> whole-run checks, use [[nvision-convergence-check]] (is the locator's output actually
> converging correctly, not just finishing?) and [[nvision-plot-integrity-check]] (does every
> repeat's plot data actually decode, not just the one you clicked on?) once progress here
> confirms the run is far enough along to be worth checking in depth.

# Monitoring an in-progress NVision run

Since the "use api for serving the UI" refactor, `nv serve` reads live from the SQLite
cache (`nvision/cli/api_server.py`) — there is no `plots_manifest.json` file, no
`nv render` step, and no per-graph disk materialization. It's safe and expected to start
`nv serve` **before or during** a long `nv run`/`nv groups ... --repeats N` job (the cache is
WAL-mode SQLite, so concurrent reads while the job writes are fine) and just reload the browser
tab to see new repeats land. "Checking on a run" is therefore two separate questions: *how far
along is it* (cheap, cache-only, no browser needed) and *does its output actually look right*
(needs a real browser check, but no rebuild step first).

## 1. How far along is it — `nv cache progress`

```bash
uv run nv cache progress --repeats 50
```

Reads `achieved_repeats` directly off each combination's cache pointer — safe to run
concurrently with a live `nv run`/`nv groups` (read-only, no lock contention with the writer).
Filter with `--strategy` / `--generator` / `--noise` / `--category` to narrow it down;
add `--verbose` to see every combo instead of just the ones still behind target.

```
900 combination(s) found, 12398 repeats completed so far.
Target: 50 repeats/combo -> 45000 total repeats. 27.6% complete (173/900 combinations fully done).
```

Don't use `nv cache list` for this — it groups by the exact `achieved_repeats` value, so a
run in progress (where every combo is at a different count) explodes into one row per distinct
count and is unreadable. Don't write an ad-hoc script to query `artifacts/cache/*.db` either —
`nv cache progress` already does this safely and correctly; reach for it first.

## 2. Seeing the actual graphs mid-run

```bash
uv run nv serve       # start any time before or during nv run / nv groups
```

`nv serve`'s manifest is built lazily from the cache on first request, then held in an
in-process cache (`nvision/cli/api_server.py`'s `cache["manifest"]`) so repeat page loads don't
re-walk the whole cache every time. That means a plain browser refresh (F5) will **not** show
repeats that landed after the manifest was built — you need to invalidate it first:

- Press **'r'** in the UI tab (or click the "Recalculate" button) — this POSTs `/api/reload`,
  which drops the in-process cache and rebuilds it from current cache state, then reloads the
  page automatically once done.
- Equivalently: `curl -X POST http://localhost:PORT/api/reload`, then reload the tab yourself.

No `nv render` step is needed for the UI at all anymore — `nv render` still exists but now only
writes `locator_results.csv`; it's unrelated to what `nv serve` shows.

## 3. Two things that look like bugs but are just scale

**The server itself starts fast** (no more pre-scan on startup — `nv serve` opens the port
almost immediately, even against a huge/in-progress cache). But **the first `/api/manifest`
request after startup or a reload can take a while** on a big cache: `_build_manifest`
(`nvision/cli/api_server.py`) walks every cached combination once, in a thread pool, to build
the response. On hundreds of combos / tens of thousands of repeats this first load (or first
load after pressing 'r') can take real time even though `/api/status` already returns 200. If
you need to confirm the server is up before checking the page, poll `/api/status`, but don't
mistake a slow first manifest load for the server being down:

```bash
until curl -s -o /dev/null -w "%{http_code}" http://localhost:PORT/api/status | grep -q 200; do sleep 3; done
```

(Use the `Monitor` tool with this as the command so you get a single notification instead of
polling manually.)

**A very large manifest can make the browser tools' `screenshot`/`computer` actions time out**
even though the page loaded fine. Past roughly tens of thousands of manifest entries, the
frontend fetches it as plain JSON (`bootstrap.js`: "Large manifest detected, fetching via
JSON...") and the parse/index step can pin the render thread long enough that CDP screenshot
capture times out. This is a rendering-pane hiccup, not proof the page is broken. Verify with
text/JS-based tools instead, which keep working:

- `get_page_text` — human-readable confirmation of real numbers/labels
- `read_network_requests` filtered to the plot paths you care about — look for `200 OK`, not `404`
- `javascript_tool`:
  ```js
  JSON.stringify({
    failedText: document.body.innerText.includes('Failed to load plot'),
    plotlyDivs: document.querySelectorAll('.js-plotly-plot').length,
  })
  ```
  `failedText: false` and a nonzero `plotlyDivs` count is real proof; a `screenshot` timeout
  alone is not evidence of a problem.
- If `read_page`/`computer` refs come back empty on a stuck-looking tab, click via JS instead
  of the `computer` tool: `document.querySelector('[role=radio][data-value="..."]').click()`
  still fires the app's real event handlers.

## 4. "Ensuring it works properly" — the actual verification recipe

Code review of the plotting/caching path is not verification. Don't declare a fix or an
in-progress combo's output "working" without doing this, in order:

1. **DB check**: query the relevant combo's cache row directly and confirm its plot entries have
   `content_bin` (or `content`) populated — not just a `path`. A `path` with no content means the
   file was never actually captured, and will 404 regardless of what the manifest says
   (`/api/graph/...` reads straight from this field — see `_graph_bytes_for_entry` in
   `nvision/cli/api_server.py`).
2. **Serve**: start (or reuse) `nv serve`. If it was already running before the repeat you care
   about landed, press 'r' in the tab (or POST `/api/reload`) to invalidate the in-process
   manifest cache so the new repeat is actually included (§2 above).
3. **Browser proof**: select the specific (generator, noise, strategy, repeat) combo you're
   checking, then confirm via §3's text/JS techniques — real network 200s, no "Failed to load
   plot" text, nonzero Plotly div count, and (via `get_page_text`) numbers that look like real
   physics output, not placeholders.

Skipping straight to "the code looks right" or "the process is still running" is not the same
claim as "the graphs render" — this repo has already hit a case where a cache-embedding bug
went unnoticed for ~99% of a 44-hour run's output because nothing checked step 1.

## 5. If a purge/cleanup is ever needed at scale

`nv cache clean` (and the `purge_cached_combination` it calls) rescans the *entire* cache
backend once per matched combination — `O(combos × total_cache_rows)`. Fine for a handful of
combos; at hundreds of combos against tens of thousands of cache rows it can take hours and
looks hung. If you genuinely need to bulk-purge many combinations (not a targeted `--generator`/
`--strategy` filter matching a few), a direct single-pass SQL delete against the SQLite file is
dramatically faster (verified: ~5 seconds vs. multi-hour for ~83k rows / 816 combos) — identify
every pointer key matching your target configs in one scan, then `DELETE FROM cache WHERE key IN
(...)` in chunks. Only do this with explicit user sign-off; it bypasses the CLI's normal
confirmation/dry-run safety net, so show `--dry-run` counts from `nv cache clean` first to agree
on scope before switching to the fast path.
