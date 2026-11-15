"""``/api/refine-rates``: the "Refine & speedup" section of the Information tab (see ``docs/evaluation_metrics.md``).

Per use case (generator x noise) of the selection it rebuilds Bayesian-SBED's and the SimpleSweeps' cached repeats
(:mod:`nvision.metrics.refine_rates_runs`) and returns the summary. It reads the cache only, so it works after
``nv render`` with no re-run, and is memoised per use case on the repeat counts and update times of its combos.
Repeats that cannot be rebuilt are not guessed: they are listed under ``errors`` with the reason and left out.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from nvision.metrics.refine_rates_runs import SBED, build_run_trace, use_case_summary
from nvision.sim.combinations import PLAIN_SWEEP_STRATEGIES, CombinationGrid

# Strategies in scope: Bayesian-SBED and the three sweep spacings (never the Smart-* experiments).
STRATEGIES: tuple[str, ...] = (SBED, *PLAIN_SWEEP_STRATEGIES)

_WORKERS = 4


def _use_case(
    generator: str,
    noise: str,
    combos: dict[str, dict],
    bridge: Any,
    combo_key: Callable[[dict], str],
    grid: CombinationGrid,
) -> dict[str, Any]:
    """Summary of one use case; ``{"errors": [...]}`` alone when nothing of it could be rebuilt."""
    repo = bridge.get_cache_for_category(CombinationGrid.generator_category(generator))
    resolved = grid.resolve(generator, noise, SBED)
    errors: list[dict[str, Any]] = []
    traces: dict[str, list] = {}
    for strat, combo in combos.items():
        key, n = combo_key(combo), int(combo.get("repeats", 0))
        variant = combo["strategy"]
        if resolved is None:
            errors.append(
                {"strategy": strat, "message": f"Generator {generator!r} is not registered in this code.", "count": n}
            )
            continue
        # Only SBED needs the scan figures (where it measured); a sweep's positions come from its design.
        repeats = repo._repeats.load_repeats(key, n) if strat == SBED else None
        repeats = repeats or repo._repeats.load_repeats_meta(key, n) or repo._repeats.load_repeats(key, n)
        offset = int(combo.get("repeat_offset", 0))
        failed: Counter[str] = Counter()
        traces[strat] = []
        for i, (entries, row) in enumerate(repeats):
            try:
                traces[strat].append(build_run_trace(variant, resolved.generator, generator, offset + i, entries, row))
            except ValueError as exc:
                failed[str(exc)] += 1
        errors += [{"strategy": strat, "message": m, "count": c} for m, c in failed.items()]
    try:
        summary = use_case_summary(generator, noise, traces)
    except ValueError as exc:
        return {
            "generator": generator,
            "noise": noise,
            "errors": [*errors, {"strategy": None, "message": str(exc), "count": 1}],
        }
    summary["errors"] = errors
    return summary


def add_refine_rates_route(
    app: FastAPI,
    *,
    get_combos: Callable[[], list[dict]],
    cache: dict,
    make_bridge: Callable[[], Any],
    combo_key: Callable[[dict], str],
) -> None:
    """Register ``GET /api/refine-rates?generator=a,b&noise=x,y``.

    ``cache`` is the server's cache dict: ``all_combos`` (every combo, ``-fixed`` too; filled by ``get_combos``)
    and ``refine_memo`` (the use-case summaries, cleared on reload).
    """
    memo = cache["refine_memo"]
    lock = threading.Lock()
    grid_holder: dict[str, CombinationGrid] = {}

    @app.get("/api/refine-rates")
    def refine_rates(generator: str, noise: str) -> dict:
        wanted_gen, wanted_noise = set(generator.split(",")), set(noise.split(","))
        by_case: dict[tuple[str, str], dict[str, dict]] = {}
        get_combos()  # fills cache["all_combos"]
        for c in cache["all_combos"]:
            strat = c["strategy"]
            if strat in STRATEGIES and c["generator"] in wanted_gen and c["noise"] in wanted_noise:
                by_case.setdefault((c["generator"], c["noise"]), {})[strat] = c
        with lock:
            grid = grid_holder.setdefault("grid", CombinationGrid())
        bridge = make_bridge()

        def one(item: tuple[tuple[str, str], dict[str, dict]]) -> dict[str, Any]:
            (gen, noise_name), combos = item
            stamp = tuple(sorted((s, c.get("repeats"), c.get("updated_at")) for s, c in combos.items()))
            hit = memo.get((gen, noise_name, stamp))
            if hit is None:
                hit = memo[(gen, noise_name, stamp)] = _use_case(gen, noise_name, combos, bridge, combo_key, grid)
            return hit

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=_WORKERS) as pool:
                cases = list(pool.map(one, by_case.items()))
        finally:
            bridge.close()
        return {
            "strategies": list(STRATEGIES),
            "shots_per_measurement": 1,
            "use_cases": [c for c in cases if "locators" in c],
            "failed": [c for c in cases if "locators" not in c],
        }
