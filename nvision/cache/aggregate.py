"""Cache-wide reconstruction of the flat locator-results table for aggregate views.

Rebuilds the same row shape ``nv render`` writes to ``locator_results.csv`` — one row
per repeat, flat scalar columns (``abs_err_x``, ``splitting_converged_step``, ...) — directly
from the cache's cheap ``:meta`` sidecars (``RepeatsRepository.load_repeats_meta``), so
aggregate views (comparisons, grid-study, experiment summaries) can be served live
without touching disk or loading any ``content_bin`` graph bytes.
"""

from __future__ import annotations

import polars as pl

from nvision.cache.bridge import CacheBridge
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import combination_base_cache_config
from nvision.sim.combinations import CombinationGrid


def build_locator_results_rows(bridge: CacheBridge) -> list[dict]:
    """Return one flat dict per cached repeat, across every combination in *bridge*."""
    rows: list[dict] = []
    for combo in bridge.list_combinations():
        category = CombinationGrid.generator_category(str(combo["generator"]))
        repo = bridge.get_cache_for_category(category)

        ptr_config = combination_base_cache_config(
            generator=combo["generator"],
            noise=combo["noise"],
            strategy=combo["strategy"],
            seed=combo["seed"],
            max_steps=combo["max_steps"],
            timeout_s=combo["timeout_s"],
        )
        combo_key = stable_config_hash(ptr_config)

        achieved = int(combo.get("repeats", 0))
        if achieved <= 0:
            continue
        meta = repo._repeats.load_repeats_meta(combo_key, achieved)
        if meta is None:
            # No :meta sidecars (pre-sidecar cache entries) — fall back to the full load.
            meta = repo._repeats.load_repeats(combo_key, achieved)
        for _entries, main_result_row in meta:
            rows.append(main_result_row)
    return rows


def build_locator_results_df(bridge: CacheBridge) -> pl.DataFrame:
    """Polars DataFrame equivalent of ``locator_results.csv``, built live from the cache."""
    rows = build_locator_results_rows(bridge)
    return pl.from_dicts(rows, infer_schema_length=None) if rows else pl.DataFrame()
