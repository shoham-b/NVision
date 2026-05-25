"""Semantic cache configs for locator runs (business keys, not storage)."""

from __future__ import annotations

from typing import Any

# Bump when repeat / cache payload semantics change.
# v3: signal generation uses strategy-independent RNG; prior cache entries are not comparable.
# v4: same ground-truth draw for all noise models (noise only affects measurement RNG stream).
# v5: NV Bayesian belief uses unit-cube parameter grids + physical signal wrapper (likelihood x-mapping).
# v6: Streaming repeat cache (pointer rows + separate repeat rows).
# v8: Standardized de-duplication attempt keys (attempt instead of repeat_id) and gap-resilient partial result harvesting.
CACHE_SCHEMA_VERSION = 8


def combination_base_cache_config(
    *,
    generator: str,
    noise: str,
    strategy: str,
    seed: int,
    max_steps: int,
    timeout_s: int,
    repeat_offset: int = 0,
) -> dict[str, Any]:
    """Base config for a combination, independent of the number of repeats (streaming pointer).

    When a combination's repeats are split across multiple sub-tasks (work-stealing),
    ``repeat_offset`` is set to 0 globally in the configuration dictionary so that all
    sub-tasks share the unified cache pointers.
    """
    d: dict[str, Any] = {
        "kind": "locator_combination_pointer",
        "schema_version": CACHE_SCHEMA_VERSION,
        "generator": generator,
        "noise": noise,
        "strategy": strategy,
        "seed": seed,
        "max_steps": max_steps,
        "timeout_s": timeout_s,
        "repeat_offset": 0,
    }
    return d


def locator_combination_cache_config(
    *,
    generator: str,
    noise: str,
    strategy: str,
    repeats: int,
    seed: int,
    max_steps: int,
    timeout_s: int,
    repeat_offset: int = 0,
) -> dict[str, Any]:
    """Config dict for a full (generator, noise, strategy) combination (matches executor + render).

    When a combination's repeats are split across multiple sub-tasks (work-stealing),
    ``repeat_offset`` is set to 0 globally in the configuration dictionary so that all
    sub-tasks share the unified cache pointers.
    """
    d: dict[str, Any] = {
        "kind": "locator_combination",
        "schema_version": CACHE_SCHEMA_VERSION,
        "generator": generator,
        "noise": noise,
        "strategy": strategy,
        "repeats": repeats,
        "seed": seed,
        "max_steps": max_steps,
        "timeout_s": timeout_s,
        "repeat_offset": 0,
    }
    return d
