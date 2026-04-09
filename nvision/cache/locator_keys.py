"""Semantic cache configs for locator runs (business keys, not storage)."""

from __future__ import annotations

from typing import Any

# Bump when repeat / cache payload semantics change.
# v3: signal generation uses strategy-independent RNG; prior cache entries are not comparable.
# v4: same ground-truth draw for all noise models (noise only affects measurement RNG stream).
# v5: NV Bayesian belief uses unit-cube parameter grids + physical signal wrapper (likelihood x-mapping).
CACHE_SCHEMA_VERSION = 5


def locator_combination_cache_config(
    *,
    generator: str,
    noise: str,
    strategy: str,
    repeats: int,
    seed: int,
    max_steps: int,
    timeout_s: int,
) -> dict[str, Any]:
    """Config dict for a full (generator, noise, strategy) combination (matches executor + render)."""
    return {
        "kind": "locator_combination",
        "schema_version": CACHE_SCHEMA_VERSION,
        "generator": generator,
        "noise": noise,
        "strategy": strategy,
        "repeats": repeats,
        "seed": seed,
        "max_steps": max_steps,
        "timeout_s": timeout_s,
    }


def locator_repeat_cache_config(
    *,
    generator: str,
    noise: str,
    strategy: str,
    repeat: int,
    seed: int,
    max_steps: int,
    timeout_s: int,
) -> dict[str, Any]:
    """Config dict for a single repeat slice (per-repeat cache rows)."""
    return {
        "kind": "locator_run",
        "schema_version": CACHE_SCHEMA_VERSION,
        "generator": generator,
        "noise": noise,
        "strategy": strategy,
        "repeat": repeat,
        "seed": seed,
        "max_steps": max_steps,
        "timeout_s": timeout_s,
    }
