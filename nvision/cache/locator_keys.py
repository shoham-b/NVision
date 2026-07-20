"""Semantic cache configs for locator runs (business keys, not storage)."""

from __future__ import annotations

from typing import Any

from nvision.spectra.nv_center import PHYSICS_CONFIG_FINGERPRINT

# Bump when repeat / cache payload semantics change.
# v3: signal generation uses strategy-independent RNG; prior cache entries are not comparable.
# v4: same ground-truth draw for all noise models (noise only affects measurement RNG stream).
# v5: NV Bayesian belief uses unit-cube parameter grids + physical signal wrapper (likelihood x-mapping).
# v6: Streaming repeat cache (pointer rows + separate repeat rows).
# v9: Sub-task cache slicing fix — partial loads capped to chunk_size, saves start at repeat_offset+n_cached.
# v10: Graph payloads move from base85-in-JSON (content_bin) to raw bytes in the
# sharded SQLite `graphs` BLOB table (nvision/cache/sqlite.py) — eliminates the
# ~25% base85 inflation entirely instead of just shrinking it. Old v9 entries
# are cleanly unreachable (repeat keys derive from combo_key, which changes
# with this bump) rather than needing dual-format read support; delete
# artifacts/cache to reclaim the disk they'd otherwise sit on unused.
CACHE_SCHEMA_VERSION = 10

# PHYSICS_CONFIG_FINGERPRINT (nvision/spectra/nv_center.py) is folded into every cache
# config below instead of relying on a manual CACHE_SCHEMA_VERSION bump: a generator's
# *name* (e.g. "NVCenter-lorentzian-w0.50MHz-c0.10") doesn't change when the underlying
# physical bounds/constants it draws from do (domain delta, Zeeman/hyperfine/linewidth
# ranges, zero-field splitting), so cache entries keyed only by name would keep matching
# and silently serving stale true-signal draws generated under the old physics forever.
# Folding it in busts every cached entry automatically on such a change, the same way
# editing this file's CACHE_SCHEMA_VERSION does -- but without a human needing to remember to.


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
        "physics_fingerprint": PHYSICS_CONFIG_FINGERPRINT,
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
        "physics_fingerprint": PHYSICS_CONFIG_FINGERPRINT,
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
