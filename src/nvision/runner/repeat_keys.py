"""Stable repeat keys and integer seeds for signal vs measurement RNGs."""

from __future__ import annotations

import hashlib
from functools import lru_cache


@lru_cache(maxsize=16_384)
def repeat_seed_int(seed_str: str) -> int:
    """Stable 32-bit-ish seed from a string (matches measurement / signal RNG derivation).

    Cached: the same key string is hashed often (measurement RNG per repeat); ``maxsize`` bounds
    memory if many unique keys appear (e.g. fuzzing).
    """
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (10**8)


def clear_repeat_seed_int_cache() -> None:
    """Drop :func:`repeat_seed_int` cache (e.g. between tests)."""
    repeat_seed_int.cache_clear()


def signal_repeat_key(seed: int, generator_name: str, repeat_idx: int) -> str:
    """Key for ground-truth signal generation — strategy- and noise-independent."""
    return f"{seed}-{generator_name}-{repeat_idx}"


def measurement_repeat_key(seed: int, generator_name: str, strategy_name: str, noise_name: str, repeat_idx: int) -> str:
    """Key for measurement noise RNG — strategy-specific."""
    return f"{seed}-{generator_name}-{strategy_name}-{noise_name}-{repeat_idx}"
