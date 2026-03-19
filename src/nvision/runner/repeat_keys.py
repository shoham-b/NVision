"""Stable repeat keys and integer seeds for signal vs measurement RNGs."""

from __future__ import annotations

import hashlib


def repeat_seed_int(seed_str: str) -> int:
    """Stable 32-bit-ish seed from a string (matches measurement / signal RNG derivation)."""
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (10**8)


def signal_repeat_key(seed: int, generator_name: str, repeat_idx: int) -> str:
    """Key for ground-truth signal generation — strategy- and noise-independent."""
    return f"{seed}-{generator_name}-{repeat_idx}"


def measurement_repeat_key(seed: int, generator_name: str, strategy_name: str, noise_name: str, repeat_idx: int) -> str:
    """Key for measurement noise RNG — strategy-specific."""
    return f"{seed}-{generator_name}-{strategy_name}-{noise_name}-{repeat_idx}"
