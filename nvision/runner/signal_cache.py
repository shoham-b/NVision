"""Process-local cache so the same ``true_signal`` (and domain) is reused for every locator task that
shares the same ``(seed, generator_name, repeat_idx)``.

Each task still gets its own :class:`~nvision.models.experiment.CoreExperiment` with the correct
``noise`` model; only the ground-truth draw is shared. That aligns cross-noise and cross-strategy
comparisons on identical underlying signals.
"""

from __future__ import annotations

import logging
import pickle
import random
import struct
import threading
from collections.abc import Callable
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any

from nvision.models.experiment import CoreExperiment
from nvision.runner.repeat_keys import repeat_seed_int, signal_repeat_key
from nvision.spectra.signal import TrueSignal

if TYPE_CHECKING:
    from nvision.models.task import LocatorTask

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SIGNAL_BUNDLE_BY_KEY: dict[str, tuple[TrueSignal, float, float]] = {}

# Multi-process shared memory cache
# Layout: [Header 4KB][Data ...]
# Header: [version: 4B][next_offset: 4B][count: 4B][entries: count * 32B]
# Entry: [hash: 8B][offset: 4B][length: 4B][reserved: 16B]
_SHM: shared_memory.SharedMemory | None = None
_SHM_LOCK: Any = None
_LOCAL_VERSION: int = -1

HEADER_SIZE = 4096
ENTRY_SIZE = 32
MAX_ENTRIES = (HEADER_SIZE - 12) // ENTRY_SIZE

# Local cache of DESERIALIZED objects to avoid repeated unpickling
_DESERIALIZED_CACHE: dict[str, tuple[TrueSignal, float, float]] = {}


def initialize_shared_cache(shm_name: str | None, lock: Any | None) -> None:
    """Attach this process to the shared memory cache."""
    global _SHM, _SHM_LOCK
    if shm_name is not None:
        try:
            _SHM = shared_memory.SharedMemory(name=shm_name)
            _SHM_LOCK = lock
        except Exception:
            _SHM = None
            _SHM_LOCK = None


def _sync_from_shm(target_key: str | None = None) -> tuple[TrueSignal, float, float] | None:
    """Sync local state from SHM. If target_key provided, try to retrieve it surgically."""
    global _LOCAL_VERSION
    if _SHM is None:
        return None

    try:
        shm_version = struct.unpack("<I", _SHM.buf[:4])[0]
        if shm_version == _LOCAL_VERSION and target_key in _DESERIALIZED_CACHE:
            return _DESERIALIZED_CACHE[target_key]

        # Scan index
        count = struct.unpack("<I", _SHM.buf[8:12])[0]
        
        target_hash = hash(target_key) if target_key else None
        found_bundle = None

        for i in range(count):
            base = 12 + i * ENTRY_SIZE
            entry_hash, offset, length = struct.unpack("<QII", _SHM.buf[base : base + 16])
            
            # If we are looking for a specific key and found its hash
            if target_hash is not None and entry_hash == target_hash:
                # To be absolutely sure about hash collisions, we'd need to store the key too.
                # But for ~100 entries, 64-bit hash is extremely safe.
                pickled_data = _SHM.buf[offset : offset + length]
                found_bundle = pickle.loads(pickled_data)
                _DESERIALIZED_CACHE[target_key] = found_bundle
                break

        _LOCAL_VERSION = shm_version
        return found_bundle
    except (struct.error, pickle.UnpicklingError):
        return None


def _write_to_shm(key: str, bundle: tuple[TrueSignal, float, float]) -> None:
    """Serialize a single bundle to SHM and update index."""
    if _SHM is None or _SHM_LOCK is None:
        return

    try:
        pickled_data = pickle.dumps(bundle)
        length = len(pickled_data)

        # Get current state from header
        version, next_offset, count = struct.unpack("<III", _SHM.buf[:12])
        
        if count >= MAX_ENTRIES:
            return
        if next_offset + length > _SHM.size:
            return

        # 1. Write data
        _SHM.buf[next_offset : next_offset + length] = pickled_data

        # 2. Write index entry
        base = 12 + count * ENTRY_SIZE
        _SHM.buf[base : base + 16] = struct.pack("<QII", hash(key), next_offset, length)

        # 3. Update header (version last)
        _SHM.buf[4:12] = struct.pack("<II", next_offset + length, count + 1)
        _SHM.buf[:4] = struct.pack("<I", version + 1)
        
        global _LOCAL_VERSION
        _LOCAL_VERSION = version + 1
    except Exception:
        pass


def clear_signal_experiment_cache() -> None:
    """Drop all cached signal bundles (local and shared)."""
    with _LOCK:
        _SIGNAL_BUNDLE_BY_KEY.clear()
        _DESERIALIZED_CACHE.clear()
        if _SHM_LOCK is not None:
            with _SHM_LOCK:
                # Reset SHM header
                if _SHM:
                    _SHM.buf[:12] = struct.pack("<III", 0, HEADER_SIZE, 0)


def get_shared_core_experiment(
    task: LocatorTask,
    repeat_idx: int,
    build: Callable[[random.Random], CoreExperiment],
) -> CoreExperiment:
    """Return a :class:`CoreExperiment` with shared ``true_signal`` for this repeat.

    On a **cache hit**, ``build`` is not called; a new experiment is returned with the cached
    signal/domain and this task's ``noise``.

    On the **first** request for a given key, ``build`` runs once; ``true_signal`` and domain
    are stored and reused for all later tasks (any strategy or noise name).

    The cache key omits ``strategy_name`` and ``noise_name`` so every combination shares one
    ground-truth draw per repeat.
    """
    key = signal_repeat_key(task.seed, task.generator_name, repeat_idx)

    # 1. Check local cache (lock-free)
    cached = _DESERIALIZED_CACHE.get(key)
    if cached is None:
        cached = _sync_from_shm(key)
    
    if cached is not None:
        true_signal, x_min, x_max = cached
        return CoreExperiment(true_signal=true_signal, noise=task.noise, x_min=x_min, x_max=x_max)

    # 2. Coordinate generation/write via locks
    if _SHM_LOCK is not None:
        with _SHM_LOCK:
            # Re-sync and re-check inside shared lock
            cached = _sync_from_shm(key)
            if cached is not None:
                true_signal, x_min, x_max = cached
                return CoreExperiment(true_signal=true_signal, noise=task.noise, x_min=x_min, x_max=x_max)

            # Generate and write to SHM
            rng = random.Random(repeat_seed_int(key))
            exp = build(rng)
            bundle = (exp.true_signal, exp.x_min, exp.x_max)
            _DESERIALIZED_CACHE[key] = bundle
            _write_to_shm(key, bundle)
            return CoreExperiment(true_signal=exp.true_signal, noise=task.noise, x_min=exp.x_min, x_max=exp.x_max)
    else:
        # Fallback to local-only logic
        with _LOCK:
            cached = _SIGNAL_BUNDLE_BY_KEY.get(key)
            if cached is not None:
                true_signal, x_min, x_max = cached
                return CoreExperiment(true_signal=true_signal, noise=task.noise, x_min=x_min, x_max=x_max)

            rng = random.Random(repeat_seed_int(key))
            exp = build(rng)
            _SIGNAL_BUNDLE_BY_KEY[key] = (exp.true_signal, exp.x_min, exp.x_max)
            return CoreExperiment(true_signal=exp.true_signal, noise=task.noise, x_min=exp.x_min, x_max=exp.x_max)
