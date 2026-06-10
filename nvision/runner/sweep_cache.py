"""Multi-process shared cache for sweep observations.

Prevents redundant sweep measurements across multiple repeats of the same experiment.
Uses multiprocessing.shared_memory for cross-process efficiency with lock-free reading.
"""

from __future__ import annotations

import logging
import pickle
import random
import struct
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any

import numpy as np

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.models.observation import Observation

if TYPE_CHECKING:
    from nvision.models.experiment import CoreExperiment
    from nvision.models.locator import Locator

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SWEEP_OBSERVATIONS_BY_KEY: dict[str, list[Observation]] = {}

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
_DESERIALIZED_CACHE: dict[str, list[Observation]] = {}


def initialize_shared_sweep_cache(shm_name: str | None, lock: Any | None) -> None:
    """Attach this process to the shared memory sweep cache."""
    global _SHM, _SHM_LOCK
    if shm_name is not None:
        try:
            _SHM = shared_memory.SharedMemory(name=shm_name)
            _SHM_LOCK = lock
        except Exception:
            _SHM = None
            _SHM_LOCK = None


def _sync_from_shm(target_key: str | None = None) -> list[Observation] | None:
    """Update local cache from shared memory if a newer version exists."""
    global _LOCAL_VERSION
    if _SHM is None:
        return None

    try:
        shm_version = struct.unpack("<I", _SHM.buf[:4])[0]
        if shm_version == _LOCAL_VERSION and target_key in _DESERIALIZED_CACHE:
            return _DESERIALIZED_CACHE[target_key]

        # Scan index
        count = struct.unpack("<I", _SHM.buf[8:12])[0]

        target_hash = hash(target_key) & 0xFFFF_FFFF_FFFF_FFFF if target_key else None
        found_obs = None

        for i in range(count):
            base = 12 + i * ENTRY_SIZE
            entry_hash, offset, length = struct.unpack("<QII", _SHM.buf[base : base + 16])

            if target_hash is not None and entry_hash == target_hash:
                pickled_data = _SHM.buf[offset : offset + length]
                found_obs = pickle.loads(pickled_data)
                _DESERIALIZED_CACHE[target_key] = found_obs
                break

        _LOCAL_VERSION = shm_version
        return found_obs
    except (struct.error, pickle.UnpicklingError, ValueError, BufferError, OSError) as exc:
        log.debug("Failed to sync sweep cache from SHM (expected during shutdown): %s", exc)
        return None


def _write_to_shm(key: str, observations: list[Observation]) -> None:
    """Serialize a single sweep to SHM and update index."""
    if _SHM is None or _SHM_LOCK is None:
        return

    try:
        pickled_data = pickle.dumps(observations)
        length = len(pickled_data)

        # Get current state from header
        version, next_offset, count = struct.unpack("<III", _SHM.buf[:12])

        if count >= MAX_ENTRIES:
            return
        if next_offset + length > _SHM.size:
            log.warning("Sweep cache SHM buffer too small")
            return

        # 1. Write data
        _SHM.buf[next_offset : next_offset + length] = pickled_data

        # 2. Write index entry
        base = 12 + count * ENTRY_SIZE
        _SHM.buf[base : base + 16] = struct.pack("<QII", hash(key) & 0xFFFF_FFFF_FFFF_FFFF, next_offset, length)

        # 3. Update header (version last)
        _SHM.buf[4:12] = struct.pack("<II", next_offset + length, count + 1)
        _SHM.buf[:4] = struct.pack("<I", version + 1)

        global _LOCAL_VERSION
        _LOCAL_VERSION = version + 1
    except (ValueError, BufferError, OSError, EOFError, RuntimeError) as exc:
        log.debug("Failed to write sweep cache to SHM (expected during shutdown): %s", exc)
    except Exception:
        log.exception("Failed to write sweep cache to SHM")


def _sweep_cache_key(experiment: CoreExperiment, sweep_steps: int) -> str:
    """Generate cache key from experiment characteristics."""
    sig_bounds = getattr(experiment.true_signal, "bounds", {})
    noise_name = experiment.noise.__class__.__name__ if experiment.noise else "none"
    noise_seed = getattr(experiment.noise, "seed", "noseed") if experiment.noise else "noseed"

    # Extract noise standard deviation to separate dynamic levels
    noise_std = 0.0
    if experiment.noise is not None:
        if hasattr(experiment.noise, "estimated_noise_std"):
            noise_std = experiment.noise.estimated_noise_std()
        elif hasattr(experiment.noise, "noise_std"):
            noise_std = experiment.noise.noise_std()

    param_values = getattr(experiment.true_signal, "parameter_values", lambda: {})()
    return (
        f"sweep:{experiment.x_min:.9f}:{experiment.x_max:.9f}:"
        f"{sweep_steps}:{noise_name}:{noise_seed}:{noise_std:.6f}:"
        f"{hash(str(sorted(sig_bounds.items())))}:{hash(str(sorted(param_values.items())))}"
    )


def get_cached_sweep(experiment: CoreExperiment, sweep_steps: int) -> list[Observation] | None:
    """Retrieve cached sweep observations if available."""
    key = _sweep_cache_key(experiment, sweep_steps)
    cached = _DESERIALIZED_CACHE.get(key)
    if cached is None:
        cached = _sync_from_shm(key)
    return cached


def put_cached_sweep(experiment: CoreExperiment, sweep_steps: int, observations: list[Observation]) -> None:
    """Store sweep observations in shared cache."""
    key = _sweep_cache_key(experiment, sweep_steps)

    if _SHM_LOCK is not None:
        try:
            with _SHM_LOCK:
                cached = _sync_from_shm(key)
                if cached is None:
                    _DESERIALIZED_CACHE[key] = observations
                    _write_to_shm(key, observations)
        except (OSError, EOFError, RuntimeError) as exc:
            log.debug("Shared lock connection closed in put_cached_sweep: %s", exc)
            with _LOCK:
                if key not in _SWEEP_OBSERVATIONS_BY_KEY:
                    _SWEEP_OBSERVATIONS_BY_KEY[key] = observations
                    _DESERIALIZED_CACHE[key] = observations
    else:
        with _LOCK:
            if key not in _SWEEP_OBSERVATIONS_BY_KEY:
                _SWEEP_OBSERVATIONS_BY_KEY[key] = observations


def has_cached_sweep(experiment: CoreExperiment, sweep_steps: int) -> bool:
    """Check if sweep is cached for this experiment."""
    key = _sweep_cache_key(experiment, sweep_steps)
    if key in _DESERIALIZED_CACHE:
        return True
    return _sync_from_shm(key) is not None


def _sobol_baseline_cache_key(
    experiment: CoreExperiment, seed: int, generator_name: str, noise_name: str, repeat_idx: int
) -> str:
    """Generate cache key for Sobol baseline steps."""
    return f"sobol_baseline:{seed}:{generator_name}:{noise_name}:{repeat_idx}"


def get_cached_sobol_baseline(
    experiment: CoreExperiment, seed: int, generator_name: str, noise_name: str, repeat_idx: int
) -> dict[str, int | None] | None:
    """Retrieve cached Sobol baseline steps if available."""
    key = _sobol_baseline_cache_key(experiment, seed, generator_name, noise_name, repeat_idx)
    cached = _DESERIALIZED_CACHE.get(key)
    if cached is None:
        cached = _sync_from_shm(key)
    if cached is None:
        return None
    if isinstance(cached, int):
        # Backward compatibility for old caches storing only the overall step count
        return {"sobol_baseline_steps": cached, "sobol_freq_steps": None}
    return cached


def put_cached_sobol_baseline(
    experiment: CoreExperiment,
    seed: int,
    generator_name: str,
    noise_name: str,
    repeat_idx: int,
    steps: dict[str, int | None],
) -> None:
    """Store Sobol baseline steps in shared cache."""
    key = _sobol_baseline_cache_key(experiment, seed, generator_name, noise_name, repeat_idx)

    if _SHM_LOCK is not None:
        try:
            with _SHM_LOCK:
                cached = _sync_from_shm(key)
                if cached is None:
                    _DESERIALIZED_CACHE[key] = steps
                    _write_to_shm(key, steps)  # type: ignore
        except (OSError, EOFError, RuntimeError) as exc:
            log.debug("Shared lock connection closed in put_cached_sobol_baseline: %s", exc)
            with _LOCK:
                if key not in _SWEEP_OBSERVATIONS_BY_KEY:
                    _SWEEP_OBSERVATIONS_BY_KEY[key] = steps  # type: ignore
                    _DESERIALIZED_CACHE[key] = steps
    else:
        with _LOCK:
            if key not in _SWEEP_OBSERVATIONS_BY_KEY:
                _SWEEP_OBSERVATIONS_BY_KEY[key] = steps  # type: ignore
                _DESERIALIZED_CACHE[key] = steps


def _simplesweep_baseline_cache_key(
    experiment: CoreExperiment, seed: int, generator_name: str, noise_name: str, repeat_idx: int
) -> str:
    return f"simplesweep_baseline:{seed}:{generator_name}:{noise_name}:{repeat_idx}"


def get_cached_simplesweep_baseline(
    experiment: CoreExperiment, seed: int, generator_name: str, noise_name: str, repeat_idx: int
) -> dict | None:
    key = _simplesweep_baseline_cache_key(experiment, seed, generator_name, noise_name, repeat_idx)
    cached = _DESERIALIZED_CACHE.get(key)
    if cached is None:
        cached = _sync_from_shm(key)
    return cached


def put_cached_simplesweep_baseline(
    experiment: CoreExperiment,
    seed: int,
    generator_name: str,
    noise_name: str,
    repeat_idx: int,
    data: dict,
) -> None:
    key = _simplesweep_baseline_cache_key(experiment, seed, generator_name, noise_name, repeat_idx)
    if _SHM_LOCK is not None:
        try:
            with _SHM_LOCK:
                cached = _sync_from_shm(key)
                if cached is None:
                    _DESERIALIZED_CACHE[key] = data
                    _write_to_shm(key, data)
        except (OSError, EOFError, RuntimeError) as exc:
            log.debug("Shared lock connection closed in put_cached_simplesweep_baseline: %s", exc)
            with _LOCK:
                if key not in _SWEEP_OBSERVATIONS_BY_KEY:
                    _SWEEP_OBSERVATIONS_BY_KEY[key] = data  # type: ignore
                    _DESERIALIZED_CACHE[key] = data
    else:
        with _LOCK:
            if key not in _SWEEP_OBSERVATIONS_BY_KEY:
                _SWEEP_OBSERVATIONS_BY_KEY[key] = data  # type: ignore
                _DESERIALIZED_CACHE[key] = data


def clear_sweep_cache() -> None:
    """Clear all cached sweep observations."""
    with _LOCK:
        _SWEEP_OBSERVATIONS_BY_KEY.clear()
        _DESERIALIZED_CACHE.clear()
        if _SHM_LOCK is not None:
            try:
                with _SHM_LOCK:
                    if _SHM:
                        _SHM.buf[:12] = struct.pack("<III", 0, HEADER_SIZE, 0)
            except (OSError, EOFError, RuntimeError) as exc:
                log.debug("Shared lock connection closed in clear_sweep_cache: %s", exc)


@dataclass
class SweepCache:
    """Per-task sweep cache wrapper that uses the shared cache."""

    def get(self, experiment: CoreExperiment, sweep_steps: int) -> list[Observation] | None:
        return get_cached_sweep(experiment, sweep_steps)

    def put(self, experiment: CoreExperiment, sweep_steps: int, observations: list[Observation]) -> None:
        put_cached_sweep(experiment, sweep_steps, observations)

    def has(self, experiment: CoreExperiment, sweep_steps: int) -> bool:
        return has_cached_sweep(experiment, sweep_steps)

    def get_sobol_baseline(
        self, experiment: CoreExperiment, seed: int, generator_name: str, noise_name: str, repeat_idx: int
    ) -> dict[str, int | None] | None:
        return get_cached_sobol_baseline(experiment, seed, generator_name, noise_name, repeat_idx)

    def put_sobol_baseline(
        self,
        experiment: CoreExperiment,
        seed: int,
        generator_name: str,
        noise_name: str,
        repeat_idx: int,
        steps: dict[str, int | None],
    ) -> None:
        put_cached_sobol_baseline(experiment, seed, generator_name, noise_name, repeat_idx, steps)

    def get_simplesweep_baseline(
        self, experiment: CoreExperiment, seed: int, generator_name: str, noise_name: str, repeat_idx: int
    ) -> dict | None:
        return get_cached_simplesweep_baseline(experiment, seed, generator_name, noise_name, repeat_idx)

    def put_simplesweep_baseline(
        self,
        experiment: CoreExperiment,
        seed: int,
        generator_name: str,
        noise_name: str,
        repeat_idx: int,
        data: dict,
    ) -> None:
        put_cached_simplesweep_baseline(experiment, seed, generator_name, noise_name, repeat_idx, data)


def _create_sweep_belief(experiment: CoreExperiment) -> AbstractMarginalDistribution:
    """Create a minimal GridMarginalDistribution for sweeping locators."""
    model = experiment.true_signal.model
    param_names = model.parameter_names()

    parameters = []
    for name in param_names:
        bounds = getattr(experiment.true_signal, "bounds", {}).get(name, (experiment.x_min, experiment.x_max))
        grid = np.linspace(bounds[0], bounds[1], 64)
        parameters.append(
            GridParameter(
                name=name,
                bounds=bounds,
                grid=grid,
                posterior=np.ones(64) / 64,
            )
        )

    return GridMarginalDistribution(model=model, parameters=parameters)


def precompute_sweep(
    locator_class: type[Locator],
    experiment: CoreExperiment,
    rng: random.Random,
    sweep_cache: SweepCache,
    **locator_config: Any,
) -> list[Observation] | None:
    """Pre-generate and cache sweep observations for Bayesian and sweep locators."""
    from nvision.sim.locs.bayesian.sequential_bayesian_locator import SequentialBayesianLocator
    from nvision.sim.locs.coarse.sweep_locator import SweepingLocator

    is_bayesian = issubclass(locator_class, SequentialBayesianLocator)
    is_sweep = issubclass(locator_class, SweepingLocator)
    if not (is_bayesian or is_sweep):
        return None

    needs_belief = getattr(locator_class, "REQUIRES_BELIEF", False)
    if needs_belief and ("belief" not in locator_config or "signal_model" not in locator_config):
        locator_config.setdefault("belief", _create_sweep_belief(experiment))
        locator_config.setdefault("signal_model", experiment.true_signal.model)

    locator = locator_class.create(**locator_config)

    sweep_steps = getattr(locator, "initial_sweep_steps", 0) if is_bayesian else getattr(locator, "max_steps", 0)

    if sweep_steps <= 0:
        return None

    if sweep_cache.has(experiment, sweep_steps):
        return sweep_cache.get(experiment, sweep_steps)

    observations: list[Observation] = []
    step = 0
    while not locator.done() and step < sweep_steps:
        # Note: _check_memory_limit is in executor.py, we don't have it here.
        # If memory becomes an issue in the pre-computation, we'll need to move it or pass the check.
        step += 1
        x_current = locator.next()
        obs = experiment.measure(x_current, rng)
        locator.observe(obs)
        observations.append(obs)

    if observations:
        sweep_cache.put(experiment, sweep_steps, observations)

    return observations
