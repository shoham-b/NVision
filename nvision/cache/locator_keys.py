"""Semantic cache configs for locator runs (business keys, not storage)."""

from __future__ import annotations

from typing import Any

from nvision.sim.defaults import NVISION_SWEEP_FIT_ONLY
from nvision.sim.grid_enums import is_plain_sweep
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
# v11: Algorithm changes that make older results non-comparable (candidate-grid
# density mixture, Rao-Blackwellized noise likelihood, no particle rejuvenation,
# noise-floor fix, prior-mean widening). The viewer (`nv serve`) still opens
# older entries via their stored schema_version (see api_server._combo_key).
# v12: the dip-centre model parameter was renamed ``frequency`` -> ``center_freq`` and the
# milestone metrics fb/fc/splitting_converged -> primary/split/primary_converged. Stored
# true_params, bounds, scan entries and metric columns carry the old names, which the viewer
# (static/) no longer understands, so unlike v11 and earlier these entries are NOT viewable:
# ``nv serve`` skips anything older than MIN_VIEWABLE_CACHE_SCHEMA_VERSION with a warning.
# Re-run (not just ``nv render``) to regenerate them.
# v13: the probe axis (``x``, ``x_min``/``x_max``, ``probe_*``, ``candidate_x*``) became ``drive_freq``
# (``drive_freq_min_phys``/``drive_freq_max_phys``, ``candidate_drive_freq_*``, ``drive_freq_phys`` in the
# MATLAB stats payload, ``matlab_drive_freq_stats``) and the locator config key ``probe_axis_param`` became
# ``center_param``. Stored experiment/spec fields, locator configs and plot payloads carry the old names,
# so older entries are hidden by ``nv serve`` (see v12); re-run, not just ``nv render``.
# v14: physical parameter renames: ``center_freq`` -> ``zero_field_freq``, ``linewidth`` -> ``hwhm``,
# ``homogeneous_linewidth`` -> ``hwhm_lorentz``, ``zeeman_split`` -> ``zeeman_shift``, hyperfine ``split`` ->
# ``hyperfine_shift``, ``dip_depth`` -> ``contrast``, ``c_total`` -> ``contrast_total`` (and the ``*_split``
# milestone metrics -> ``*_shift``, ``center_param`` -> ``zero_field_param``). Stored params, bounds, metric
# columns and plot payloads carry the old names, so older entries are hidden by ``nv serve`` (see v12);
# re-run, not just ``nv render``.
CACHE_SCHEMA_VERSION = 14

# Oldest cache schema the current viewer can still read (see v14 above).
MIN_VIEWABLE_CACHE_SCHEMA_VERSION = 14

# PHYSICS_CONFIG_FINGERPRINT (nvision/spectra/nv_center.py) is folded into every cache
# config below instead of relying on a manual CACHE_SCHEMA_VERSION bump: a generator's
# *name* (e.g. "NVCenter-lorentzian-w0.50MHz-c0.10") doesn't change when the underlying
# physical bounds/constants it draws from do (domain delta, Zeeman/hyperfine/hwhm
# ranges, zero-field splitting), so cache entries keyed only by name would keep matching
# and silently serving stale true-signal draws generated under the old physics forever.
# Folding it in busts every cached entry automatically on such a change, the same way
# editing this file's CACHE_SCHEMA_VERSION does -- but without a human needing to remember to.


def _physics_fingerprint_for(strategy: str) -> str:
    """The physics fingerprint folded into ``strategy``'s cache keys.

    The plain sweeps infer with a least-squares fit instead of an SMC belief when ``NVISION_SWEEP_FIT_ONLY`` is
    set; their results are not comparable to the SMC sweep's, so they are keyed apart from them.
    """
    if NVISION_SWEEP_FIT_ONLY and is_plain_sweep(strategy):
        return f"{PHYSICS_CONFIG_FINGERPRINT}+sweep-fit-only"
    return PHYSICS_CONFIG_FINGERPRINT


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
        "physics_fingerprint": _physics_fingerprint_for(strategy),
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
        "physics_fingerprint": _physics_fingerprint_for(strategy),
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
