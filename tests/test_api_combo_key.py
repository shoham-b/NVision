"""The viewer must rebuild the key an entry was *stored* under, not the current code's."""

from __future__ import annotations

from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import (
    CACHE_SCHEMA_VERSION,
    PHYSICS_CONFIG_FINGERPRINT,
    combination_base_cache_config,
)
from nvision.cli.api_server import _combo_key

_BASE = {
    "generator": "NVCenter-lorentzian",
    "noise": "Gauss(0.01)",
    "strategy": "Bayesian-SBED",
    "seed": 1,
    "max_steps": 100,
    "timeout_s": 60,
}


def _stored_key(*, schema_version: int, fingerprint: str | None) -> str:
    cfg = combination_base_cache_config(**_BASE)
    cfg["schema_version"] = schema_version
    if fingerprint is None:
        cfg.pop("physics_fingerprint")
    else:
        cfg["physics_fingerprint"] = fingerprint
    return stable_config_hash(cfg)


def test_combo_key_uses_stored_schema_version_and_fingerprint():
    old_schema = CACHE_SCHEMA_VERSION - 1
    combo = {**_BASE, "physics_fingerprint": "oldfingerp12", "schema_version": old_schema}
    assert _combo_key(combo) == _stored_key(schema_version=old_schema, fingerprint="oldfingerp12")


def test_combo_key_current_entry_matches_current_code():
    combo = {
        **_BASE,
        "physics_fingerprint": PHYSICS_CONFIG_FINGERPRINT,
        "schema_version": CACHE_SCHEMA_VERSION,
    }
    assert _combo_key(combo) == _stored_key(schema_version=CACHE_SCHEMA_VERSION, fingerprint=PHYSICS_CONFIG_FINGERPRINT)


def test_combo_key_differs_between_schema_versions():
    a = _combo_key({**_BASE, "physics_fingerprint": "x", "schema_version": 10})
    b = _combo_key({**_BASE, "physics_fingerprint": "x", "schema_version": 11})
    assert a != b
