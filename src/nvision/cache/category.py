"""Backward-compatible re-exports (prefer :class:`LocatorResultsRepository` / :class:`CategoryDataStore`)."""

from __future__ import annotations

from nvision.cache.data_store import CategoryDataStore
from nvision.cache.locator_repository import LocatorResultsRepository

# Historical name used by tests and external imports.
CategoryCache = LocatorResultsRepository

__all__ = ["CategoryCache", "CategoryDataStore", "LocatorResultsRepository"]
