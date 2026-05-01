"""Tests for caching utilities."""

from __future__ import annotations

from nvision.runner.cache import strip_heavy_fields


def test_strip_heavy_fields() -> None:
    """Ensure heavy fields are removed from the dictionary."""
    entry = {
        "id": 1,
        "name": "test",
        "content": "heavy_content_here",
        "plot_data": {"x": [1, 2, 3], "y": [4, 5, 6]},
        "other_field": "keep_this",
    }

    result = strip_heavy_fields(entry)

    # Assert original dict is unmodified
    assert "content" in entry
    assert "plot_data" in entry

    # Assert stripped dict is correct
    assert result == {
        "id": 1,
        "name": "test",
        "other_field": "keep_this",
    }
    assert "content" not in result
    assert "plot_data" not in result


def test_strip_heavy_fields_missing_keys() -> None:
    """Ensure function works fine when heavy fields are not present."""
    entry = {
        "id": 2,
        "name": "test2",
    }

    result = strip_heavy_fields(entry)

    assert result == {
        "id": 2,
        "name": "test2",
    }
