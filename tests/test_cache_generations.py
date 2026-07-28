from pathlib import Path

import pytest

from nvision.cache.generations import (
    delete_generation,
    list_generations,
    prune_generations,
    restore_generation,
    save_generation,
)


def _make_fake_cache(out: Path, content: str) -> Path:
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "nv_center.db").write_text(content, encoding="utf-8")
    return cache_dir


def test_save_generation_copies_without_touching_source(tmp_path: Path):
    _make_fake_cache(tmp_path, "gen-a")

    dest = save_generation(tmp_path, label="gen-a")

    assert (dest / "cache" / "nv_center.db").read_text(encoding="utf-8") == "gen-a"
    assert (tmp_path / "cache" / "nv_center.db").read_text(encoding="utf-8") == "gen-a"
    assert (dest / "generation.json").exists()


def test_save_generation_missing_cache_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        save_generation(tmp_path, label="whatever")


def test_list_generations_most_recent_first(tmp_path: Path):
    _make_fake_cache(tmp_path, "v1")
    save_generation(tmp_path, label="first")
    (tmp_path / "cache" / "nv_center.db").write_text("v2", encoding="utf-8")
    save_generation(tmp_path, label="second")

    infos = list_generations(tmp_path)

    assert [i.label for i in infos] == ["second", "first"]


def test_list_generations_empty_when_no_generations_dir(tmp_path: Path):
    assert list_generations(tmp_path) == []


def test_restore_generation_replaces_cache_and_archives_current(tmp_path: Path):
    _make_fake_cache(tmp_path, "old-code")
    save_generation(tmp_path, label="old-code-gen")
    (tmp_path / "cache" / "nv_center.db").write_text("new-code", encoding="utf-8")

    restore_generation(tmp_path, "old-code-gen")

    assert (tmp_path / "cache" / "nv_center.db").read_text(encoding="utf-8") == "old-code"
    labels = {i.label for i in list_generations(tmp_path)}
    assert "pre-restore" in labels


def test_restore_generation_by_label_ambiguous_raises(tmp_path: Path):
    _make_fake_cache(tmp_path, "v1")
    save_generation(tmp_path, label="dup")
    (tmp_path / "cache" / "nv_center.db").write_text("v2", encoding="utf-8")
    save_generation(tmp_path, label="dup")

    with pytest.raises(ValueError, match="matches multiple generations"):
        restore_generation(tmp_path, "dup")


def test_restore_generation_unknown_raises(tmp_path: Path):
    _make_fake_cache(tmp_path, "v1")
    with pytest.raises(FileNotFoundError):
        restore_generation(tmp_path, "does-not-exist")


def test_prune_generations_keeps_only_n_most_recent(tmp_path: Path):
    _make_fake_cache(tmp_path, "v1")
    for i in range(4):
        (tmp_path / "cache" / "nv_center.db").write_text(f"v{i}", encoding="utf-8")
        save_generation(tmp_path, label=f"gen{i}")

    removed = prune_generations(tmp_path, keep=2)

    remaining = {i.label for i in list_generations(tmp_path)}
    assert remaining == {"gen3", "gen2"}
    assert set(removed) == {i for i in removed}  # names returned
    assert len(removed) == 2


def test_prune_generations_dry_run_does_not_delete(tmp_path: Path):
    _make_fake_cache(tmp_path, "v1")
    for i in range(3):
        (tmp_path / "cache" / "nv_center.db").write_text(f"v{i}", encoding="utf-8")
        save_generation(tmp_path, label=f"gen{i}")

    removed = prune_generations(tmp_path, keep=1, dry_run=True)

    assert len(removed) == 2
    assert len(list_generations(tmp_path)) == 3


def test_delete_generation_removes_single_entry(tmp_path: Path):
    _make_fake_cache(tmp_path, "v1")
    save_generation(tmp_path, label="to-delete")

    deleted_name = delete_generation(tmp_path, "to-delete")

    assert list_generations(tmp_path) == []
    assert deleted_name.endswith("to-delete")
