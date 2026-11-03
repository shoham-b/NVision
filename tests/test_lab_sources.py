"""Remembered lab sources and their incremental sync (nvision.tools.lab_sources)."""

from __future__ import annotations

import json
import os

import pytest

from nvision.metrics.lab_reference import lab_sweeps_payload
from nvision.tools import lab_sources
from nvision.tools.lab_sources import (
    PROCESSING_VERSION,
    FileIndex,
    SyncLock,
    add_source,
    load_sources,
    load_sweep_records,
    sweep_record_is_current,
)


def test_add_source_remembers_once_and_suffixes_clashing_labels(tmp_path):
    a = tmp_path / "x" / "data"
    b = tmp_path / "y" / "data"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    first = add_source(tmp_path / "art", str(a))
    assert add_source(tmp_path / "art", str(a)) == first
    second = add_source(tmp_path / "art", str(b))
    assert (first["label"], second["label"]) == ("data", "data-2")
    assert [e["label"] for e in load_sources(tmp_path / "art")] == ["data", "data-2"]


def test_add_source_rejects_a_missing_path(tmp_path):
    with pytest.raises(ValueError, match="neither a Drive folder link nor an existing path"):
        add_source(tmp_path, str(tmp_path / "nope"))


def test_file_index_rereads_only_new_or_changed_files(tmp_path, monkeypatch):
    src = tmp_path / "src" / "runs"
    src.mkdir(parents=True)
    a, b = src / "a.mat", src / "b.mat"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    calls = []
    monkeypatch.setattr(lab_sources, "classify_mat", lambda p: calls.append(p.name) or "bayesian")

    index = FileIndex(tmp_path / "labs")
    index.classify([a, b], tmp_path / "src")
    index.save()
    assert sorted(calls) == ["a.mat", "b.mat"]

    calls.clear()
    index = FileIndex(tmp_path / "labs")
    index.classify([a, b], tmp_path / "src")
    assert calls == []
    assert not index.folder_changed([a, b], tmp_path / "src")

    os.utime(b, (1, 1))
    index = FileIndex(tmp_path / "labs")
    index.classify([a, b], tmp_path / "src")
    assert calls == ["b.mat"]
    assert index.folder_changed([a, b], tmp_path / "src")


def test_file_index_notices_a_removed_file(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    a, b = src / "a.mat", src / "b.mat"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    monkeypatch.setattr(lab_sources, "classify_mat", lambda p: "bayesian")
    index = FileIndex(tmp_path / "labs")
    index.classify([a, b], src)
    index.save()
    b.unlink()
    index = FileIndex(tmp_path / "labs")
    index.classify([a], src)
    assert index.folder_changed([a], src)


def test_sweep_record_is_current_until_the_file_changes_or_it_failed(tmp_path):
    mat = tmp_path / "s.mat"
    mat.write_bytes(b"x")
    st = mat.stat()
    rec = tmp_path / "s.json"
    base = {"version": PROCESSING_VERSION, "source_size": st.st_size, "source_mtime": st.st_mtime}
    rec.write_text(json.dumps(base))
    assert sweep_record_is_current(rec, mat)
    rec.write_text(json.dumps({**base, "error": "boom"}))
    assert not sweep_record_is_current(rec, mat)
    rec.write_text(json.dumps({**base, "version": PROCESSING_VERSION - 1}))
    assert not sweep_record_is_current(rec, mat)
    rec.write_text(json.dumps(base))
    mat.write_bytes(b"xy")
    assert not sweep_record_is_current(rec, mat)


def test_sync_lock_is_exclusive(tmp_path):
    with SyncLock(tmp_path) as first:
        assert first.acquired
        with SyncLock(tmp_path) as second:
            assert not second.acquired
    with SyncLock(tmp_path) as third:
        assert third.acquired


def _run(z, est=1.0e6):
    return {
        "n_valid_shots": 75,
        "drive_freq_min_mhz": 2860.0,
        "drive_freq_max_mhz": 2880.0,
        "total_steps": 300,
        "primary_converged_step": None,
        "estimates": {"zeeman_shift": est},
        "uncertainties": {"zeeman_shift": 1e5},
        "shot_reuse": 1.2,
        "holdout": {
            "ref_zeeman_shift": est,
            "ref_zeeman_shift_sigma": 2e4,
            "ref_chi2_dof": 1.1,
            "holdout_z_zeeman_shift": z,
            "holdout_z_zero_field_freq": 0.1,
        },
    }


def test_sweeps_payload_groups_by_folder_and_lists_failures(tmp_path):
    art = tmp_path / "art"
    (tmp_path / "data").mkdir()
    entry = add_source(art, str(tmp_path / "data"))
    sweeps = art / "labs" / entry["label"] / "sweeps"
    for folder, name, z in (("a", "1", 0.5), ("a", "2", 2.5), ("b", "3", -0.2)):
        (sweeps / folder).mkdir(parents=True, exist_ok=True)
        rec = {"file": f"{folder}/{name}.mat", "folder": folder, "full": _run(z), "holdout": _run(z)}
        (sweeps / folder / f"{name}.json").write_text(json.dumps(rec))
    (sweeps / "b" / "bad.json").write_text(json.dumps({"file": "b/bad.mat", "folder": "b", "error": "ValueError: x"}))

    payload = lab_sweeps_payload(load_sweep_records(art))
    groups = {g["folder"]: g for g in payload["groups"]}
    assert set(groups) == {"a", "b"}
    assert groups["a"]["coverage"]["z_zeeman"] == {"n": 2, "within_1": 0.5, "within_2": 0.5, "median_abs": 1.5}
    assert groups["b"]["coverage"]["z_zeeman"]["n"] == 1
    assert groups["a"]["rows"][0]["zeeman_mhz"] == pytest.approx(1.0)
    assert payload["failed"] == [{"label": "data", "file": "b/bad.mat", "error": "ValueError: x"}]
