"""Drive link parsing and the folder-tree processing the Drive fetch feeds."""

import pytest

from nvision.tools.drive_fetch import fetch_drive_folder, parse_drive_folder_id
from tests.test_lab_process import _save_run

_ID = "1a2AQl7IQQw78g1w5r14Y6g_quJcl_byB"


@pytest.mark.parametrize(
    "text",
    [
        f"https://drive.google.com/drive/folders/{_ID}",
        f"https://drive.google.com/drive/folders/{_ID}?usp=sharing",
        f"https://drive.google.com/drive/u/0/folders/{_ID}",
        _ID,
    ],
)
def test_parse_drive_folder_id(text):
    assert parse_drive_folder_id(text) == _ID


def test_parse_rejects_non_folder_link():
    with pytest.raises(ValueError, match="Not a Google Drive folder"):
        parse_drive_folder_id("https://example.com/file.mat")


def test_fetch_fails_fast_without_rclone(tmp_path, monkeypatch):
    monkeypatch.setattr("nvision.tools.drive_fetch.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="rclone is not on PATH"):
        fetch_drive_folder(_ID, tmp_path)


def test_matlab_command_fetches_a_drive_link_and_serve_finds_the_lab(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import nvision.__main__  # noqa: F401  (registers every command; a lone command would become the root)
    from nvision.cli.main import app
    from nvision.metrics.lab_study import discover_labs, labs_dir

    def fake_fetch(url, dest, remote="gdrive"):
        (dest / "runs").mkdir(parents=True)
        _save_run(dest / "runs" / "a.mat", no_stop=1, std=[2.0, 0.5], stop_reason="budget", step_count=2)
        return dest

    monkeypatch.setattr("nvision.tools.lab_sources.fetch_drive_folder", fake_fetch)
    result = CliRunner().invoke(
        app, ["matlab", f"https://drive.google.com/drive/folders/{_ID}", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert list(discover_labs([labs_dir(tmp_path)])) == ["runs"]
    # Remembered, so `nv matlab` with no source re-syncs it without the link.
    from nvision.tools.lab_sources import load_sources

    assert [(e["kind"], e["source"]) for e in load_sources(tmp_path)] == [("drive", _ID)]


def test_matlab_command_loads_a_local_folder_and_skips_sweeps_on_request(tmp_path):
    import scipy.io as sio
    from typer.testing import CliRunner

    import nvision.__main__  # noqa: F401  (registers every command; a lone command would become the root)
    from nvision.cli.main import app

    src = tmp_path / "src"
    src.mkdir()
    _save_run(src / "a.mat", no_stop=1, std=[2.0, 0.5], stop_reason="budget", step_count=2)
    sio.savemat(src / "sweep.mat", {"myStruct": {"ESR": {"x": 1}}})
    result = CliRunner().invoke(app, ["matlab", str(src), "--dir", str(tmp_path / "art"), "--no-sweeps"])
    assert result.exit_code == 0, result.output
    assert "Skipped 1 raw sweep file(s)" in result.output
    assert (tmp_path / "art" / "labs" / "src" / "convergence_statistics").is_dir()


def test_fetch_mirrors_drive_so_deleted_files_disappear(monkeypatch, tmp_path):
    """`rclone copy` kept files removed on Drive (a deleted bad run kept feeding the statistics); `sync` mirrors."""
    import subprocess

    from nvision.tools import drive_fetch

    calls: list[list[str]] = []
    monkeypatch.setattr(drive_fetch.shutil, "which", lambda _: "rclone")
    monkeypatch.setattr(
        drive_fetch.subprocess, "run", lambda cmd, check: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0)
    )
    drive_fetch.fetch_drive_folder("a" * 33, tmp_path / "dest")
    assert calls[0][1] == "sync"
