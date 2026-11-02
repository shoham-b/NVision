"""Drive link parsing and the folder-tree processing the Drive fetch feeds."""

import pytest

from nvision.metrics.lab_process import process_lab_tree
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


def test_process_lab_tree_reports_each_folder(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "other").mkdir()
    _save_run(tmp_path / "good" / "a.mat", no_stop=1, std=[2.0, 0.5], stop_reason="budget", step_count=2)
    _save_run(tmp_path / "other" / "b.mat", no_stop=1, std=[2.0, 0.5], stop_reason="budget", step_count=2)
    import scipy.io as sio

    sio.savemat(tmp_path / "other" / "b.mat", {"myStruct": {"ESR": {"x": 1}}})  # a raw sweep, not a Bayesian run
    outcomes = process_lab_tree(tmp_path)
    assert outcomes[tmp_path / "good"] == "processed"
    assert outcomes[tmp_path / "other"].startswith("skipped")


def test_lab_add_stores_under_labs_and_serve_finds_it(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from nvision.cli.lab_cmd import app as _  # noqa: F401  (registers the command)
    from nvision.cli.main import app
    from nvision.metrics.lab_study import discover_labs, labs_dir

    def fake_fetch(url, dest, remote="gdrive"):
        (dest / "runs").mkdir(parents=True)
        _save_run(dest / "runs" / "a.mat", no_stop=1, std=[2.0, 0.5], stop_reason="budget", step_count=2)
        return dest

    monkeypatch.setattr("nvision.tools.drive_fetch.fetch_drive_folder", fake_fetch)
    result = CliRunner().invoke(app, ["lab-add", _ID, "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert list(discover_labs([labs_dir(tmp_path)])) == ["runs"]
