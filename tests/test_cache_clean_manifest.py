import json
from pathlib import Path

from typer.testing import CliRunner

from nvision.cli.cache_cmd import cache_app
from nvision.spectra.nv_center import PHYSICS_CONFIG_FINGERPRINT

runner = CliRunner()


def _write_manifest(out: Path, plots: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "plots_manifest.json", "w") as f:
        json.dump(plots, f)


def test_dynamically_named_grid_generator_is_kept(tmp_path: Path):
    """A grid-swept generator name like "NVCenter-voigt-w5.00MHz-c0.40-si0.00MHz" is not a
    literal member of the static GeneratorName enum, but must not be treated as invalid --
    it's the normal naming scheme for parameter-grid runs, not a defunct generator."""
    out = tmp_path / "artifacts"
    _write_manifest(
        out,
        [
            {
                "generator": "NVCenter-voigt-w5.00MHz-c0.40-si0.00MHz",
                "type": "scan",
                "true_params": {"config_fingerprint": PHYSICS_CONFIG_FINGERPRINT},
            },
            {
                "generator": "NVCenter-lorentzian",
                "type": "scan",
                "true_params": {"config_fingerprint": PHYSICS_CONFIG_FINGERPRINT},
            },
        ],
    )

    result = runner.invoke(cache_app, ["clean-manifest", "--out", str(out), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "No invalid entries found" in result.output


def test_defunct_generator_is_removed(tmp_path: Path):
    """A generator prefix with no relation to any current GeneratorName member (e.g. the
    retired "TwoPeak-*" family) should still be flagged and removed."""
    out = tmp_path / "artifacts"
    _write_manifest(
        out,
        [
            {
                "generator": "TwoPeak-w1.00MHz",
                "type": "scan",
                "true_params": {"config_fingerprint": PHYSICS_CONFIG_FINGERPRINT},
            },
        ],
    )

    result = runner.invoke(cache_app, ["clean-manifest", "--out", str(out), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Would remove 1 manifest entries" in result.output


def test_stale_physics_scan_entry_is_removed(tmp_path: Path):
    """A scan entry whose config_fingerprint doesn't match the current physics config is
    stale even though its generator name is a valid, currently-used grid name."""
    out = tmp_path / "artifacts"
    _write_manifest(
        out,
        [
            {
                "generator": "NVCenter-voigt-w5.00MHz-c0.40-si0.00MHz",
                "type": "scan",
                "true_params": {"config_fingerprint": "stale-fingerprint"},
            },
        ],
    )

    result = runner.invoke(cache_app, ["clean-manifest", "--out", str(out), "--dry-run", "--stale-physics"])

    assert result.exit_code == 0, result.output
    assert "Would remove 1 manifest entries" in result.output
