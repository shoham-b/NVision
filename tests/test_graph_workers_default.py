"""graph_workers_for must scale with the runner count (half, min 1) unless overridden,
so a single graph worker doesn't fall permanently behind several simulation runners --
see the "why the graph_queue spool is 25GB but the archive is only 2.6GB" case."""

import importlib

import nvision.cli.defaults as cli_defaults


def test_scales_to_half_the_runners(monkeypatch):
    monkeypatch.delenv("NVISION_GRAPH_WORKERS", raising=False)
    importlib.reload(cli_defaults)
    assert cli_defaults.graph_workers_for(8) == 4
    assert cli_defaults.graph_workers_for(1) == 1
    assert cli_defaults.graph_workers_for(3) == 1


def test_env_override_wins_including_zero(monkeypatch):
    monkeypatch.setenv("NVISION_GRAPH_WORKERS", "0")
    importlib.reload(cli_defaults)
    assert cli_defaults.graph_workers_for(8) == 0

    monkeypatch.setenv("NVISION_GRAPH_WORKERS", "5")
    importlib.reload(cli_defaults)
    assert cli_defaults.graph_workers_for(8) == 5

    monkeypatch.delenv("NVISION_GRAPH_WORKERS", raising=False)
    importlib.reload(cli_defaults)
