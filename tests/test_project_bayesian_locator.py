import numpy as np
import pytest
from nvision.sim.locs.nv_center.project_bayesian_locator import ProjectBayesianLocator
from nvision.sim.locs.base import ScanBatch


def build_locator(**overrides):
    config = {
        "max_evals": 10,
        "prior_bounds": (2.7e9, 3.0e9),
        "grid_resolution": 64,
        "n_monte_carlo": 10,
        "n_warmup": 2,
    }
    config.update(overrides)
    return ProjectBayesianLocator(**config)


def build_scan():
    return ScanBatch(
        x_min=2.7e9,
        x_max=3.0e9,
        signal=lambda x: 1.0,
        meta={"inference": {"peaks": []}},
        truth_positions=[2.85e9],
    )


def test_project_locator_initialization():
    locator = build_locator()
    assert isinstance(locator, ProjectBayesianLocator)


def test_project_locator_propose_next():
    locator = build_locator()
    scan = build_scan()
    # This should trigger _optimize_acquisition and fail due to typo
    proposal = locator.propose_next([], scan)
    assert isinstance(proposal, float)
