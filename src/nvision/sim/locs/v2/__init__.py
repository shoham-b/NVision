"""Stateless locator architecture v2.

This module provides a clean, stateless interface for locators where:
- Locators are fully stateless and reusable across repeats
- History is the only state, passed as a Polars DataFrame
- Runner owns the loop with no locator-specific logic
"""

from nvision.sim.locs.v2.base import Locator, Observation
from nvision.sim.locs.v2.experiment import Experiment
from nvision.sim.locs.v2.runner import Runner
from nvision.sim.locs.v2.simple import GridMaxLocator

__all__ = ["Experiment", "GridMaxLocator", "Locator", "Observation", "Runner"]
