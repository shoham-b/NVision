from __future__ import annotations

from typing import Any

from nvision.models.noise import CompositeNoise
from nvision.sim import (
    cases as sim_cases,
)
from nvision.sim.locs.sweep_locator import SimpleSweepLocator


def _noise_presets() -> list[tuple[str, CompositeNoise | None]]:
    """Return the predefined noise combinations for scenarios."""
    return sim_cases.noises_none() + sim_cases.noises_single_each() + sim_cases.noises_complex()


def _locator_strategies_for_generator(generator_name: str) -> list[tuple[str, Any]]:
    """Get the appropriate locator strategies for a given generator category."""
    return [
        ("SimpleSweep", SimpleSweepLocator),
    ]


def _get_generator_category(generator_name: str) -> str:
    """Determine the category of a generator from its name."""
    if generator_name.startswith("OnePeak-"):
        return "OnePeak"
    elif generator_name.startswith("TwoPeak-"):
        return "TwoPeak"
    elif generator_name.startswith("NVCenter-"):
        return "NVCenter"
    return "Unknown"
