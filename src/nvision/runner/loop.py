"""Core measurement loop for localization experiments."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Type

from nvision.models.experiment import CoreExperiment
from nvision.models.locator import Locator


def run_loop(
    locator_class: Type[Locator],
    experiment: CoreExperiment,
    rng: random.Random,
    **locator_config,
) -> Iterator[Locator]:
    """Run one locator repeat as a generator.

    Yields the locator after each observation so the caller can inspect belief
    evolution at every step and break early if desired.

    Parameters
    ----------
    locator_class : Type[Locator]
        Locator subclass with a ``create()`` classmethod.
    experiment : CoreExperiment
        Experiment supplying noisy measurements.
    rng : random.Random
        Random number generator (seeded externally for reproducibility).
    **locator_config
        Keyword arguments forwarded to ``locator_class.create()``.

    Yields
    ------
    Locator
        Locator state after each observation.

    Examples
    --------
    >>> for locator in run_loop(SimpleSweepLocator, experiment, rng, max_steps=50):
    ...     if locator.done():
    ...         break
    """
    locator = locator_class.create(**locator_config)

    while not locator.done():
        x_normalized = locator.next()
        obs = experiment.measure(x_normalized, rng)
        locator.observe(obs)
        yield locator
