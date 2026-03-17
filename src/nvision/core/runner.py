"""Generic runner for localization experiments."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Type

from nvision.core.experiment import CoreExperiment
from nvision.core.locator import Locator


class Runner:
    """Generic runner for Bayesian localization.

    Owns the measurement loop with no locator-specific logic.
    Implemented as a generator so caller controls what happens at each step.
    """

    def run(
        self,
        locator_class: Type[Locator],
        experiment: CoreExperiment,
        rng: random.Random,
        **locator_config,
    ) -> Iterator[Locator]:
        """Run locator for one repeat as generator.

        Yields locator state after each observation so caller can inspect
        belief evolution. Caller controls iteration and can break early.

        Parameters
        ----------
        locator_class : Type[Locator]
            Locator class with `create()` classmethod
        experiment : CoreExperiment
            Experiment setup with true signal and noise
        rng : random.Random
            Random number generator for reproducible noise
        **locator_config
            Configuration passed to locator_class.create()

        Yields
        ------
        Locator
            Locator state after each observation update

        Examples
        --------
        >>> runner = Runner()
        >>> for locator in runner.run(SimpleSweepLocator, experiment, rng, max_steps=50):
        ...     print(f"Entropy = {locator.belief.entropy()}")
        ...     if locator.done():
        ...         break
        """
        locator = locator_class.create(**locator_config)

        while not locator.done():
            # Locator proposes next measurement (normalized [0,1])
            x_normalized = locator.next()

            # Measure signal with noise
            obs = experiment.measure(x_normalized, rng)

            # Update locator belief
            locator.observe(obs)

            # Yield for caller to inspect state
            yield locator
