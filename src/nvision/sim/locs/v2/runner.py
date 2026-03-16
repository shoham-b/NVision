"""Runner for stateless locator architecture."""

import random

import pandas as pd

from nvision.sim.locs.v2.base import Locator
from nvision.sim.locs.v2.experiment import Experiment


class Runner:
    """Generic runner for stateless locators.

    Owns the measurement loop with no locator-specific logic.
    The same locator instance is safely reused across all repeats.
    """

    def run(self, locator: Locator, experiment: Experiment, repeats: int, rng: random.Random) -> list[dict[str, float]]:
        """Run the locator for multiple repeats.

        Parameters
        ----------
        locator : Locator
            Stateless locator instance (reused across repeats)
        experiment : Experiment
            Experimental setup for generating measurements
        repeats : int
            Number of independent repeats to run
        rng : random.Random
            Random number generator for reproducible results

        Returns
        -------
        list[dict[str, float]]
            List of result dictionaries, one per repeat
        """
        results = []

        for _ in range(repeats):
            # Fresh history for each repeat - use list for efficiency
            measurements = []

            while True:
                # Convert to DataFrame for locator
                history = (
                    pd.DataFrame(measurements, columns=["x", "signal_value"])
                    if measurements
                    else pd.DataFrame(columns=["x", "signal_value"])
                )

                if locator.done(history):
                    break

                x = locator.next(history)
                y = experiment.measure(x, rng)
                measurements.append({"x": x, "signal_value": y})

            # Final history for result extraction
            history = pd.DataFrame(measurements, columns=["x", "signal_value"])
            results.append(locator.result(history))

        return results
