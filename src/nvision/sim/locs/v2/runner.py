"""Runner for stateless locator architecture."""

import random
import time

import polars as pl

from nvision.sim.locs.v2.base import Locator
from nvision.sim.locs.v2.experiment import Experiment


class Runner:
    """Generic runner for stateless locators.

    Owns the measurement loop with no locator-specific logic.
    The same locator instance is safely reused across all repeats.
    """

    def run(
        self,
        locator: Locator,
        experiment: Experiment,
        repeats: int,
        rng: random.Random,
        *,
        max_steps: int | None = None,
        timeout_s: float | None = None,
        return_history: bool = False,
    ) -> list[dict[str, float]]:
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
        results: list[dict[str, float]] = []
        histories: list[pl.DataFrame] = []
        stop_reasons: list[str] = []

        for _ in range(repeats):
            # Fresh history for each repeat - use list for efficiency
            measurements = []
            start_t = time.perf_counter()

            while True:
                # Convert to DataFrame for locator
                history = (
                    pl.DataFrame(measurements)
                    if measurements
                    else pl.DataFrame(schema={"x": pl.Float64, "signal_value": pl.Float64})
                )

                if locator.done(history):
                    stop_reasons.append("locator_stop")
                    break

                if max_steps is not None and history.height >= max_steps:
                    stop_reasons.append("max_steps_reached")
                    break

                if timeout_s is not None and (time.perf_counter() - start_t) >= timeout_s:
                    stop_reasons.append("repeat_timeout")
                    break

                x = locator.next(history)
                y = experiment.measure(x, rng, locator)
                measurements.append({"x": x, "signal_value": y})

            # Final history for result extraction
            history = pl.DataFrame(measurements)
            results.append(locator.result(history))
            if return_history:
                histories.append(history)
                if len(stop_reasons) < len(histories):
                    stop_reasons.append("locator_stop")

        if return_history:
            return results, histories, stop_reasons  # type: ignore[return-value]
        return results
