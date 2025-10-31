from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from nvision.sim.locs.models.obs import Obs
from nvision.sim.scan_batch import ScanBatch


@dataclass
class GoldenSectionSearchLocator:
    """A locator that uses the golden-section search algorithm to find a maximum.

    Golden-section search is an algorithm for finding the extremum (minimum or
    maximum) of a unimodal function by successively narrowing the range of values
    inside which the extremum is known to exist.

    This implementation is adapted for a noisy function by:
    1.  Terminating after a fixed number of evaluations (`max_evals`) rather
        than a precision threshold, which is more robust to noise.
    2.  Using the observation with the highest intensity found so far as the
        final result, rather than the midpoint of the final interval.
    3.  Optionally sampling each point multiple times (`samples_per_point`) and
        averaging the results to mitigate the effects of measurement noise.

    It is suitable for finding a single peak in a noisy environment.
    """

    max_evals: int = 25
    samples_per_point: int = 3  # Take multiple samples to average out noise
    _golden_ratio: float = (5**0.5 - 1) / 2  # approx 0.618

    # Internal state for the search algorithm
    _lower_bound: float | None = field(default=None, init=False, repr=False)
    _upper_bound: float | None = field(default=None, init=False, repr=False)
    _inner_point_c: float | None = field(default=None, init=False, repr=False)
    _inner_point_d: float | None = field(default=None, init=False, repr=False)

    _scan: ScanBatch | None = field(default=None, init=False, repr=False)

    def set_scan(self, scan: ScanBatch) -> None:
        """Provide the domain for the search and reset internal state."""
        self._scan = scan
        self._lower_bound = None
        self._upper_bound = None
        self._inner_point_c = None
        self._inner_point_d = None

    def _get_averaged_history(self, history: Sequence[Obs]) -> dict[float, float]:
        """Averages intensities for points that were sampled multiple times."""
        point_intensities = defaultdict(list)
        for obs in history:
            point_intensities[obs.x].append(obs.intensity)

        averaged_points = {
            x: sum(intensities) / len(intensities) for x, intensities in point_intensities.items()
        }
        return averaged_points

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float:
        """Proposes the next point to sample using the golden-section search logic."""
        # Check if we need to re-sample a point for averaging
        point_counts = defaultdict(int)
        for obs in history:
            point_counts[obs.x] += 1

        if self._inner_point_c and point_counts[self._inner_point_c] < self.samples_per_point:
            return self._inner_point_c
        if self._inner_point_d and point_counts[self._inner_point_d] < self.samples_per_point:
            return self._inner_point_d

        # Get the current state of the search from averaged history
        averaged_history = self._get_averaged_history(history)

        # Initialize search on the first call
        if not history:
            self._lower_bound, self._upper_bound = domain
            self._inner_point_c = self._upper_bound - self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            self._inner_point_d = None  # d is not yet defined
            return self._inner_point_c

        # After the first point (c) is sampled, propose the second (d)
        if len(averaged_history) == 1:
            self._lower_bound, self._upper_bound = domain
            # Re-establish c from history
            self._inner_point_c = next(iter(averaged_history.keys()))
            self._inner_point_d = self._lower_bound + self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            return self._inner_point_d

        # Main golden-section search iteration
        f_at_c = averaged_history.get(self._inner_point_c)
        f_at_d = averaged_history.get(self._inner_point_d)

        if f_at_c is None or f_at_d is None:
            # Should not happen if sampling logic is correct, but as a fallback
            return self._inner_point_c if f_at_c is None else self._inner_point_d

        if f_at_c > f_at_d:
            # The maximum is in the interval [a, d]
            self._upper_bound = self._inner_point_d
            self._inner_point_d = self._inner_point_c
            self._inner_point_c = self._upper_bound - self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            return self._inner_point_c
        else:
            # The maximum is in the interval [c, b]
            self._lower_bound = self._inner_point_c
            self._inner_point_c = self._inner_point_d
            self._inner_point_d = self._lower_bound + self._golden_ratio * (
                self._upper_bound - self._lower_bound
            )
            return self._inner_point_d

    def should_stop(self, history: Sequence[Obs]) -> bool:
        """Stops after a fixed number of evaluations."""
        return len(history) >= self.max_evals

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]:
        """Returns the point with the highest observed intensity."""
        if not history:
            return {"n_peaks": 0.0, "x1": 0.0, "uncert": float("inf")}

        best_obs = max(history, key=lambda o: o.intensity)
        return {"n_peaks": 1.0, "x1": best_obs.x, "uncert": best_obs.uncertainty}
