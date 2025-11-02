from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from nvision.sim.locs.base import ScanBatch
from nvision.sim.locs.models.obs import Obs


class LocatorStrategy(Protocol):
    def set_scan(self, scan: ScanBatch) -> None:
        """Provide the scan context to the strategy."""
        ...

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float: ...

    def should_stop(self, history: Sequence[Obs]) -> bool: ...

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]: ...


class ObservationModel(Protocol):
    def get_uncertainty(self, posterior: list[float]) -> list[float]:
        """Calculates the uncertainty based on the posterior distribution."""
        ...

    def update_posterior(
        self,
        posterior: list[float],
        x_measured: float,
        y_measured: float,
        min_x: float,
        max_x: float,
        num_x_bins: int,
    ) -> list[float]:
        """Updates the posterior distribution given a new measurement."""
        ...
