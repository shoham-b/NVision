from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from nvision.sim.locs.models.obs import Obs
from nvision.sim.scan_batch import ScanBatch


class LocatorStrategy(Protocol):
    def set_scan(self, scan: ScanBatch) -> None:
        """Provide the scan context to the strategy."""
        ...

    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float: ...

    def should_stop(self, history: Sequence[Obs]) -> bool: ...

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]: ...
