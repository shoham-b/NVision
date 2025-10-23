from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .obs import Obs


class LocatorStrategy(Protocol):
    def propose_next(self, history: Sequence[Obs], domain: tuple[float, float]) -> float: ...

    def should_stop(self, history: Sequence[Obs]) -> bool: ...

    def finalize(self, history: Sequence[Obs]) -> dict[str, float]: ...
