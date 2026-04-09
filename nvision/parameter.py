from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Parameter:
    """Legacy numeric parameter with bounds and current value."""

    name: str
    bounds: tuple[float, float]
    value: float

    def __post_init__(self) -> None:
        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        if hi <= lo:
            raise ValueError(f"Invalid bounds for {self.name}: {(lo, hi)}")
        self.bounds = (lo, hi)
        self.value = float(self.value)
