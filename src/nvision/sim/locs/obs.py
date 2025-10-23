from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Obs:
    x: float
    intensity: float
    uncertainty: float = 0.0
