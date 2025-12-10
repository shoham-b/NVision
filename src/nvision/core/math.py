from __future__ import annotations

import math
from typing import Sequence


def normalized_sum(values: Sequence[float]) -> float:
    """Return the sum of values, normalized by the length of the sequence."""
    if not values:
        return 0.0
    return sum(values) / len(values)
