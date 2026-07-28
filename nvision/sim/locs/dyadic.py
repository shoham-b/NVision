"""Exact dyadic-fraction helpers shared by SimpleSweep and SimpleSobol.

Both locators sample from the same nested dyadic grid: :class:`GenericSweepLocator`
scans a uniform ``2**k + 1``-point grid over ``[0, 1]``, and
:class:`SimpleSobolBayesianLocator`'s base-2 van der Corput sequence visits the
*interior* of that same grid, just in bit-reversed order (this is the classical
nested-refinement property of the base-2 van der Corput sequence: for
``n = 1, ..., 2**k - 1`` its values are exactly ``{i / 2**k : i = 1, ..., 2**k - 1}``).
That means whichever of the two locators runs first for a given experiment can
hand its noisy measurements to the other one point-for-point, instead of each
drawing its own fresh noise at the same physical positions.

Fractions (not floats) are the cache key so lookups are exact -- no floating-point
comparison between two different computation paths (``np.linspace`` vs bit
manipulation).
"""

from __future__ import annotations

import math
from fractions import Fraction


def van_der_corput_fraction(n: int, base: int = 2) -> Fraction:
    """Exact rational value of the base-``base`` van der Corput sequence at index ``n``.

    Mirrors :func:`~nvision.sim.locs.bayesian.sobol_bayesian_locator.van_der_corput`'s
    float algorithm digit-for-digit, accumulating in :class:`Fraction` instead so the
    result is exact.
    """
    vdc = Fraction(0)
    denom = 1
    val = n
    while val > 0:
        denom *= base
        val, remainder = divmod(val, base)
        vdc += Fraction(remainder, denom)
    return vdc


def round_to_dyadic_points(n: int) -> int:
    """Round a requested point count to the nearest ``2**k + 1``.

    ``2**k + 1`` points (endpoints included) span ``2**k`` equal intervals, so the
    ``2**k - 1`` interior points are exactly the base-2 van der Corput sequence's
    first ``2**k - 1`` values (see module docstring).
    """
    if n <= 2:
        return 2
    k = max(round(math.log2(n - 1)), 1)
    return 2**k + 1
