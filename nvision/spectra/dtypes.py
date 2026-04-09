"""Shared numeric dtypes for NumPy/Numba-heavy paths."""

from __future__ import annotations

import numpy as np

# Use float32 for vectorized evaluation, belief arrays, and Numba kernels.
FLOAT_DTYPE = np.float32
