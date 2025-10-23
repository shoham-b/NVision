from __future__ import annotations

import math
import random

from nvision.sim.gen import OnePeakGenerator, RabiManufacturer, T1DecayManufacturer


def _argmax_x_in_grid(f, x_min: float, x_max: float, n: int = 1001) -> float:
    step = (x_max - x_min) / (n - 1)
    best_x = x_min
    best_y = -float("inf")
    for i in range(n):
        x = x_min + i * step
        y = float(f(x))
        if y > best_y:
            best_y = y
            best_x = x
    return best_x


def test_onepeak_rabi_mode_has_peak_near_truth():
    rng = random.Random(123)
    sigma = 0.06
    gen = OnePeakGenerator(
        manufacturer=RabiManufacturer(amplitude=1.0, sigma=sigma, rabi_freq=7.0),
        base=0.0,
    )
    scan = gen.generate(rng)
    # crude grid search for maximum
    x_hat = _argmax_x_in_grid(scan.signal, scan.x_min, scan.x_max, n=2001)
    x0 = scan.truth_positions[0]
    # expect the detected max to be near the hidden x0 within a few sigmas
    assert math.isfinite(x_hat)
    assert abs(x_hat - x0) <= 3.0 * sigma + 1e-3


def test_onepeak_t1_decay_mode_has_peak_at_truth():
    rng = random.Random(321)
    gen = OnePeakGenerator(
        manufacturer=T1DecayManufacturer(amplitude=1.0, t1_tau=0.08),
        base=0.0,
    )
    scan = gen.generate(rng)
    x_hat = _argmax_x_in_grid(scan.signal, scan.x_min, scan.x_max, n=2001)
    x0 = scan.truth_positions[0]
    assert math.isfinite(x_hat)
    # t1_decay is sharply peaked at x0; allow small grid resolution tolerance
    assert abs(x_hat - x0) <= (scan.x_max - scan.x_min) / 2000 + 1e-9
