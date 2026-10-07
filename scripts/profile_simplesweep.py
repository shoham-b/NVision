"""Profile SimpleSweep to see whether the scalar loop overhead matters.

Run with:
    uv run python scripts/profile_simplesweep.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import random
import time

import numpy as np

from nvision import CoreExperiment, GenericSweepLocator, NVCenterCoreGenerator, run_loop

# ── Experiment setup ─────────────────────────────────────────────────────────


def make_experiment(rng: random.Random) -> CoreExperiment:
    gen = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    drive_freq_min_phys, drive_freq_max_phys = None, None
    for name in true_signal.parameter_names:
        if "center_freq" in name:
            drive_freq_min_phys, drive_freq_max_phys = true_signal.get_param_bounds(name)
            break
    assert drive_freq_min_phys is not None
    return CoreExperiment(
        true_signal=true_signal,
        noise=None,
        drive_freq_min_phys=drive_freq_min_phys,
        drive_freq_max_phys=drive_freq_max_phys,
    )


def run_sweep(exp: CoreExperiment, rng: random.Random, n_steps: int) -> None:
    list(run_loop(GenericSweepLocator, exp, rng, max_steps=n_steps))


# ── 1. cProfile the full loop ─────────────────────────────────────────────────

rng_setup = random.Random(42)
exp = make_experiment(rng_setup)
N = 500  # realistic step count

print(f"Profiling SimpleSweep with {N} steps …\n")

pr = cProfile.Profile()
pr.enable()
run_sweep(exp, random.Random(42), N)
pr.disable()

stream = io.StringIO()
ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
ps.print_stats(25)
print(stream.getvalue())


# ── 2. Isolate: scalar loop vs. batch signal eval ────────────────────────────

exp2 = make_experiment(random.Random(99))
xs_norm = np.linspace(0.0, 1.0, N)
drive_freq_min_phys, drive_freq_max_phys = exp2.drive_freq_min_phys, exp2.drive_freq_max_phys
xs_phys = drive_freq_min_phys + xs_norm * (drive_freq_max_phys - drive_freq_min_phys)

# Time the scalar measure loop (current behavior)
t0 = time.perf_counter()
rng_scalar = random.Random(1)
for x in xs_norm:
    exp2.measure(float(x), rng_scalar)
t_scalar = time.perf_counter() - t0

# Time vectorized signal evaluation only (no noise, no Observation overhead)
t0 = time.perf_counter()
ys_batch = np.array([exp2.true_signal(float(x)) for x in xs_phys])
t_signal_loop = time.perf_counter() - t0

# Time with numpy vectorization if signal supports it
try:
    t0 = time.perf_counter()
    ys_vec = exp2.true_signal(xs_phys)  # may or may not work
    t_signal_vec = time.perf_counter() - t0
    vec_supported = True
except Exception:
    t_signal_vec = None
    vec_supported = False

print("=" * 60)
print(f"  N = {N} steps")
print(f"  Scalar measure() loop        : {t_scalar * 1000:.2f} ms")
print(f"  Signal eval loop (no noise)  : {t_signal_loop * 1000:.2f} ms")
if vec_supported:
    print(f"  Vectorized signal eval       : {t_signal_vec * 1000:.2f} ms")
else:
    print("  Vectorized signal eval       : not supported by this model")

overhead_ms = (t_scalar - t_signal_loop) * 1000
print(
    f"  Loop/noise/obs overhead      : {overhead_ms:.2f} ms  "
    f"({overhead_ms / max(t_scalar * 1000, 1e-9) * 100:.0f}% of total)"
)
print("=" * 60)
