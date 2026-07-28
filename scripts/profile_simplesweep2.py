"""Warm-run timing comparison: belief-per-step vs. deferred belief update.

Run with:
    uv run python scripts/profile_simplesweep2.py
"""

from __future__ import annotations

import random
import time

import numpy as np

from nvision import CoreExperiment, GenericSweepLocator, NVCenterCoreGenerator, run_loop
from nvision.models.observation import Observation
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator


def make_experiment(seed: int) -> CoreExperiment:
    rng = random.Random(seed)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    x_min, x_max = None, None
    for name in true_signal.parameter_names:
        if "frequency" in name:
            x_min, x_max = true_signal.get_param_bounds(name)
            break
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)


N_STEPS = 500
N_REPEATS = 5


# ── Warm up Numba (one throwaway run) ────────────────────────────────────────
print("Warming up Numba JIT …", flush=True)
list(run_loop(GenericSweepLocator, make_experiment(0), random.Random(0), max_steps=20))
print("  done.\n")


# ── Current implementation (deferred belief update) ──────────────────────────
times_deferred = []
for i in range(N_REPEATS):
    exp = make_experiment(i + 1)
    rng = random.Random(i + 1)
    t0 = time.perf_counter()
    list(run_loop(GenericSweepLocator, exp, rng, max_steps=N_STEPS))
    times_deferred.append(time.perf_counter() - t0)

# ── Baseline: force per-step belief update (monkey-patch observe) ─────────────
_orig_observe = SweepingLocator.observe


def _eager_observe(self, obs: Observation) -> None:
    if self.step_count <= self.max_steps:
        self.history.append(obs)
        self.belief.last_obs = obs
        self.belief.update(obs)


# Temporarily swap in the eager observe for GenericSweepLocator
_gsl_observe_saved = GenericSweepLocator.observe
GenericSweepLocator.observe = _eager_observe

times_eager = []
for i in range(N_REPEATS):
    exp = make_experiment(i + 1)
    rng = random.Random(i + 1)
    t0 = time.perf_counter()
    list(run_loop(GenericSweepLocator, exp, rng, max_steps=N_STEPS))
    times_eager.append(time.perf_counter() - t0)

GenericSweepLocator.observe = _gsl_observe_saved  # restore


# ── Report ────────────────────────────────────────────────────────────────────
def stats(label, times):
    arr = np.array(times) * 1000  # ms
    print(f"  {label:<40s}  mean={arr.mean():.1f} ms  min={arr.min():.1f} ms  max={arr.max():.1f} ms")


print(f"Warm runs (N={N_STEPS} steps, {N_REPEATS} repeats each)")
print("=" * 75)
stats("Per-step belief update (old)", times_eager)
stats("Deferred belief update (new)", times_deferred)
print("-" * 75)
ratio = np.mean(times_eager) / max(np.mean(times_deferred), 1e-9)
print(f"  Speedup: {ratio:.1f}×")
print("=" * 75)
