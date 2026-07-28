"""Regression tests: StagedSobolSweepLocator (SimpleSobol) batches belief updates.

Before this change, StagedSobolSweepLocator called belief.update() once per
single observation -- a full SMC posterior update per step. It now buffers
observations and flushes them via belief.batch_update() in bounded chunks
(NVISION_SOBOL_BATCH_CHUNK_SIZE), the same pattern GenericSweepLocator already
used, because none of the stage-transition/stopping logic
(_check_for_dips / _noise_estimate_stable / _infer_tight_focus_window) reads
the belief -- only the raw observation history.

These tests guard the two ways that kind of deferred-update change can go
wrong: (1) observations silently lost or double-counted across a chunk
boundary or an early stop, and (2) the buffer growing unbounded instead of
flushing periodically.
"""

from __future__ import annotations

import math
import random
from types import MethodType

from nvision import CoreExperiment, NVCenterCoreGenerator, StagedSobolSweepLocator, run_loop
from nvision.models.observation import Observation
from nvision.models.observer import Observer
from nvision.sim.defaults import NVISION_SOBOL_BATCH_CHUNK_SIZE
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief, nv_lineshape_for_model
from nvision.spectra.nv_center import NVCenterLorentzianModel


class _FakeBelief:
    """Minimal stand-in exposing only what observe()/done()/finalize() touch.

    Lets the chunk-boundary/flush-timing tests run fast and deterministically,
    independent of real SMC physics.
    """

    def __init__(self) -> None:
        self.last_obs: Observation | None = None
        self.resampled = False
        self.update_calls = 0
        self.batch_update_sizes: list[int] = []

    def update(self, obs: Observation) -> None:
        self.update_calls += 1

    def batch_update(self, observations: list[Observation]) -> None:
        self.batch_update_sizes.append(len(observations))


def _make_experiment(
    rng: random.Random, x_min: float = 2.6e9, x_max: float = 3.1e9, *, free_frequency: bool = False
) -> CoreExperiment:
    gen = NVCenterCoreGenerator(x_min=x_min, x_max=x_max, variant="lorentzian")
    true_signal = gen.generate(rng)
    if free_frequency:
        # NVCenterCoreGenerator always fixes frequency (a known instrument constant,
        # not inferred -- see its docstring). Callers that specifically check
        # frequency-estimate convergence need it free instead -- swap in an
        # otherwise-identical model; typed_parameters/bounds (and hence the
        # randomized draw) are unaffected.
        true_signal.model = NVCenterLorentzianModel(
            with_hyperfine_splitting=gen.with_hyperfine_splitting,
            with_zeeman_splitting=gen.with_zeeman_splitting,
            with_fixed_frequency=False,
        )
    # noise=None -> zero measurement noise (mirrors test_simplesweep_finalize.py)
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)


def _spy_belief(belief):
    """Wrap belief.update/batch_update with call-recording, preserving behavior."""
    calls = {"update": 0, "batch_update_sizes": []}
    real_update = belief.update
    real_batch_update = belief.batch_update

    def update(self, obs):
        calls["update"] += 1
        return real_update(obs)

    def batch_update(self, observations):
        calls["batch_update_sizes"].append(len(observations))
        return real_batch_update(observations)

    belief.update = MethodType(update, belief)
    belief.batch_update = MethodType(batch_update, belief)
    return calls


def test_batch_flush_boundaries_are_exact():
    """Periodic flush cadence is exactly NVISION_SOBOL_BATCH_CHUNK_SIZE per chunk;
    the remainder is only flushed by finalize() -- proves no observation is
    silently dropped or double-counted across chunk boundaries.
    """
    n_total = 2 * NVISION_SOBOL_BATCH_CHUNK_SIZE + 7
    belief = _FakeBelief()
    locator = StagedSobolSweepLocator(
        belief=belief,
        signal_model=None,
        max_steps=n_total,  # ObservationHistory capacity must fit n_total regardless of chunk size
        domain_lo=0.0,
        domain_hi=1.0,
    )

    for _ in range(n_total):
        obs = Observation(x=0.5, signal_value=1.0, noise_std=0.01)
        locator.observe(obs)
        assert len(locator._pending_obs) <= NVISION_SOBOL_BATCH_CHUNK_SIZE

    assert belief.update_calls == 0, "must drive the belief only via batch_update(), never update()"
    assert belief.batch_update_sizes == [NVISION_SOBOL_BATCH_CHUNK_SIZE, NVISION_SOBOL_BATCH_CHUNK_SIZE]
    assert len(locator._pending_obs) == 7

    locator.finalize()

    assert belief.batch_update_sizes == [NVISION_SOBOL_BATCH_CHUNK_SIZE, NVISION_SOBOL_BATCH_CHUNK_SIZE, 7]
    assert locator._pending_obs == []
    assert sum(belief.batch_update_sizes) == n_total == locator.history.count


def test_finalize_is_a_noop_when_nothing_pending():
    """Calling finalize() again after everything is already flushed must not
    re-flush an empty batch (would otherwise register as a spurious
    batch_update([]) call downstream).
    """
    belief = _FakeBelief()
    locator = StagedSobolSweepLocator(belief=belief, signal_model=None, max_steps=10, domain_lo=0.0, domain_hi=1.0)
    for _ in range(5):
        locator.observe(Observation(x=0.5, signal_value=1.0, noise_std=0.01))

    locator.finalize()
    assert belief.batch_update_sizes == [5]

    locator.finalize()
    assert belief.batch_update_sizes == [5], "a second finalize() must not flush an empty buffer"


def test_flush_triggered_by_done_when_step_budget_exhausted():
    """done() must flush any still-pending observations the moment it is about
    to report the run finished, *before* the run loop stops calling observe()
    -- otherwise the last (< chunk size) batch would never reach the belief in
    time for the executor's post-finalize() snapshot re-sync.
    """
    belief = _FakeBelief()
    max_steps = 50  # well under NVISION_SOBOL_BATCH_CHUNK_SIZE, isolates this from periodic flush
    locator = StagedSobolSweepLocator(
        belief=belief, signal_model=None, max_steps=max_steps, domain_lo=0.0, domain_hi=1.0
    )
    for _ in range(max_steps):
        locator.step_count += 1  # mirror next()'s bookkeeping without needing the sobol generator
        locator.observe(Observation(x=0.5, signal_value=1.0, noise_std=0.01))

    assert belief.batch_update_sizes == [], "nothing should have flushed mid-run yet"
    assert locator.done() is True
    assert belief.batch_update_sizes == [max_steps]
    assert locator._pending_obs == []


def test_end_to_end_never_calls_update_directly():
    """Through the real run_loop()/Observer/finalize() path (not just direct
    observe() calls), the locator must still drive its belief exclusively via
    batch_update(), and every observation taken must reach the belief exactly
    once.
    """
    rng = random.Random(11)
    exp = _make_experiment(rng)
    parameter_bounds = {k: v for k, v in exp.true_signal.bounds.items() if not k.startswith("_")}
    belief = nv_center_smc_belief(
        parameter_bounds,
        num_particles=64,
        lineshape=nv_lineshape_for_model(exp.true_signal.model),
    )
    calls = _spy_belief(belief)

    observer = Observer(exp.true_signal, exp.x_min, exp.x_max)
    observer.watch(
        run_loop(
            StagedSobolSweepLocator,
            exp,
            rng,
            belief=belief,
            signal_model=exp.true_signal.model,
            max_steps=300,
            parameter_bounds=parameter_bounds,
        )
    )
    locator = observer.last_locator
    assert locator is not None
    locator.finalize()

    assert calls["update"] == 0
    assert calls["batch_update_sizes"], "batch_update() was never called"
    assert all(size <= NVISION_SOBOL_BATCH_CHUNK_SIZE for size in calls["batch_update_sizes"])
    assert sum(calls["batch_update_sizes"]) == locator.history.count
    assert locator._pending_obs == []


def test_sobol_converges_with_batched_updates():
    """Regression/sanity check: batching the belief updates must not degrade
    convergence quality -- a clean (zero-noise) dip should still localize far
    below the uniform-prior uncertainty, same standard as
    test_simplesweep_finalize.py's equivalent check for GenericSweepLocator.
    """
    rng = random.Random(3)
    exp = _make_experiment(rng, free_frequency=True)
    truth = float(exp.true_signal.get_param_value("frequency"))
    prior_std = (exp.x_max - exp.x_min) / math.sqrt(12)

    parameter_bounds = {k: v for k, v in exp.true_signal.bounds.items() if not k.startswith("_")}
    belief = nv_center_smc_belief(
        parameter_bounds,
        num_particles=300,
        lineshape=nv_lineshape_for_model(exp.true_signal.model),
        with_fixed_frequency=False,
    )

    observer = Observer(exp.true_signal, exp.x_min, exp.x_max)
    observer.watch(
        run_loop(
            StagedSobolSweepLocator,
            exp,
            rng,
            belief=belief,
            signal_model=exp.true_signal.model,
            max_steps=400,
            noise_std=0.01,
            parameter_bounds=parameter_bounds,
        )
    )
    locator = observer.last_locator
    assert locator is not None
    locator.finalize()

    estimate = locator.belief.estimates().get("frequency")
    assert estimate is not None
    assert abs(estimate - truth) < 0.2 * prior_std, (
        f"final frequency estimate {estimate:.4e} Hz is not well below the prior std "
        f"{prior_std:.4e} Hz around truth {truth:.4e} Hz -- batched updates may have "
        "dropped or misapplied observations"
    )
