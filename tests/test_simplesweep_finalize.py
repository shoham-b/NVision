"""Regression tests: SimpleSweep's recorded estimate must come from its dip fit.

Before the fix, locator.finalize() was never called, so GenericSweepLocator's
parabolic fit never ran and the finalize record silently fell back to the
un-updated belief prior (estimate = domain center, uncert = uniform-prior std
~ domain/sqrt(12)). Every "vs sweep" comparison built on abs_err_x was
meaningless as a result.
"""

from __future__ import annotations

import math
import random

from nvision import CoreExperiment, NVCenterCoreGenerator, GenericSweepLocator, run_loop
from nvision.models.observer import Observer
from nvision.runner.convert import run_result_to_finalize_record
from nvision.runner.metrics import _scan_attempt_metrics
from nvision.spectra.nv_center import DEFAULT_NV_CENTER_FREQ_X_MAX, DEFAULT_NV_CENTER_FREQ_X_MIN


def _make_experiment(rng: random.Random) -> CoreExperiment:
    # x_min/x_max must stay symmetric around NV_ZERO_FIELD_SPLITTING_HZ -- the
    # model's fixed (not inferred) frequency parameter -- or the "true" center_freq
    # the generator draws drifts away from what the locator's fit assumes.
    gen = NVCenterCoreGenerator(x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="lorentzian")
    true_signal = gen.generate(rng)
    x_min, x_max = None, None
    for name, bounds in true_signal.bounds.items():
        if "frequency" in name:
            x_min, x_max = bounds
            break
    assert x_min is not None
    # noise=None -> zero measurement noise
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)


def test_simplesweep_zero_noise_fit_beats_prior():
    """A dense zero-noise sweep must localize the dip far below the prior std."""
    rng = random.Random(7)
    exp = _make_experiment(rng)
    truth = float(exp.true_signal.get_param_value("frequency"))
    prior_std = (exp.x_max - exp.x_min) / math.sqrt(12)

    # Full physical bounds for every model parameter (not just frequency) —
    # GenericSweepLocator's finalize() now requires a complete parameter set
    # to fit the model; it no longer falls back to a boundless peak-detection
    # heuristic when bounds are incomplete.
    parameter_bounds = {k: v for k, v in exp.true_signal.bounds.items() if not k.startswith("_")}

    observer = Observer(exp.true_signal, exp.x_min, exp.x_max)
    result = observer.watch(
        run_loop(
            GenericSweepLocator,
            exp,
            rng,
            max_steps=1000,
            parameter_bounds=parameter_bounds,
        )
    )
    locator = observer.last_locator
    assert locator is not None

    # Mirror the executor's finalize path (executor._run_single_repeat).
    locator.finalize()
    locator_result = locator.result()

    assert "frequency" in locator_result, "finalize() must produce a frequency fit"
    assert abs(locator_result["frequency"] - truth) < 0.05 * prior_std
    assert locator_result["uncert"] < 0.1 * prior_std

    # End-to-end through the finalize record + metrics extraction the
    # manifest entries are built from.
    record = run_result_to_finalize_record(result, locator_result, 0, exp.x_min, exp.x_max)
    metrics = _scan_attempt_metrics([truth], record)

    assert metrics["abs_err_x"] < 0.05 * prior_std, (
        f"abs_err_x={metrics['abs_err_x']:.3e} Hz is not well below the prior std "
        f"{prior_std:.3e} Hz — estimate fell back to the belief prior"
    )
    assert metrics["uncert"] < 0.1 * prior_std
    # The exact prior-std value was the bug's fingerprint; make sure it cannot return.
    assert not math.isclose(metrics["uncert"], prior_std, rel_tol=1e-6)
