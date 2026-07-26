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

from nvision import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    CoreExperiment,
    NVCenterCoreGenerator,
    GenericSweepLocator,
    run_loop,
)
from nvision.models.observer import Observer
from nvision.runner.convert import run_result_to_finalize_record
from nvision.runner.metrics import _scan_attempt_metrics


def _make_experiment(rng: random.Random) -> CoreExperiment:
    # Domain must be centered on NV_ZERO_FIELD_SPLITTING_HZ (the default domain is,
    # by construction) so the generator's fixed center frequency (domain midpoint)
    # agrees with the model's fixed "frequency" value (see NVCenterLorentzianModel
    # with_fixed_frequency=True) -- otherwise the two disagree and every fit looks
    # like it's "off" by a constant offset that has nothing to do with fit quality.
    gen = NVCenterCoreGenerator(
        x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="lorentzian"
    )
    true_signal = gen.generate(rng)
    # "frequency" is the probe x-axis regardless of whether it's a free/inferred
    # parameter or fixed (see NVCenterCoreGenerator) -- its bounds are always present.
    x_min, x_max = true_signal.get_param_bounds("frequency")
    assert x_min is not None
    # noise=None -> zero measurement noise
    return CoreExperiment(true_signal=true_signal, noise=None, x_min=x_min, x_max=x_max)


def test_simplesweep_zero_noise_fit_beats_prior():
    """A dense zero-noise sweep must localize the dip pair far below the prior std.

    ``frequency`` is now a fixed, known instrument constant (see
    ``NVCenterCoreGenerator``), so it can no longer serve as this test's original
    "did finalize() actually run the fit, rather than fall back to the belief
    prior" guard -- its reported value is a hardcoded constant regardless of
    whether the fit ran at all. ``zeeman_split`` is the analogous free/inferred
    quantity now (randomly drawn per-repeat, and the one that actually moves the
    two dips), so the regression guard lives there instead.
    """
    rng = random.Random(7)
    exp = _make_experiment(rng)
    freq_truth = float(exp.true_signal.get_param_value("frequency"))
    zeeman_truth = float(exp.true_signal.get_param_value("zeeman_split"))
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
    assert locator_result["frequency"] == freq_truth, "fixed frequency must be reported verbatim"

    fit = locator.fit_mode_estimates()
    assert fit is not None, "finalize() must populate fit_mode_estimates()"
    zeeman_tol = 0.05 * prior_std
    assert abs(fit["zeeman_split"] - zeeman_truth) < zeeman_tol, (
        f"zeeman_split error {abs(fit['zeeman_split'] - zeeman_truth):.3e} Hz is not well below "
        f"the prior std {prior_std:.3e} Hz — estimate fell back to the belief prior"
    )

    # End-to-end through the finalize record + metrics extraction the
    # manifest entries are built from.
    record = run_result_to_finalize_record(result, locator_result, 0, exp.x_min, exp.x_max)
    metrics = _scan_attempt_metrics([freq_truth], record)

    assert metrics["abs_err_x"] < 0.05 * prior_std, (
        f"abs_err_x={metrics['abs_err_x']:.3e} Hz is not well below the prior std "
        f"{prior_std:.3e} Hz — estimate fell back to the belief prior"
    )
