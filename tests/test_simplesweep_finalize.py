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

import pytest

from nvision import CoreExperiment, GenericSweepLocator, NVCenterCoreGenerator, run_loop
from nvision.models.observer import Observer
from nvision.runner.convert import run_result_to_finalize_record
from nvision.runner.metrics import _scan_attempt_metrics
from nvision.spectra.nv_center import NVCenterLorentzianModel


def _make_experiment(rng: random.Random) -> CoreExperiment:
    gen = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian")
    true_signal = gen.generate(rng)
    # NVCenterCoreGenerator always fixes center_freq (a known instrument constant,
    # not inferred -- see its docstring), so the generated model's center_freq isn't
    # a free/fit parameter. This test specifically checks that GenericSweepLocator's
    # dip fit recovers *center_freq* (not, say, zeeman_split), so swap in an
    # otherwise-identical model with center_freq free -- typed_parameters/bounds
    # (and hence the randomized draw) are unaffected.
    true_signal.model = NVCenterLorentzianModel(
        hyperfine=gen.hyperfine,
        infer_hyperfine=gen.infer_hyperfine,
        with_zeeman_splitting=gen.with_zeeman_splitting,
        with_fixed_center_freq=False,
    )
    drive_freq_min_phys, drive_freq_max_phys = true_signal.get_param_bounds("center_freq")
    assert drive_freq_min_phys is not None
    # noise=None -> zero measurement noise
    return CoreExperiment(
        true_signal=true_signal,
        noise=None,
        drive_freq_min_phys=drive_freq_min_phys,
        drive_freq_max_phys=drive_freq_max_phys,
    )


@pytest.mark.timeout(200)
def test_simplesweep_zero_noise_fit_beats_prior():
    """A dense zero-noise sweep must localize the dip far below the prior std.

    Runs in ~20s standalone (1000-step dense sweep -> multi-start curve_fit),
    but measured ~105s under the full suite's coverage instrumentation
    (pyproject.toml's `--cov=nvision` addopts), which disproportionately slows
    down this kind of call-heavy Python loop -- see
    test_sweep_fit_asymmetric_triplet_shallow_line_hidden in
    test_generic_sweep_locator_fit.py for the same root cause. That pushes it
    past the default 60s pytest-timeout, which on Windows falls back to its
    "thread" method and hard-kills the whole pytest process rather than just
    failing this test. 200s keeps headroom under coverage plus real load.
    """
    rng = random.Random(7)
    exp = _make_experiment(rng)
    truth = float(exp.true_signal.get_param_value("center_freq"))
    prior_std = (exp.drive_freq_max_phys - exp.drive_freq_min_phys) / math.sqrt(12)

    # Full physical bounds for every model parameter (not just center_freq) —
    # GenericSweepLocator's finalize() now requires a complete parameter set
    # to fit the model; it no longer falls back to a boundless peak-detection
    # heuristic when bounds are incomplete.
    parameter_bounds = {k: v for k, v in exp.true_signal.bounds.items() if not k.startswith("_")}

    observer = Observer(exp.true_signal, exp.drive_freq_min_phys, exp.drive_freq_max_phys)
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

    assert "center_freq" in locator_result, "finalize() must produce a center_freq fit"
    assert abs(locator_result["center_freq"] - truth) < 0.05 * prior_std
    assert locator_result["uncert"] < 0.1 * prior_std

    # End-to-end through the finalize record + metrics extraction the
    # manifest entries are built from.
    record = run_result_to_finalize_record(result, locator_result, 0)
    metrics = _scan_attempt_metrics([truth], record)

    assert metrics["abs_err_x"] < 0.05 * prior_std, (
        f"abs_err_x={metrics['abs_err_x']:.3e} Hz is not well below the prior std "
        f"{prior_std:.3e} Hz — estimate fell back to the belief prior"
    )
    assert metrics["uncert"] < 0.1 * prior_std
    # The exact prior-std value was the bug's fingerprint; make sure it cannot return.
    assert not math.isclose(metrics["uncert"], prior_std, rel_tol=1e-6)
