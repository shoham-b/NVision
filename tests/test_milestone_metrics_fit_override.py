"""Tests that final-state milestone metrics prefer the least-squares fit over a
collapsed SMC belief marginal.

Sweep-family locators (GenericSweepLocator, e.g. SimpleSweep) defer their belief
update to a batch flush that can collapse to a garbage marginal even when the
actual model fit is accurate (see nvision/runner/executor.py). Before this fix,
``final_err_fb``/``final_overall_uncert`` were computed straight from
``snapshot.belief``, disagreeing with ``final_est_frequency`` (which already
preferred the fit) by tens of MHz on real runs.
"""

import math

import pytest

from nvision.metrics.milestones import calculate_zeeman_metrics, extract_milestone_metrics
from nvision.models.observer import RunResult, StepSnapshot

TRUE_FREQUENCY = 2.87e9
TRUE_SPLIT = 0.01


class _FakeBelief:
    def __init__(self, estimates: dict[str, float], uncertainties: dict[str, float]):
        self._estimates = estimates
        self._uncertainties = uncertainties

    def estimates(self) -> dict[str, float]:
        return dict(self._estimates)

    def uncertainty(self) -> dict[str, float]:
        return dict(self._uncertainties)

    def reported_uncertainty(self) -> dict[str, float]:
        return dict(self._uncertainties)


class _FakeTrueSignal:
    def __init__(self, values: dict[str, float]):
        self._values = values

    def parameter_values(self) -> dict[str, float]:
        return dict(self._values)


class _FakeObs:
    x = 0.0
    signal_value = 0.0


def _make_run_result(
    belief_estimates: dict[str, float],
    fit_mode_estimates: dict[str, float] | None,
) -> RunResult:
    true_signal = _FakeTrueSignal({"frequency": TRUE_FREQUENCY, "split": TRUE_SPLIT})
    belief = _FakeBelief(belief_estimates, {"frequency": 5e5, "split": 1e-3})
    snapshot = StepSnapshot(obs=_FakeObs(), belief=belief, true_signal=true_signal)
    return RunResult(snapshots=[snapshot], true_signal=true_signal, fit_mode_estimates=fit_mode_estimates)


def test_final_state_prefers_fit_mode_estimates_over_collapsed_belief():
    """A collapsed belief marginal must not corrupt final_err_fb when a fit exists."""
    accurate_fit_freq = TRUE_FREQUENCY + 7e3  # ~7 kHz off, like the real sweep fit
    run_result = _make_run_result(
        belief_estimates={"frequency": TRUE_FREQUENCY - 4.25e7, "split": 0.05},  # collapsed, ~42 MHz off
        fit_mode_estimates={"frequency": accurate_fit_freq},
    )

    metrics = calculate_zeeman_metrics(run_result)

    assert metrics["final_err_fb"] == pytest.approx(abs(accurate_fit_freq - TRUE_FREQUENCY))
    assert metrics["final_err_fb"] < 1e5


def test_final_state_falls_back_to_belief_when_no_fit_available():
    """Non-sweep locators (fit_mode_estimates=None, e.g. SBED/Sobol) are unaffected."""
    belief_freq = TRUE_FREQUENCY + 1e4
    run_result = _make_run_result(
        belief_estimates={"frequency": belief_freq, "split": TRUE_SPLIT},
        fit_mode_estimates=None,
    )

    metrics = calculate_zeeman_metrics(run_result)

    assert metrics["final_err_fb"] == pytest.approx(abs(belief_freq - TRUE_FREQUENCY))


def test_extract_milestone_metrics_override_only_touches_named_keys():
    """override_estimates should merge in, not clobber, unrelated belief output."""
    run_result = _make_run_result(
        belief_estimates={"frequency": TRUE_FREQUENCY - 1e8, "split": TRUE_SPLIT},
        fit_mode_estimates={"frequency": TRUE_FREQUENCY},
    )

    fs = extract_milestone_metrics(
        run_result, 0, "frequency", "split", override_estimates=run_result.fit_mode_estimates
    )

    assert fs["est_fb"] == pytest.approx(TRUE_FREQUENCY)
    assert fs["err_fb"] == pytest.approx(0.0, abs=1e-6)
    # split has no override entry, so it still comes from the belief.
    assert fs["est_fc"] == pytest.approx(TRUE_SPLIT)
    assert not math.isnan(fs["overall_uncert"])
