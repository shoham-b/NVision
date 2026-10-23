from types import SimpleNamespace

from nvision.runner.convert import extract_peak_estimates, run_result_to_finalize_record


def test_extract_peak_estimates_passes_locator_values_through_unchanged():
    """Locator results are already physical, so values (even ones inside [0, 1]) are never rescaled."""
    belief_estimates = {}
    locator_result = {
        "x": 0.5,
        "peak_x": 0.2,
        "zero_field_freq": 2.87e9,
        "acquisition_lo": 2.8e9,
        "amplitude": 0.5,
        "x1_hat": 0.8,
    }

    result = extract_peak_estimates(belief_estimates, locator_result)

    assert result == {
        "x": 0.5,
        "peak_x": 0.2,
        "zero_field_freq": 2.87e9,
        "acquisition_lo": 2.8e9,
        "amplitude": 0.5,
        "x1_hat": 0.8,
    }


def test_extract_peak_estimates_ignore_non_numeric():
    """Non-numeric values in locator_result are dropped."""
    result = extract_peak_estimates({}, {"x": 150.0, "invalid": "string", "also_invalid": None})

    assert result == {"x": 150.0}


def test_extract_peak_estimates_belief_fallback():
    """The belief's zero_field_freq seeds peak_x / x1_hat when the locator gave none."""
    result = extract_peak_estimates({"zero_field_freq": 150.0, "hyperfine_shift": 10.0}, {})

    assert result == {"peak_x": 150.0, "x1_hat": 150.0, "hyperfine_shift": 10.0}


def test_extract_peak_estimates_priority():
    """Locator values win over belief fallbacks; only the missing key gets the fallback."""
    result = extract_peak_estimates({"zero_field_freq": 150.0}, {"peak_x": 180.0})

    assert result == {"peak_x": 180.0, "x1_hat": 150.0}


def test_extract_peak_estimates_locator_zero_field_freq_seeds_peak():
    """The locator's own zero_field_freq fit takes priority over the belief's for the seeded keys."""
    result = extract_peak_estimates({"zero_field_freq": 150.0}, {"zero_field_freq": 2.0})

    assert result == {"zero_field_freq": 2.0, "peak_x": 2.0, "x1_hat": 2.0}


def test_extract_peak_estimates_split_mapping():
    """belief_estimates['hyperfine_shift'] is unconditionally mapped, overwriting the locator's."""
    result = extract_peak_estimates({"hyperfine_shift": 15.0}, {"hyperfine_shift": 20.0})

    assert result == {"hyperfine_shift": 15.0}


def _stub_run_result():
    belief = SimpleNamespace(
        reported_uncertainty=lambda: {"zeeman_shift": 3.0},
        entropy=lambda: 1.5,
        converged=lambda threshold: False,
    )
    return SimpleNamespace(
        final_estimates=lambda: {"zeeman_shift": 30.0e6, "zero_field_freq": 2.9e9},
        fit_mode_estimates=None,
        fit_mode_uncertainties=None,
        snapshots=[SimpleNamespace(belief=belief)],
    )


def test_finalize_record_without_fit_reports_no_estimate():
    """A sweep whose fit found no dip must not pass its un-updated belief off as an estimate."""
    record = run_result_to_finalize_record(_stub_run_result(), {"fit_failed": True, "fit_chi2_gain": 4.5}, 7)
    assert record["fit_failed"] is True
    assert record["measurements"] == 1  # the repeat is kept, not dropped
    for key in ("zeeman_shift", "zero_field_freq", "peak_x", "x1_hat", "uncert_zeeman_shift"):
        assert key not in record


def test_finalize_record_with_fit_keeps_belief_fallbacks():
    record = run_result_to_finalize_record(_stub_run_result(), {"fit_failed": False}, 7)
    assert record["zeeman_shift"] == 30.0e6
    assert record["uncert_zeeman_shift"] == 3.0
