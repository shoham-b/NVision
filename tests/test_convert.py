from nvision.runner.convert import extract_peak_estimates


def test_extract_peak_estimates_passes_locator_values_through_unchanged():
    """Locator results are already physical, so values (even ones inside [0, 1]) are never rescaled."""
    belief_estimates = {}
    locator_result = {
        "x": 0.5,
        "peak_x": 0.2,
        "center_freq": 2.87e9,
        "acquisition_lo": 2.8e9,
        "amplitude": 0.5,
        "x1_hat": 0.8,
    }

    result = extract_peak_estimates(belief_estimates, locator_result)

    assert result == {
        "x": 0.5,
        "peak_x": 0.2,
        "center_freq": 2.87e9,
        "acquisition_lo": 2.8e9,
        "amplitude": 0.5,
        "x1_hat": 0.8,
    }


def test_extract_peak_estimates_ignore_non_numeric():
    """Non-numeric values in locator_result are dropped."""
    result = extract_peak_estimates({}, {"x": 150.0, "invalid": "string", "also_invalid": None})

    assert result == {"x": 150.0}


def test_extract_peak_estimates_belief_fallback():
    """The belief's center_freq seeds peak_x / x1_hat when the locator gave none."""
    result = extract_peak_estimates({"center_freq": 150.0, "split": 10.0}, {})

    assert result == {"peak_x": 150.0, "x1_hat": 150.0, "split": 10.0}


def test_extract_peak_estimates_priority():
    """Locator values win over belief fallbacks; only the missing key gets the fallback."""
    result = extract_peak_estimates({"center_freq": 150.0}, {"peak_x": 180.0})

    assert result == {"peak_x": 180.0, "x1_hat": 150.0}


def test_extract_peak_estimates_locator_center_freq_seeds_peak():
    """The locator's own center_freq fit takes priority over the belief's for the seeded keys."""
    result = extract_peak_estimates({"center_freq": 150.0}, {"center_freq": 2.0})

    assert result == {"center_freq": 2.0, "peak_x": 2.0, "x1_hat": 2.0}


def test_extract_peak_estimates_split_mapping():
    """belief_estimates['split'] is unconditionally mapped, overwriting the locator's."""
    result = extract_peak_estimates({"split": 15.0}, {"split": 20.0})

    assert result == {"split": 15.0}
