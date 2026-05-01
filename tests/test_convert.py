from nvision.runner.convert import extract_peak_estimates


def test_extract_peak_estimates_denormalization():
    """Test that position-like keys inside [0, 1] are denormalized."""
    belief_estimates = {}
    locator_result = {
        "x": 0.5,
        "peak_x": 0.2,
        "x1_hat": 0.8,
        "pos_y": 0.1,
        "freq_hz": 0.9,
    }
    x_min = 100.0
    x_max = 200.0

    result = extract_peak_estimates(belief_estimates, locator_result, x_min, x_max)

    assert result == {
        "x": 150.0,
        "peak_x": 120.0,
        "x1_hat": 180.0,
        "pos_y": 110.0,
        "freq_hz": 190.0,
    }


def test_extract_peak_estimates_keep_outside():
    """Test that position-like keys outside [0, 1] are kept as is."""
    belief_estimates = {}
    locator_result = {
        "x": 150.0,
        "peak_x": 1.2,
        "x1_hat": -0.1,
    }
    x_min = 100.0
    x_max = 200.0

    result = extract_peak_estimates(belief_estimates, locator_result, x_min, x_max)

    assert result == {
        "x": 150.0,
        "peak_x": 1.2,
        "x1_hat": -0.1,
    }


def test_extract_peak_estimates_ignore_non_numeric():
    """Test that non-numeric values in locator_result are ignored."""
    belief_estimates = {}
    locator_result = {
        "x": 0.5,
        "invalid": "string",
        "also_invalid": None,
    }
    x_min = 100.0
    x_max = 200.0

    result = extract_peak_estimates(belief_estimates, locator_result, x_min, x_max)

    assert result == {
        "x": 150.0,
    }


def test_extract_peak_estimates_non_position():
    """Test that non-position-like keys are kept as is."""
    belief_estimates = {}
    locator_result = {
        "amplitude": 0.5,
        "width": 10.0,
        "something_else": 0.1,
    }
    x_min = 100.0
    x_max = 200.0

    result = extract_peak_estimates(belief_estimates, locator_result, x_min, x_max)

    assert result == {
        "amplitude": 0.5,
        "width": 10.0,
        "something_else": 0.1,
    }


def test_extract_peak_estimates_belief_fallback():
    """Test that belief_estimates are used as fallbacks."""
    belief_estimates = {
        "frequency": 150.0,
        "split": 10.0,
    }
    locator_result = {}
    x_min = 100.0
    x_max = 200.0

    result = extract_peak_estimates(belief_estimates, locator_result, x_min, x_max)

    assert result == {
        "peak_x": 150.0,
        "x1_hat": 150.0,
        "split": 10.0,
    }


def test_extract_peak_estimates_priority():
    """Test that locator_result keys prioritize over belief_estimates."""
    belief_estimates = {
        "frequency": 150.0,
    }
    locator_result = {
        "peak_x": 180.0,
    }
    x_min = 100.0
    x_max = 200.0

    result = extract_peak_estimates(belief_estimates, locator_result, x_min, x_max)

    # In the code, locator_result items are inserted first.
    # Then `estimates.setdefault("peak_x", freq_phys)` is called, so it won't overwrite 180.0.
    # `x1_hat` is not in locator_result, so it gets the `freq_phys` fallback.
    assert result == {
        "peak_x": 180.0,
        "x1_hat": 150.0,
    }


def test_extract_peak_estimates_split_mapping():
    """Test that belief_estimates['split'] is unconditionally mapped."""
    belief_estimates = {
        "split": 15.0,
    }
    locator_result = {
        "split": 20.0,
    }
    x_min = 100.0
    x_max = 200.0

    result = extract_peak_estimates(belief_estimates, locator_result, x_min, x_max)

    # The code maps `estimates["split"] = belief_estimates["split"]`, overwriting locator_result.
    assert result == {
        "split": 15.0,
    }
