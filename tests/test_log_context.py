import logging

from nvision.tools.log_context import (
    CombinationLogFilter,
    _combo_initials,
    format_combination_log_initials,
    reset_combination_log_initials,
    set_combination_log_initials,
)


def test_format_combination_log_initials():
    # Test simple abbreviation
    assert format_combination_log_initials("unit", "base", "ekf") == "U.B.E"

    # Test hyphen/underscore
    assert format_combination_log_initials("unit_cube", "some-noise", "ekf-two_three") == "UC.SN.ET"

    # Test empty parts and multiple hyphens
    assert format_combination_log_initials("unit__cube", "some--noise", "ekf") == "UC.SN.E"

    # Test empty string
    assert format_combination_log_initials("", "", "") == "?.?.?"
    assert format_combination_log_initials("_", "-", "") == "?.?.?"

    # Test repeat_idx
    assert format_combination_log_initials("unit", "base", "ekf", repeat_idx=5) == "U.B.E#5"
    assert format_combination_log_initials("unit", "base", "ekf", repeat_idx=0) == "U.B.E#0"


def test_set_and_reset_combination_log_initials():
    # Verify default
    assert _combo_initials.get() == ""

    token = set_combination_log_initials("gen", "noise", "strat", repeat_idx=2)
    assert _combo_initials.get() == "G.N.S#2"

    reset_combination_log_initials(token)
    assert _combo_initials.get() == ""


def test_combination_log_filter():
    log_filter = CombinationLogFilter()

    # Create a dummy LogRecord
    record = logging.LogRecord("name", logging.INFO, "pathname", 1, "msg", (), None)

    # Verify filter adds empty prefix when contextvar is default
    # To ensure fresh state, let's set context var to empty string explicitly just in case
    token_empty = _combo_initials.set("")

    assert _combo_initials.get() == ""
    assert log_filter.filter(record) is True
    assert record.combo_prefix == ""

    _combo_initials.reset(token_empty)

    # Verify filter adds prefix when contextvar is set
    token = set_combination_log_initials("gen", "noise", "strat")

    record2 = logging.LogRecord("name", logging.INFO, "pathname", 1, "msg", (), None)
    assert log_filter.filter(record2) is True
    assert record2.combo_prefix == "[G.N.S] "

    # Verify filter doesn't override existing combo_prefix
    record3 = logging.LogRecord("name", logging.INFO, "pathname", 1, "msg", (), None)
    record3.combo_prefix = "[CUSTOM] "
    assert log_filter.filter(record3) is True
    assert record3.combo_prefix == "[CUSTOM] "

    reset_combination_log_initials(token)
