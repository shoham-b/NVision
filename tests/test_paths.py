import pytest

from nvision.tools.paths import slugify


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        ("normal string", "normal-string"),
        ("  extra   spaces  ", "extra-spaces"),
        ("punctuation!@#$%", "punctuation"),
        ("special chars &*()", "special-chars"),
        ("", ""),
        ("!@#$%", ""),
        ("12345", "12345"),
        ("leading and trailing-", "leading-and-trailing"),
        ("-leading and trailing", "leading-and-trailing"),
        ("mixed 123 and symbols !@", "mixed-123-and-symbols"),
        (12345, "12345"),  # Non-string input
        (None, "none"),  # Non-string input
    ],
)
def test_slugify(input_value, expected):
    """Test slugify function with various edge cases."""
    assert slugify(input_value) == expected


def test_slugify_lru_cache():
    """Test that slugify function is cached using lru_cache."""
    # Ensure cache is empty or in a known state
    slugify.cache_clear()

    initial_info = slugify.cache_info()

    # Call once
    result1 = slugify("test string")
    info_after_first_call = slugify.cache_info()

    # Call again with same input
    result2 = slugify("test string")
    info_after_second_call = slugify.cache_info()

    assert result1 == "test-string"
    assert result2 == "test-string"

    # Check cache hits/misses
    assert info_after_first_call.misses == initial_info.misses + 1
    assert info_after_first_call.hits == initial_info.hits

    assert info_after_second_call.misses == info_after_first_call.misses
    assert info_after_second_call.hits == info_after_first_call.hits + 1
