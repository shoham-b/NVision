from __future__ import annotations

import numpy as np
import pytest

from nvision.models.fisher_information import gaussian_fisher_matrix


def test_gaussian_fisher_matrix_1d() -> None:
    grad = np.array([1.0, 2.0, 3.0])
    sigma = 2.0

    result = gaussian_fisher_matrix(grad, sigma)

    # Expected: outer(g, g) / (sigma^2)
    expected_outer = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ])
    expected = expected_outer / 4.0

    np.testing.assert_allclose(result, expected)

def test_gaussian_fisher_matrix_2d() -> None:
    grad = np.array([[1.0, 2.0], [3.0, 4.0]])
    sigma = 2.0

    # Note: np.outer flattens the arrays first if they are not 1D
    result = gaussian_fisher_matrix(grad, sigma)

    expected_outer = np.outer(grad.flatten(), grad.flatten())
    expected = expected_outer / 4.0

    np.testing.assert_allclose(result, expected)

def test_gaussian_fisher_matrix_zero_sigma() -> None:
    grad = np.array([1.0, 2.0, 3.0])
    sigma = 0.0

    with pytest.warns(RuntimeWarning, match="divide by zero"):
        result = gaussian_fisher_matrix(grad, sigma)

    assert np.all(np.isinf(result))

def test_gaussian_fisher_matrix_type_handling() -> None:
    # Test with list input
    grad = [1.0, 2.0, 3.0]
    sigma = 2.0

    result = gaussian_fisher_matrix(grad, sigma)

    expected_outer = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ])
    expected = expected_outer / 4.0

    np.testing.assert_allclose(result, expected)
