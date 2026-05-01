from unittest import mock

import numpy as np

from nvision.models.fisher_information import single_shot_marginal_stds_from_fim


def test_single_shot_marginal_stds_from_fim_none():
    result = single_shot_marginal_stds_from_fim(None, 2)
    assert np.all(np.isnan(result))
    assert result.shape == (2,)


def test_single_shot_marginal_stds_from_fim_empty():
    result = single_shot_marginal_stds_from_fim(np.array([]), 2)
    assert np.all(np.isnan(result))
    assert result.shape == (2,)


def test_single_shot_marginal_stds_from_fim_wrong_shape():
    result = single_shot_marginal_stds_from_fim(np.ones((2, 3)), 2)
    assert np.all(np.isnan(result))
    assert result.shape == (2,)


def test_single_shot_marginal_stds_from_fim_identity():
    # If FIM is I, then (I + ridge*I) = (1 + ridge) * I
    # pinv is (1 / (1 + ridge)) * I
    # sqrt(diag(pinv)) = sqrt(1 / (1 + ridge))
    ridge = 1e-6
    n_params = 2
    fim = np.eye(n_params)
    result = single_shot_marginal_stds_from_fim(fim, n_params, ridge=ridge)

    expected_val = np.sqrt(1.0 / (1.0 + ridge))
    expected = np.full(n_params, expected_val)

    np.testing.assert_allclose(result, expected)


def test_single_shot_marginal_stds_from_fim_zero():
    # If FIM is 0, then (0 + ridge*I) = ridge * I
    # pinv is (1 / ridge) * I
    # sqrt(diag(pinv)) = sqrt(1 / ridge)
    ridge = 1e-4
    n_params = 2
    fim = np.zeros((n_params, n_params))
    result = single_shot_marginal_stds_from_fim(fim, n_params, ridge=ridge)

    expected_val = np.sqrt(1.0 / ridge)
    expected = np.full(n_params, expected_val)

    np.testing.assert_allclose(result, expected)


@mock.patch("numpy.linalg.pinv")
def test_single_shot_marginal_stds_from_fim_negative_diag(mock_pinv):
    # Test the max(0.0, cov[i, i]) logic when pinv returns negative diagonal
    # This shouldn't normally happen with positive semi-definite FIMs but is a safety check
    n_params = 2
    fim = np.eye(n_params)

    # Mock pinv to return a matrix with negative diagonals
    mock_pinv.return_value = np.array([
        [-1.0, 0.0],
        [0.0, -2.0]
    ])

    result = single_shot_marginal_stds_from_fim(fim, n_params)

    # max(0.0, negative) should be 0.0
    expected = np.zeros(n_params)

    np.testing.assert_allclose(result, expected)
    mock_pinv.assert_called_once()
