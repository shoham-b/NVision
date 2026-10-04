from __future__ import annotations

import numpy as np
import pytest

from nvision.models.fisher_information import gaussian_fisher_matrix


def test_gaussian_fisher_matrix_1d() -> None:
    grad = np.array([1.0, 2.0, 3.0])
    sigma = 2.0

    result = gaussian_fisher_matrix(grad, sigma)

    # Expected: outer(g, g) / (sigma^2)
    expected_outer = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
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

    expected_outer = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    expected = expected_outer / 4.0

    np.testing.assert_allclose(result, expected)


def test_uniform_steps_inverts_lorentzian_center_freq_crlb() -> None:
    """One closed form: the steps needed for a target CRLB reproduce that CRLB."""
    from nvision.models.fisher_information import lorentzian_center_freq_crlb, uniform_steps_for_center_freq_crlb

    n = uniform_steps_for_center_freq_crlb(2e6, 0.25, 0.02, 1.5e8, 5e4)
    assert np.isclose(lorentzian_center_freq_crlb(2e6, 0.25, 0.02, n, 1.5e8), 5e4, rtol=1e-9)


def test_belief_and_history_share_one_cumulative_fisher() -> None:
    """The belief's running CRLBs equal the per-run history's final bounds for the same observations."""
    from types import SimpleNamespace

    from nvision.models.fisher_information import fisher_history
    from nvision.models.observation import Observation
    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
    from tests.noise import gaussian_noise

    belief = nv_center_smc_belief(num_particles=100, noise_model=gaussian_noise(), seed=0)
    belief.auto_resample = False
    names = list(belief.model.inner.parameter_names())
    snapshots, estimates = [], []
    for x in np.linspace(0.05, 0.4, 15):
        obs = Observation(x=float(x), signal_value=0.95, noise_std=0.02)
        belief.update(obs)
        belief.accumulate_fim(obs)
        lo, hi = belief.physical_x_bounds
        physical_obs = Observation(x=lo + float(x) * (hi - lo), signal_value=0.95, noise_std=0.02)
        snapshots.append(SimpleNamespace(obs=physical_obs, belief=belief))
        estimates.append({k: v for k, v in belief.estimates().items() if k in names})

    # the same posterior-mean point at every step for a like-for-like comparison
    final = estimates[-1]
    _, bounds_hist, degenerate = fisher_history(
        snapshots, [final] * len(snapshots), names, belief.physical_param_bounds
    )
    belief_crlbs = belief.crlb_per_param()
    assert not degenerate
    assert set(belief_crlbs) == set(names)
    assert all(np.isfinite(belief_crlbs[n]) and belief_crlbs[n] > 0 for n in names)
    # the belief evaluates each observation at its own step's estimate, so only the order of magnitude must
    # agree; directions with no information yet are NaN in the history and ridge-limited in the belief.
    informative = [n for n in names if np.isfinite(bounds_hist[-1][n])]
    assert informative
    for n in informative:
        assert 0.1 < belief_crlbs[n] / bounds_hist[-1][n] < 10
