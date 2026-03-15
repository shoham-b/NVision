"""Helper to build kwargs for the batched Bayesian adapter."""

from __future__ import annotations

from typing import Any


def build_adapter_kwargs(locator: Any) -> dict[str, Any]:
    """Build the kwargs dict used to construct an adapter from a locator instance.

    Args:
        locator: An ``NVCenterBayesianLocatorBase`` (or subclass) instance.

    Returns:
        Dict ready to pass to ``NVCenterSequentialBayesianLocatorBatched``.
    """
    kwargs: dict[str, Any] = {
        "max_evals": locator.max_evals,
        "prior_bounds": locator.prior_bounds,
        "noise_model": locator.noise_model,
        "acquisition_function": locator.acquisition_function,
        "convergence_threshold": locator.convergence_threshold,
        "min_uncertainty_reduction": locator.min_uncertainty_reduction,
        "n_monte_carlo": locator.n_monte_carlo,
        "grid_resolution": locator.grid_resolution,
        "linewidth_prior": locator.linewidth_prior,
        "distribution": locator.distribution,
        "gaussian_width_prior": locator.gaussian_width_prior,
        "split_prior": locator.split_prior,
        "amplitude_prior": locator.amplitude_prior,
        "background_prior": locator.background_prior,
        "bo_enabled": locator.bo_enabled,
        "bo_acq_func": locator.bo_acq_func,
        "bo_kappa": locator.bo_kappa,
        "bo_xi": locator.bo_xi,
        "bo_random_state": locator.bo_random_state,
        "utility_history_window": locator.utility_history_window,
        "n_warmup": locator.n_warmup,
        "locator_cls": locator.__class__,
    }
    if hasattr(locator, "pickiness"):
        kwargs["pickiness"] = locator.pickiness
    return kwargs
