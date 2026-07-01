"""Observation dataclass for a single measurement."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nvision.models.measurement_noise import DEFAULT_MEASUREMENT_NOISE_STD

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "DEFAULT_MEASUREMENT_NOISE_STD",
    "Observation",
    "ObservationHistory",
    "aggregate_shots",
    "gaussian_likelihood_std",
]


@dataclass
class Observation:
    """Single measurement observation.

    Attributes
    ----------
    x : float
        Position where measurement was taken
    signal_value : float
        Measured signal value at x
    noise_std : float
        Known (or estimated) standard deviation of the measurement noise.
        Used by belief distributions for the likelihood function.
        Defaults to 0.05 (suitable for normalized [0, 1] signals with no noise).
    frequency_noise_model : tuple[dict[str, Any], ...] | None
        Optional structured description of over-frequency noise components used
        to generate this observation. When present, Bayesian updates can use a
        component-specific likelihood (e.g. Poisson counting) instead of the
        default Gaussian approximation.
    n_shots : int
        Number of repeated shots taken at ``x`` and averaged into
        ``signal_value``. Defaults to 1 (a single measurement). When > 1,
        ``signal_value`` is the batch mean ȳ (precision σ/√n_shots) and
        ``sample_var`` carries the within-batch variance.
    sample_var : float | None
        Within-batch variance s² (unbiased, ddof=1) of the raw shots. ``None``
        when fewer than two shots exist (no variance is estimable). Beliefs with
        an explicit noise posterior use this as direct, model-free evidence
        about σ, orthogonal to the fit residuals.
    """

    x: float
    signal_value: float
    noise_std: float = field(default=DEFAULT_MEASUREMENT_NOISE_STD)
    frequency_noise_model: tuple[dict[str, Any], ...] | None = field(default=None)
    n_shots: int = field(default=1)
    sample_var: float | None = field(default=None)


def aggregate_shots(
    x: float,
    ys: np.ndarray,
    prior_noise_std: float,
    frequency_noise_model: tuple[dict[str, Any], ...] | None = None,
) -> Observation:
    """Collapse a batch of repeated shots at one frequency into a sufficient-statistic Observation.

    For k i.i.d. shots ``ys`` at position ``x``, the batch mean ȳ is the signal
    estimate with precision σ/√k, and the within-batch std ``s`` (ddof=1) is a
    direct estimate of the per-shot noise σ.

    Parameters
    ----------
    x : float
        Position where the shots were taken.
    ys : np.ndarray
        Raw per-shot signal values (length k).
    prior_noise_std : float
        Fallback per-shot σ used when the empirical std is unavailable (k < 2)
        or degenerate (s == 0). The returned ``noise_std`` is then prior/√k.
    frequency_noise_model : tuple[dict, ...] | None
        Passed through to the returned Observation unchanged.

    Returns
    -------
    Observation
        ``signal_value`` = ȳ, ``noise_std`` = s/√k (or prior/√k fallback),
        ``n_shots`` = k, ``sample_var`` = s² (None when k < 2).
    """
    import numpy as np

    ys = np.asarray(ys, dtype=np.float64)
    k = int(ys.size)
    if k == 0:
        raise ValueError("aggregate_shots requires at least one shot.")
    y_bar = float(np.mean(ys))
    sqrt_k = math.sqrt(k)
    if k >= 2:
        s = float(np.std(ys, ddof=1))
        noise_std = s / sqrt_k if s > 0 else prior_noise_std / sqrt_k
        sample_var = s * s
    else:
        noise_std = prior_noise_std / sqrt_k
        sample_var = None
    return Observation(
        x=x,
        signal_value=y_bar,
        noise_std=noise_std,
        frequency_noise_model=frequency_noise_model,
        n_shots=k,
        sample_var=sample_var,
    )


class ObservationHistory:
    """Maintains a collection of observations with parallel pre-allocated numpy arrays for fast data access."""

    def __init__(self, max_steps: int):
        import numpy as np

        self.max_steps = max_steps
        self.observations: list[Observation] = []
        self._xs = np.empty(max_steps, dtype=np.float64)
        self._ys = np.empty(max_steps, dtype=np.float64)
        self.count = 0

    def append(self, obs: Observation) -> None:
        if self.count >= self.max_steps:
            raise ValueError(f"ObservationHistory capacity of {self.max_steps} exceeded.")

        self.observations.append(obs)
        self._xs[self.count] = obs.x
        self._ys[self.count] = obs.signal_value
        self.count += 1

    @property
    def xs(self) -> Any:
        """Returns the valid slice of the x positions array."""
        return self._xs[: self.count]

    @property
    def ys(self) -> Any:
        """Returns the valid slice of the signal values array."""
        return self._ys[: self.count]


def gaussian_likelihood_std(obs: Observation | None) -> float:
    """Sigma for the Gaussian likelihood and Fisher terms: ``obs.noise_std`` or the global default."""
    if obs is not None and obs.noise_std > 0:
        return float(obs.noise_std)
    return DEFAULT_MEASUREMENT_NOISE_STD
