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
    drive_freq_unit : float
        Drive frequency where the measurement was taken, as a unit-cube coordinate in ``[0, 1]`` over the
        experiment's full drive-frequency range (never physical Hz).
    signal_value : float
        Measured signal value at ``drive_freq_unit``
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
        Number of repeated shots taken at ``drive_freq_unit`` and averaged into
        ``signal_value``. Defaults to 1 (a single measurement). When > 1,
        ``signal_value`` is the batch mean ȳ (precision σ/√n_shots) and
        ``sample_var`` carries the within-batch variance.
    sample_var : float | None
        Within-batch variance s² (unbiased, ddof=1) of the raw shots. ``None``
        when fewer than two shots exist (no variance is estimable). Beliefs with
        an explicit noise posterior use this as direct, model-free evidence
        about σ, orthogonal to the fit residuals.
    sweep_index : int | None
        Which repeated pass over the full measurement domain this shot came
        from, when the source data has that structure (e.g. a MATLAB file
        where every drive-frequency point is scanned once, then all scanned again, etc.
        — shot column *j* is sweep *j* for every drive-frequency point). ``None`` when the
        source has no such notion (e.g. the simulated generators, which draw
        a fresh sample on demand with no fixed sweep order).
    """

    drive_freq_unit: float
    signal_value: float
    noise_std: float = field(default=DEFAULT_MEASUREMENT_NOISE_STD)
    frequency_noise_model: tuple[dict[str, Any], ...] | None = field(default=None)
    n_shots: int = field(default=1)
    sample_var: float | None = field(default=None)
    sweep_index: int | None = field(default=None)

    def __post_init__(self) -> None:
        if not 0.0 <= self.drive_freq_unit <= 1.0:
            raise ValueError(
                f"Observation.drive_freq_unit must be a unit coordinate in [0, 1]; got {self.drive_freq_unit!r} "
                "(convert physical Hz to unit before constructing the Observation)."
            )


def aggregate_shots(
    drive_freq_unit: float,
    ys: np.ndarray,
    prior_noise_std: float,
    frequency_noise_model: tuple[dict[str, Any], ...] | None = None,
) -> Observation:
    """Collapse a batch of repeated shots at one drive frequency into a sufficient-statistic Observation.

    For k i.i.d. shots ``ys`` at ``drive_freq_unit``, the batch mean ȳ is the signal
    estimate with precision σ/√k, and the within-batch std ``s`` (ddof=1) is a
    direct estimate of the per-shot noise σ.

    Parameters
    ----------
    drive_freq_unit : float
        Unit-cube drive frequency where the shots were taken.
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
        drive_freq_unit=drive_freq_unit,
        signal_value=y_bar,
        noise_std=noise_std,
        frequency_noise_model=frequency_noise_model,
        n_shots=k,
        sample_var=sample_var,
    )


class ObservationHistory:
    """Maintains a collection of observations with parallel pre-allocated numpy arrays for fast data access.

    Observations are always stored in unit drive-frequency coordinates (``Observation.drive_freq_unit``).
    Pass ``drive_freq_bounds_phys`` (the full drive-frequency range in Hz that those coordinates span) to
    also read the history in physical Hz via :attr:`drive_freqs_phys`.
    """

    def __init__(self, max_steps: int, drive_freq_bounds_phys: tuple[float, float] | None = None):
        import numpy as np

        if drive_freq_bounds_phys is not None and not drive_freq_bounds_phys[1] > drive_freq_bounds_phys[0]:
            raise ValueError(f"drive_freq_bounds_phys must satisfy lo < hi; got {drive_freq_bounds_phys}")
        self.max_steps = max_steps
        self.drive_freq_bounds_phys = drive_freq_bounds_phys
        self.observations: list[Observation] = []
        self._drive_freqs_unit = np.empty(max_steps, dtype=np.float64)
        self._ys = np.empty(max_steps, dtype=np.float64)
        self.count = 0

    def append(self, obs: Observation) -> None:
        if self.count >= self.max_steps:
            raise ValueError(f"ObservationHistory capacity of {self.max_steps} exceeded.")

        self.observations.append(obs)
        self._drive_freqs_unit[self.count] = obs.drive_freq_unit
        self._ys[self.count] = obs.signal_value
        self.count += 1

    @property
    def drive_freqs_unit(self) -> Any:
        """Valid slice of the drive frequencies in unit coordinates. shape: (count,), values in [0, 1]."""
        return self._drive_freqs_unit[: self.count]

    @property
    def drive_freqs_phys(self) -> Any:
        """Valid slice of the drive frequencies in Hz. shape: (count,).

        Requires ``drive_freq_bounds_phys`` to have been given at construction.
        """
        if self.drive_freq_bounds_phys is None:
            raise ValueError("ObservationHistory was built without drive_freq_bounds_phys; cannot give Hz.")
        lo_phys, hi_phys = self.drive_freq_bounds_phys
        return lo_phys + self.drive_freqs_unit * (hi_phys - lo_phys)

    @property
    def ys(self) -> Any:
        """Returns the valid slice of the signal values array."""
        return self._ys[: self.count]


def gaussian_likelihood_std(obs: Observation | None) -> float:
    """Sigma for the Gaussian likelihood and Fisher terms: ``obs.noise_std`` or the global default."""
    if obs is not None and obs.noise_std > 0:
        return float(obs.noise_std)
    return DEFAULT_MEASUREMENT_NOISE_STD
