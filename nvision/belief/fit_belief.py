"""Belief stand-in for locators that infer by least-squares fitting instead of an SMC particle filter.

:class:`FitBelief` carries what the snapshot / metrics / plotting code reads off a belief at each step
(estimates, uncertainties, the dips the data show, the parameter bounds), without any particles. The locator
that owns it (``GenericSweepLocator``) pushes each new fit into it with :meth:`set_fit`;
between fits the estimates and uncertainties hold their last values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.stats import norm

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.belief.dip_detection import DipCandidate, find_dips, hwhm_fwhm_prior_hz
from nvision.models.observation import Observation, ObservationHistory
from nvision.spectra.nv_center_drive import is_drive_param_dict


class FitBelief(AbstractMarginalDistribution):
    """Gaussian-marginal belief whose mean and standard deviation come from the latest least-squares fit.

    Built from a *template* belief (any belief of the same model, e.g. an SMC belief with a handful of
    particles): the template supplies the model, the parameter bounds, the priors and the starting estimates
    and uncertainties (the prior's), and is not kept. The measured data are shared with the locator's
    :class:`~nvision.models.observation.ObservationHistory`; a :meth:`copy` (the per-step snapshot) is cheap
    because it only remembers how many of those measurements it covers.

    Parameters
    ----------
    template : AbstractMarginalDistribution
        Belief of the same model; read once.
    history : ObservationHistory
        The locator's measurements in unit drive-frequency coordinates, grown by the locator.
    grid_points : int
        Points in one pass of the sweep grid (later passes repeat it, and are averaged for the dip scan).
    noise_std : float
        Standard deviation of one measurement; the dip scan treats it as known.
    """

    def __init__(
        self,
        template: AbstractMarginalDistribution,
        history: ObservationHistory,
        grid_points: int,
        noise_std: float,
    ) -> None:
        super().__init__(model=template.model)
        if grid_points < 1:
            raise ValueError(f"grid_points must be >= 1, got {grid_points!r}")
        if not noise_std > 0:
            raise ValueError(f"noise_std must be positive, got {noise_std!r}")
        self.physical_param_bounds: dict[str, tuple[float, float]] = {
            name: (float(lo), float(hi)) for name, (lo, hi) in template.physical_param_bounds.items()
        }
        self.priors = getattr(template, "priors", None)
        self._names: tuple[str, ...] = tuple(self.model.parameter_names())
        self._drive_freq_bounds_phys: tuple[float, float] = tuple(float(v) for v in template.drive_freq_bounds_phys)  # type: ignore[assignment]
        self._prior_estimates: dict[str, float] = {n: float(v) for n, v in template.estimates().items()}
        self._prior_uncertainties: dict[str, float] = {n: float(v) for n, v in template.uncertainty().items()}
        self._estimates: dict[str, float] = dict(self._prior_estimates)
        self._uncertainties: dict[str, float] = dict(self._prior_uncertainties)
        self._history = history
        self._grid_points = int(grid_points)
        self._noise_std = float(noise_std)
        # The boxcar dip scan needs the dip-width prior implied by the bounds; a model with no width parameter
        # (a toy signal) cannot be scanned.
        names_with_bounds = set(self.physical_param_bounds)
        self.can_scan_dips: bool = bool(
            names_with_bounds & {"hwhm", "hwhm_lorentz"}
            or {"saturation", "sigma_inhom"} <= names_with_bounds
            or is_drive_param_dict(self.physical_param_bounds)
        )
        # (mean, std, min, max) of the dip FWHM implied by the prior bounds; resolved at the first dip scan.
        self._gamma_prior: tuple[float, float, float, float] | None = None
        # Measurements this snapshot covers; ``_step_count`` is the key Observer.watch uses to dedupe snapshots.
        self._count = 0
        self._step_count = 0
        # Shared by every copy: dips found for (measurement count, false-alarm level).
        self._dip_cache: dict[tuple[int, float | None], list[DipCandidate]] = {}

    # ------------------------------------------------------------------ state pushed by the locator

    def advance(self, obs: Observation) -> None:
        """Count ``obs`` (already appended to the shared history) as covered by this belief."""
        self.last_obs = obs
        self.resampled = False
        self.sync()

    def sync(self) -> None:
        """Cover every measurement in the shared history (for data added without :meth:`advance`)."""
        self._count = self._history.count
        self._step_count = self._count

    def set_fit(self, estimates: Mapping[str, float], stds: Mapping[str, float]) -> None:
        """Adopt a fit: ``estimates`` / ``stds`` (physical units) of every model parameter.

        A non-finite fit standard deviation (a degenerate covariance: the data do not constrain that
        parameter) keeps the parameter's prior uncertainty rather than reporting a made-up precision.
        """
        for name in self._names:
            if name not in estimates:
                raise ValueError(f"fit has no estimate for parameter {name!r}; got {sorted(estimates)}")
            self._estimates[name] = float(estimates[name])
            std = float(stds.get(name, math.nan))
            self._uncertainties[name] = std if math.isfinite(std) and std > 0 else self._prior_uncertainties[name]

    # ------------------------------------------------------------------ dips

    def current_dips(self, p_fa: float | None = None) -> list[DipCandidate]:
        """Dips the measurements covered so far show, most significant first (``find_dips``).

        Scans the first pass as measured; once a pass is complete, the mean over the completed passes (whose
        noise is ``noise_std / sqrt(passes)``). ``p_fa`` is the family-wise false-alarm level (default
        ``NVISION_DIP_P_FA``). Empty until 5 measurements exist.
        """
        if not self.can_scan_dips:
            raise ValueError("this model has no dip-width parameter in its bounds; the dip scan cannot run")
        k = self._count
        if k < 5:
            return []
        key = (k, p_fa)
        cached = self._dip_cache.get(key)
        if cached is None:
            x_unit, ys, passes = self.averaged_data(k)
            if self._gamma_prior is None:
                self._gamma_prior = hwhm_fwhm_prior_hz(self.physical_param_bounds)
            lo, hi = self._drive_freq_bounds_phys
            cached = find_dips(
                lo + x_unit * (hi - lo),
                ys,
                self._noise_std / math.sqrt(passes),
                math.inf,
                *self._gamma_prior,
                p_fa=p_fa,
                assume_sorted=True,
            )
            self._dip_cache[key] = cached
        return list(cached)

    def averaged_data(self, count: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        """The data a fit or dip scan should use after ``count`` measurements: ``(x_unit, ys, passes)``.

        Before the first pass completes, the measurements as taken; afterwards the per-grid-point mean over the
        completed passes (a partial pass is left out). shapes: ``(m,)``, ``(m,)``; ``x_unit`` ascending in [0, 1].
        """
        k = self._count if count is None else count
        n = self._grid_points
        passes = k // n
        x_all = self._history.drive_freqs_unit
        y_all = self._history.ys
        if passes == 0:
            return x_all[:k], y_all[:k], 1
        return x_all[:n], y_all[: passes * n].reshape(passes, n).mean(axis=0), passes

    # ------------------------------------------------------------------ AbstractMarginalDistribution

    @property
    def drive_freq_bounds_phys(self) -> tuple[float, float]:
        return self._drive_freq_bounds_phys

    def update(self, obs: Observation) -> None:
        raise NotImplementedError("FitBelief is not Bayesian-updated; the locator pushes fits via set_fit.")

    def estimates(self) -> dict[str, float]:
        return dict(self._estimates)

    def mode_estimates(self) -> dict[str, float]:
        return dict(self._estimates)

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        return ParameterValues.from_mapping(list(self._names), self._uncertainties)

    def entropy(self) -> float:
        """Differential entropy (nats) of the independent Gaussian marginals."""
        return float(sum(0.5 * math.log(2.0 * math.pi * math.e * self._uncertainties[n] ** 2) for n in self._names))

    def converged(self, threshold: float) -> bool:
        from nvision.metrics.milestones import belief_target_params_converged

        return belief_target_params_converged(self, convergence_threshold=threshold)

    def copy(self) -> FitBelief:
        clone = object.__new__(FitBelief)
        clone.__dict__.update(self.__dict__)
        clone._estimates = dict(self._estimates)
        clone._uncertainties = dict(self._uncertainties)
        return clone

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        rng = np.random.default_rng(0)
        draws = [rng.normal(self._estimates[p], self._uncertainties[p], size=n) for p in self._names]
        return ParameterValues(self._names, tuple(draws))

    def marginal_pdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        return norm.pdf(x, self._estimates[param_name], self._uncertainties[param_name])

    def marginal_cdf(self, param_name: str, x: np.ndarray) -> np.ndarray:
        return norm.cdf(x, self._estimates[param_name], self._uncertainties[param_name])
