"""Generic uniform sweep locator — a concrete SweepingLocator implementation."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.models.fisher_information import lorentzian_zero_field_freq_crlb
from nvision.models.observation import Observation, ObservationHistory
from nvision.sim.defaults import (
    NVISION_CONVERGENCE_THRESHOLD,
    NVISION_MAX_MEASUREMENTS,
    NVISION_SWEEP_BELIEF_CHUNK,
    NVISION_SWEEP_FIT_EARLY_STOP_SIGMAS,
)
from nvision.sim.locs.bayesian.sequential_bayesian_locator import NVISION_CONVERGENCE_PATIENCE, ConvergenceStopRules
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator
from nvision.sim.locs.search_grid import SWEEP_POINTS_PER_FWHM, prior_fwhm_phys, uniform_grid_phys
from nvision.spectra.signal import SignalModel
from nvision.tools.renamed_env import reject_renamed_env_vars

reject_renamed_env_vars()

if TYPE_CHECKING:
    pass

# Per-fit evaluation budget.  curve_fit's default tolerances (~1.5e-8) converge in
# ~50-200 evaluations; a larger budget only matters if a start is far off.
_FIT_MAXFEV: int = int(os.getenv("NVISION_SWEEP_FIT_MAXFEV", "4000"))

# The zero_field_freq step tolerance (curve_fit xtol) is derived from the expected
# zero_field_freq CRLB rather than hard-coded: refine until the zero_field_freq is pinned to
# this fraction of its Cramér-Rao floor.  Refining tighter than the CRLB is
# meaningless (it is below the information the data carries) and only burns
# iterations; refining looser leaves accuracy on the table.  This mirrors how the
# sbed locator gates convergence on NVISION_ZERO_FIELD_FREQ_CRLB_SAFETY_FACTOR × CRLB — here
# we target CRLB × _FIT_CRLB_TOL_FRAC as the optimizer's stopping precision.
_FIT_CRLB_TOL_FRAC: float = float(os.getenv("NVISION_SWEEP_FIT_CRLB_TOL_FRAC", "0.1"))
# Clamp the resulting relative tolerance to a sane band so a degenerate CRLB
# estimate can't make the fit either never converge (too tight) or quit
# immediately (too loose).
_FIT_XTOL_MIN: float = float(os.getenv("NVISION_SWEEP_FIT_XTOL_MIN", "1e-10"))
_FIT_XTOL_MAX: float = float(os.getenv("NVISION_SWEEP_FIT_XTOL_MAX", "1e-6"))

# Multi-start grid sizes.  The fit is non-convex: at low SNR a single start lands
# in a wrong local minimum.  Restarting from a grid of (zero_field_freq × zeeman_shift)
# and keeping the lowest-residual fit escapes those basins — this is where extra
# compute actually refines the result (more iterations per fit do not).  The grid
# must be fine enough: a too-coarse grid can miss the true basin and do worse than
# a smart single start.
_FIT_ZERO_FIELD_FREQ_STARTS: int = int(os.getenv("NVISION_SWEEP_FIT_ZERO_FIELD_FREQ_STARTS", "5"))
_FIT_ZEEMAN_STARTS: int = int(os.getenv("NVISION_SWEEP_FIT_ZEEMAN_STARTS", "4"))

# Chi-square profile over ``zeeman_shift`` for a fixed ``zero_field_freq`` (see ``_zeeman_profile_starts``): the
# number of best profile minima raced as extra starts, the widths tried as fractions of the data-driven width
# estimate (a coarse sweep overestimates it, since its step exceeds the true linewidth), and the cap on the
# number of ``zeeman_shift`` grid points.
_FIT_PROFILE_STARTS: int = 3
_FIT_PROFILE_WIDTH_FRACTIONS: tuple[float, ...] = (1.0, 0.5, 0.25)
_FIT_PROFILE_MAX_POINTS: int = 1000

# Peak-SNR (dip depth / noise_std) below which the multi-start grid is engaged.
# Above it, the two smart starts already converge to sub-MHz accuracy, so the grid
# (~20 extra fits) is skipped.  Set from empirical accuracy: at SNR>=6 the smart
# starts give sub-MHz median with no failures; below that a single start starts
# landing in wrong basins and the grid is needed.
_FIT_GRID_SNR: float = float(os.getenv("NVISION_SWEEP_FIT_GRID_SNR", "6.0"))

# Minimum chi-square gain of the winning fit over the best flat baseline (a constant at the data's mean), in
# units of the measurement noise. curve_fit can "converge" on a plateau where the model does not touch the
# dip (a start far from it): such a fit is no better than a flat line, and reporting its parameters would pass
# the start point off as an estimate. A dip the stop rule accepted gains far more than this (the boxcar scan
# fires at z > ~4.6, i.e. a gain above ~20), while fitting noise alone gains about the parameter count.
_FIT_MIN_CHI2_GAIN: float = float(os.getenv("NVISION_SWEEP_FIT_MIN_CHI2_GAIN", "9.0"))


class SweepFitNoDipError(RuntimeError):
    """The sweep's best fit explains the data no better than a flat line: the sweep holds no detectable dip.

    At low SNR on a coarse grid this is a measured outcome, not a bug (the model evaluated at the *true*
    parameters can itself gain less than ``_FIT_MIN_CHI2_GAIN`` over flat). ``GenericSweepLocator.finalize``
    records it (``fit_failed``) and reports no estimate; every other fit failure still raises.

    ``chi2_gain`` is the chi-square gain over the flat baseline, in units of the measurement noise.
    """

    def __init__(self, message: str, chi2_gain: float) -> None:
        super().__init__(message)
        self.chi2_gain = chi2_gain

    def __reduce__(self):
        # Default RuntimeError pickling replays only ``args`` (the message), dropping ``chi2_gain``; an unpicklable
        # exception raised in a pool worker breaks the whole ProcessPool instead of reaching the parent.
        return (type(self), (self.args[0], self.chi2_gain))


# Experimental: evaluate curve_fn via the model's vectorized float32 Numba kernel
# instead of a per-point Python loop in float64. Off by default -- needs an A/B
# against the float64 path (see NVISION_SWEEP_FIT_FLOAT32_DIFF_STEP below for why
# float32 alone isn't enough) before it can replace the default.
NVISION_SWEEP_FIT_FLOAT32: bool = os.getenv("NVISION_SWEEP_FIT_FLOAT32", "0") not in ("0", "false", "False")
# Relative finite-difference step for curve_fit's numerical Jacobian when the
# float32 path is active. curve_fit's default step (~sqrt(float64 eps) ~= 1.5e-8)
# is finer than float32's ~1.2e-7 relative resolution, so the perturbed model
# output rounds back to the same float32 value and the Jacobian column goes to
# zero. A step well above the float32 noise floor keeps the gradient nonzero.
_FIT_FLOAT32_DIFF_STEP: float = float(os.getenv("NVISION_SWEEP_FIT_FLOAT32_DIFF_STEP", "1e-4"))


class GenericSweepLocator(ConvergenceStopRules, SweepingLocator):
    """Uniform grid sweep over the whole drive-frequency axis, then a least-squares model fit.

    The grid spacing is the prior FWHM divided by ``spacing_divisor`` (3 by default: three points inside the
    dip, so a dip cannot be stepped over and ``find_dips`` has the cluster it needs), where the FWHM is
    ``2 * effective_hwhm`` of the belief's *prior* estimate -- never the true signal's width. The number of
    steps follows from that spacing and the axis width. No refocusing -- every grid point is measured -- but the
    belief is updated after each measurement, so its uncertainty can be followed step by step like any other
    locator's. The grid is repeated (averaging the passes) until the stopping rules Bayesian-SBED uses
    (:class:`ConvergenceStopRules`: every target uncertainty under its ceiling, and a dip found) has fired (the rules
    run every step, but the sweep only stops at the end of a pass), for at most ``max_measurements`` measurements
    (whole passes only, at least one). The reported estimate is the final fit to the averaged passes (see
    :meth:`finalize`).

    Parameters
    ----------
    belief : AbstractMarginalDistribution
        Belief distribution (required by Locator parent class); its prior sets the dip width.
    signal_model : SignalModel
        Signal model for sweep detection.
    max_steps : int
        Upper bound on the step count (0 = none). The count itself is derived from the prior FWHM,
        ``spacing_divisor`` and the axis width; exceeding the bound raises.
    spacing_divisor : int | None, default 3
        Grid points per FWHM (spacing = FWHM / spacing_divisor). ``None`` selects a fixed-count sweep of exactly
        ``max_steps`` evenly spaced points instead (for models without a hwhm prior).
    max_measurements : int, default NVISION_MAX_MEASUREMENTS
        Cap on the total measurements. The sweep repeats whole passes over the grid while they fit, so it makes
        ``max(1, max_measurements // grid points)`` passes (1 = a single pass, no averaging).
    noise_std : float, default 0.01
        Estimated measurement noise standard deviation.
    noise_max_dev : float | None, default None
        Pre-computed maximum noise deviation for thresholding.
    signal_min_span : float | None, default None
        Minimum expected signal span for density calculation.
    signal_max_span : float | None, default None
        Maximum expected signal span for window sizing.
    center_param : str | None, default None
        Parameter name being scanned.
    domain_lo : float, default 0.0
        Domain lower bound in physical units.
    domain_hi : float, default 1.0
        Domain upper bound in physical units.
    """

    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution | None = None,
        signal_model: SignalModel | None = None,
        max_steps: int = 0,
        *,
        builder: Callable[..., AbstractMarginalDistribution] | None = None,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        center_param: str | None = None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        spacing_divisor: int | None = SWEEP_POINTS_PER_FWHM,
        max_measurements: int = NVISION_MAX_MEASUREMENTS,
        **kwargs: Any,
    ) -> GenericSweepLocator:
        domain_lo = kwargs.pop("domain_lo", None)
        domain_hi = kwargs.pop("domain_hi", None)
        if builder is not None:
            # The belief is built exactly as the Bayesian locators build theirs (``kwargs`` are the
            # builder's own options), so every locator infers with the same belief and prior.
            if belief is not None or signal_model is not None:
                raise ValueError("GenericSweepLocator.create: pass either a builder or a belief + signal_model.")
            belief = builder(parameter_bounds, **kwargs)
            signal_model = belief.model
        elif belief is None or signal_model is None:
            raise ValueError("GenericSweepLocator.create needs either a builder or both belief and signal_model.")

        # The scan axis is the belief's own drive-frequency axis: observations are unit coordinates of it.
        belief_lo, belief_hi = belief.drive_freq_bounds_phys
        domain_lo = belief_lo if domain_lo is None else domain_lo
        domain_hi = belief_hi if domain_hi is None else domain_hi

        inst = cls(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            center_param=center_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            spacing_divisor=spacing_divisor,
            max_measurements=max_measurements,
        )
        if parameter_bounds is not None:
            inst._parameter_bounds = dict(parameter_bounds)
        return inst

    def __init__(
        self,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        center_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
        spacing_divisor: int | None = SWEEP_POINTS_PER_FWHM,
        max_measurements: int = NVISION_MAX_MEASUREMENTS,
    ):
        if max_measurements < 1:
            raise ValueError(f"max_measurements must be >= 1, got {max_measurements!r}")
        if spacing_divisor is not None and spacing_divisor < 1:
            raise ValueError(f"spacing_divisor must be >= 1 (or None), got {spacing_divisor!r}")
        if spacing_divisor is None and max_steps < 2:
            raise ValueError("A fixed-count sweep (spacing_divisor=None) needs max_steps >= 2.")
        super().__init__(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            center_param=center_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )

        # Fit results set by finalize() — reported via result().
        self._zero_field_freq_estimate_phys: float | None = None
        self._zero_field_freq_uncert_phys: float | None = None
        # Set by finalize() when the fit found no dip (SweepFitNoDipError): True, and the chi-square gain over flat.
        self._fit_failed: bool = False
        self._fit_chi2_gain: float | None = None

        # Full fitted physical parameter vector from the model fit (set by
        # _fit_model).  Used to draw the actual fit in the visualization
        # instead of the SMC belief marginal mode.
        self._fit_params_phys: dict[str, float] | None = None
        # Standard deviation of each *fitted* parameter (sqrt of the covariance diagonal; NaN when the fit
        # gives no usable covariance). Absent for a fixed zero_field_freq: it was not estimated.
        self._fit_std_phys: dict[str, float] | None = None
        self._center_fitted: bool = False

        # Parameter bounds stored by create() when provided as an argument.
        # Used by _fit_model() when signal_model has no param_bounds_phys.
        self._parameter_bounds: dict[str, tuple[float, float]] | None = None

        if not np.allclose(belief.drive_freq_bounds_phys, (domain_lo, domain_hi), rtol=1e-9, atol=0.0):
            raise ValueError(
                f"belief drive_freq_bounds_phys {belief.drive_freq_bounds_phys} do not match this locator's "
                f"domain ({domain_lo}, {domain_hi}); observations are unit coordinates of one range."
            )
        self.spacing_divisor = spacing_divisor
        if spacing_divisor is None:
            # Explicit fixed-count mode: exactly max_steps evenly spaced points, no width needed.
            self._fwhm_prior_phys = None
            n_points = max_steps
        else:
            # The grid follows from the prior FWHM: spacing = FWHM / spacing_divisor, endpoints included.
            self._fwhm_prior_phys = prior_fwhm_phys(belief)
            n_points = uniform_grid_phys(domain_lo, domain_hi, self._fwhm_prior_phys / spacing_divisor).size
            if max_steps > 0 and n_points > max_steps:
                raise ValueError(
                    f"GenericSweepLocator: a spacing of FWHM/{spacing_divisor} = "
                    f"{self._fwhm_prior_phys / spacing_divisor:g} Hz needs {n_points} points, over the budget of "
                    f"{max_steps} steps."
                )
        # ``max_steps`` stays the length of ONE pass over the grid; averaging repeats the pass.
        self.max_steps = n_points
        self.max_passes = max(1, max_measurements // n_points)
        self._init_stop_rules(NVISION_CONVERGENCE_THRESHOLD, NVISION_CONVERGENCE_PATIENCE)
        self._pending_obs: list[Observation] = []
        self.history = ObservationHistory(n_points * self.max_passes)
        self._sweep_points = self._generate_sweep_points(n_points)

    @property
    def passes_done(self) -> int:
        """Number of complete passes over the grid measured so far."""
        return self.step_count // self.max_steps if self.max_steps else 0

    def next(self) -> float:
        """Propose the next grid point; after the last one the sweep starts over (averaging mode)."""
        self.step_count += 1
        if self.step_count > self.max_steps * self.max_passes:
            return 0.5
        return float(self._sweep_points[(self.step_count - 1) % self.max_steps])

    def observe(self, obs: Observation) -> None:
        """Record the observation; absorb it into the belief in chunks and run Bayesian-SBED's stopping rules.

        The grid is fixed in advance, so the belief absorbs ``NVISION_SWEEP_BELIEF_CHUNK`` observations at once (one
        model evaluation, one resample check) instead of one by one. A chunk is also flushed at the end of every
        pass, so the belief is complete whenever :meth:`done` can stop the sweep. The stop rules run once per
        absorbed observation after the flush, so patience and the dip-scan cadence stay in measurement units
        (the belief is unchanged within a chunk, so the repeated checks agree).
        """
        if self.step_count <= self.max_steps * self.max_passes:
            self.history.append(obs)
            self.belief.last_obs = obs
            self._pending_obs.append(obs)
            if len(self._pending_obs) >= NVISION_SWEEP_BELIEF_CHUNK or self.step_count % self.max_steps == 0:
                self._flush_pending_obs()

    def _flush_pending_obs(self) -> None:
        """Absorb the buffered observations into the belief, then run the stop rules once per observation."""
        pending, self._pending_obs = self._pending_obs, []
        self.belief.batch_update(pending)
        for _ in pending:
            self._update_stop_rules()

    def done(self) -> bool:
        """Stop once the stopping rules fired -- at the end of a pass, so the average covers whole passes -- or
        after ``max_passes`` passes (the ``max_measurements`` cap)."""
        return self.step_count >= self.max_steps * self.max_passes or (
            self._is_converged and self.step_count % self.max_steps == 0
        )

    def _generate_sweep_points(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            return np.array([], dtype=float)
        return np.linspace(0.0, 1.0, n, dtype=float)

    def _expected_dip_count_from_model(self) -> int:
        """Return expected number of dips from the signal model.

        Delegates to the model's expected_dip_count() method, which knows its
        own structure (1, 2, 3, or 6 dips based on physics).
        """
        return self._inner_model().expected_dip_count()

    def _fit_model(self, drive_freqs_unit: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        """Fit the physical model to sweep data via least squares.

        Returns (zero_field_freq_phys, uncert_phys).  The fit uses all sweep points
        jointly, which gives sub-step accuracy even when dips are partially
        merged — much better than centroid heuristics.  Raises if the model
        can't be fit; a `GenericSweepLocator` sweep is only useful *because*
        of this fit, so a failure here is a real problem to surface, not a
        case to silently paper over with a cruder estimate.
        """
        inner = self._inner_model()
        param_names = inner.parameter_names()
        center_param = self._center_param
        # center_param (almost always "zero_field_freq") may be fixed rather than a free
        # fit parameter (e.g. NVCenterVoigtModel(with_fixed_zero_field_freq=True)) --
        # curve_fit then only fits the *other* parameters (hwhm, zeeman_shift,
        # contrast_total, ...); the scan axis's value comes from the model's fixed_values
        # instead of the fitted vector, with zero uncertainty (it wasn't estimated).
        center_idx: int | None
        fixed_center_value: float | None = None
        if center_param in param_names:
            center_idx = param_names.index(center_param)
        else:
            center_idx = None
            fixed_vals = getattr(inner.spec, "fixed_values", None) or {}
            if center_param not in fixed_vals:
                raise ValueError(
                    f"center_param {center_param!r} is neither one of the model's fitted "
                    f"parameters {param_names} nor a fixed value on its spec."
                )
            fixed_center_value = float(fixed_vals[center_param])
        # Contrast/amplitude parameter name: "contrast_total" for every lineshape now
        # (Lorentzian, plain Voigt, saturation-Voigt all share this convention).
        ct_idx = next((param_names.index(n) for n in ("contrast_total", "c_max") if n in param_names), None)
        zs_idx = param_names.index("zeeman_shift") if "zeeman_shift" in param_names else None
        hyperfine_shift_idx = param_names.index("hyperfine_shift") if "hyperfine_shift" in param_names else None

        param_bounds_phys = self._resolve_physical_bounds(param_names)
        lo_bounds = [param_bounds_phys[n][0] for n in param_names]
        hi_bounds = [param_bounds_phys[n][1] for n in param_names]
        phys_priors = self._resolve_physical_priors(param_names, center_param, param_bounds_phys)

        domain_lo, domain_hi = self._domain_lo, self._domain_hi
        domain_width = domain_hi - domain_lo
        drive_freqs_phys = domain_lo + drive_freqs_unit * domain_width
        n_pts = len(drive_freqs_unit)

        smoothed, window = self._smooth_for_peak_detection(ys)
        contrast = self._estimate_contrast(ys, smoothed, window)
        hwhm_est = self._estimate_hwhm(drive_freqs_unit, smoothed, contrast, domain_width)
        # Width parameter name: "hwhm" (Lorentzian) or "hwhm_lorentz"
        # (Voigt) — both HWHM-scale now, unlike the old kernel-native "fwhm_total"
        # (full-width) Voigt used to expose directly.
        width_idx = next((param_names.index(n) for n in ("hwhm", "hwhm_lorentz") if n in param_names), None)
        # Voigt's inhomogeneous (Gaussian) width has no data-driven seed and would
        # otherwise default to make_p0's generic bounds-midpoint fallback -- for the
        # typical (0, ~1.2 MHz) sigma_inhom range that's ~0.6 MHz of *assumed* Gaussian
        # width before the fit sees any data, frequently larger than the entire true
        # dip. curve_fit then satisfies the total observed width by driving
        # hwhm_lorentz to its floor and letting sigma_inhom absorb the rest --
        # a real local-minimum trap, not just an imprecise start. Seeding at the lineshape's
        # own lower bound (Lorentzian-first assumption) starts the fit's total width at
        # roughly hwhm_est instead of hwhm_est-plus-an-assumed-Gaussian-component, so the
        # optimizer only grows sigma_inhom if the data actually supports it.
        sigma_inhom_idx = param_names.index("sigma_inhom") if "sigma_inhom" in param_names else None
        seeds = self._seed_zero_field_freq_and_splits(
            drive_freqs_unit,
            ys,
            smoothed,
            contrast,
            domain_lo,
            domain_width,
            zs_idx,
            hyperfine_shift_idx,
            lo_bounds,
            hi_bounds,
        )

        if fixed_center_value is not None and zs_idx is not None:
            seeds = self._anchor_zeeman_seeds_to_fixed_center(seeds, fixed_center_value, domain_lo, domain_hi)

        # Parameters with no data-driven seed below (e.g. k_np) start at their belief-prior
        # mean rather than the bounds midpoint. The prior mean is deliberately offset from the
        # truth by the generator (see _widened_prior_mean), so this is not an oracle start.
        # Data-seeded parameters keep their data-driven seeds: starting sigma_inhom at a broad
        # default is the documented wrong-basin trap, so priors are NOT used for those.
        data_seeded_idx = {
            i for i in (center_idx, ct_idx, width_idx, sigma_inhom_idx, zs_idx, hyperfine_shift_idx) if i is not None
        }
        start_priors = self._resolve_physical_priors(
            param_names, center_param, param_bounds_phys, names=frozenset(param_names)
        )

        def make_p0(
            zero_field_freq_phys: float, half_sep_hz: float | None, hyperfine_shift_hz: float | None
        ) -> list[float]:
            p0 = [(lo_bounds[i] + hi_bounds[i]) / 2.0 for i in range(len(param_names))]
            for name, (prior_mean, _prior_std) in start_priors.items():
                i = param_names.index(name)
                if i not in data_seeded_idx:
                    p0[i] = float(np.clip(prior_mean, lo_bounds[i], hi_bounds[i]))
            if center_idx is not None:
                p0[center_idx] = float(np.clip(zero_field_freq_phys, domain_lo, domain_hi))
            if ct_idx is not None:
                p0[ct_idx] = float(np.clip(contrast, lo_bounds[ct_idx], hi_bounds[ct_idx]))
            if width_idx is not None and hwhm_est is not None and hwhm_est > 0:
                # Both remaining width keys are HWHM-scale now; kept as an explicit
                # membership check (rather than assuming) in case a future model adds
                # a full-width key back.
                width_val = hwhm_est if param_names[width_idx] in ("hwhm", "hwhm_lorentz") else 2.0 * hwhm_est
                p0[width_idx] = float(np.clip(width_val, lo_bounds[width_idx], hi_bounds[width_idx]))
            if sigma_inhom_idx is not None:
                # A tiny but nonzero fraction of hwhm_est, not the bare lower bound: at
                # sigma_inhom=0 exactly, _pv_factors' has_sigma gate makes the Gaussian
                # term's contribution (and curve_fit's numerical Jacobian column for this
                # parameter) identically zero, which can stall the optimizer at that
                # boundary instead of letting it explore away from it.
                si_seed = 0.05 * hwhm_est if hwhm_est is not None and hwhm_est > 0 else lo_bounds[sigma_inhom_idx]
                p0[sigma_inhom_idx] = float(np.clip(si_seed, lo_bounds[sigma_inhom_idx], hi_bounds[sigma_inhom_idx]))
            if zs_idx is not None and half_sep_hz is not None:
                p0[zs_idx] = float(np.clip(half_sep_hz, lo_bounds[zs_idx], hi_bounds[zs_idx]))
            if hyperfine_shift_idx is not None and hyperfine_shift_hz is not None:
                p0[hyperfine_shift_idx] = float(
                    np.clip(hyperfine_shift_hz, lo_bounds[hyperfine_shift_idx], hi_bounds[hyperfine_shift_idx])
                )
            return p0

        candidates = self._build_fit_candidates(
            drive_freqs_unit=drive_freqs_unit,
            ys=ys,
            domain_lo=domain_lo,
            domain_width=domain_width,
            contrast=contrast,
            seeds=seeds,
            zs_idx=zs_idx,
            hyperfine_shift_idx=hyperfine_shift_idx,
            lo_bounds=lo_bounds,
            hi_bounds=hi_bounds,
        )

        if hwhm_est is not None and hwhm_est > 0:
            lw_guess = hwhm_est
        elif width_idx is not None:
            mid = (lo_bounds[width_idx] + hi_bounds[width_idx]) / 2.0
            lw_guess = mid if param_names[width_idx] in ("hwhm", "hwhm_lorentz") else mid / 2.0
        else:
            lw_guess = domain_width * 0.01
        xtol = self._fit_xtol(contrast, n_pts, domain_width, lw_guess)

        # MAP regularization: append one synthetic "residual row" per prior'd parameter,
        # each pinning curve_fn's output for that row directly to the parameter's own
        # value (bypassing the physics model) so its residual reduces to
        # (param - prior_mean) / prior_std — a Tikhonov/MAP penalty scipy's curve_fit has
        # no native support for. Weighted jointly with the data residuals via `sigma` below
        # (see _resolve_physical_priors for why this only matters on degenerate directions).
        prior_names = list(phys_priors.keys())
        prior_param_idx = [param_names.index(n) for n in prior_names]
        n_prior_terms = len(prior_param_idx)

        if NVISION_SWEEP_FIT_FLOAT32:
            # Vectorized float32 path: one Numba call over all n_pts instead of a
            # per-point Python loop. Requires a widened curve_fit diff_step (see
            # _FIT_FLOAT32_DIFF_STEP above) so the Jacobian doesn't get quantized
            # away by float32's coarser resolution.
            from nvision.spectra.dtypes import FLOAT_DTYPE

            def curve_fn(xs: np.ndarray, *params: float) -> np.ndarray:
                samples = inner.spec.unpack_samples([np.array([v], dtype=FLOAT_DTYPE) for v in params])
                data_vals = np.asarray(inner.compute_vectorized_many(xs[:n_pts], samples), dtype=np.float64)[:, 0]
                if n_prior_terms:
                    prior_vals = np.array([params[i] for i in prior_param_idx])
                    return np.concatenate([data_vals, prior_vals])
                return data_vals
        else:
            # Evaluate in float64 (scalar path).  The vectorized kernel uses float32,
            # whose ~1e-7 relative resolution is coarser than curve_fit's numerical
            # Jacobian step for a GHz-scale zero_field_freq (~tens of Hz), which zeroes out
            # the zero_field_freq gradient and stalls the fit.  float64 is required here.
            def curve_fn(xs: np.ndarray, *params: float) -> np.ndarray:
                typed = inner.spec.unpack_params(list(params))
                data_vals = inner.compute_many_float64(xs[:n_pts], typed)
                if n_prior_terms:
                    prior_vals = np.array([params[i] for i in prior_param_idx])
                    return np.concatenate([data_vals, prior_vals])
                return data_vals

        if n_prior_terms:
            xs_fit = np.concatenate([drive_freqs_phys, np.zeros(n_prior_terms)])
            ys_fit = np.concatenate([ys, np.array([phys_priors[n][0] for n in prior_names])])
            sigma_fit = np.concatenate(
                [
                    np.full(n_pts, max(float(self._noise_std), 1e-12)),
                    np.array([phys_priors[n][1] for n in prior_names]),
                ]
            )
        else:
            xs_fit, ys_fit, sigma_fit = drive_freqs_phys, ys, None

        # Early stop: a start that reaches the noise floor (reduced chi-square within K sigmas of
        # 1) ends the race. Only at high peak SNR, where the smart seeds are trusted; at low SNR
        # the chi-square gap between a right and a wrong basin is within the noise band, so every
        # start is still raced (see _build_fit_candidates).
        early_stop_redchi2: float | None = None
        dof = len(ys_fit) - len(param_names)
        peak_snr = contrast / max(float(self._noise_std), 1e-9)
        if NVISION_SWEEP_FIT_EARLY_STOP_SIGMAS > 0 and dof > 0 and peak_snr >= _FIT_GRID_SNR:
            early_stop_redchi2 = 1.0 + NVISION_SWEEP_FIT_EARLY_STOP_SIGMAS * float(np.sqrt(2.0 / dof))

        def zeeman_profile_starts() -> list[list[float]]:
            return self._zeeman_profile_starts(
                curve_fn,
                xs_fit,
                ys,
                make_p0(fixed_center_value, None, None),
                zs_idx,
                width_idx,
                lo_bounds,
                hi_bounds,
                domain_width,
                max(float(self._noise_std), 1e-12),
            )

        best_popt, best_pcov = self._run_curve_fit_candidates(
            curve_fn,
            xs_fit,
            ys_fit,
            candidates,
            make_p0,
            lo_bounds,
            hi_bounds,
            xtol,
            sigma=sigma_fit,
            early_stop_redchi2=early_stop_redchi2,
            data_noise_std=max(float(self._noise_std), 1e-12),
            # With the centre fixed the dip pattern's position reaches the fit only through zeeman_shift.
            late_starts=zeeman_profile_starts if center_idx is None and zs_idx is not None else None,
        )

        self._require_fit_beats_flat_baseline(curve_fn(xs_fit, *best_popt)[:n_pts], ys, best_popt, param_names)

        zero_field_freq_phys = float(best_popt[center_idx]) if center_idx is not None else fixed_center_value
        if not (domain_lo <= zero_field_freq_phys <= domain_hi):
            raise RuntimeError(
                f"fitted zero_field_freq {zero_field_freq_phys} fell outside the domain [{domain_lo}, {domain_hi}]"
            )

        # Store the full fitted parameter vector so the visualization can draw the
        # actual fit instead of the (collapsed) SMC belief marginal mode. center_param
        # isn't part of best_popt when fixed, so add it back explicitly.
        self._fit_params_phys = {n: float(v) for n, v in zip(param_names, best_popt, strict=False)}
        diag = np.diag(best_pcov) if best_pcov is not None and np.all(np.isfinite(best_pcov)) else None
        self._fit_std_phys = {
            n: (float(np.sqrt(diag[i])) if diag is not None and diag[i] > 0 else float("nan"))
            for i, n in enumerate(param_names)
        }
        self._center_fitted = center_idx is not None
        if center_idx is None:
            self._fit_params_phys[center_param] = zero_field_freq_phys

        # center_idx is None => center_param was fixed, not fit -- no covariance to
        # report, and it wasn't estimated, so its uncertainty is exactly zero.
        uncert_phys = (
            self._report_fit_uncertainty(best_pcov, center_idx, contrast, n_pts, domain_width, lw_guess)
            if center_idx is not None
            else 0.0
        )
        return zero_field_freq_phys, uncert_phys

    def _require_fit_beats_flat_baseline(
        self, model_ys: np.ndarray, ys: np.ndarray, popt: np.ndarray, param_names: list[str]
    ) -> None:
        """Raise ``SweepFitNoDipError`` if the winning fit explains the data no better than a constant.

        ``model_ys`` is the fitted model at the sweep's points and ``ys`` the measured signal (both
        shape ``(n_pts,)``, signal units); the chi-square gain over a flat line at ``mean(ys)`` is measured in
        units of the measurement noise. A fit that did not reach the dip leaves the gain near zero, and
        reporting it would present its start point as the estimate.
        """
        noise = max(float(self._noise_std), 1e-12)
        chi2_fit = float(np.sum(((model_ys - ys) / noise) ** 2))
        chi2_flat = float(np.sum(((ys - float(np.mean(ys))) / noise) ** 2))
        if chi2_flat - chi2_fit < _FIT_MIN_CHI2_GAIN:
            fitted = dict(zip(param_names, (float(v) for v in popt), strict=True))
            raise SweepFitNoDipError(
                f"sweep fit is no better than a flat baseline (chi2 gain {chi2_flat - chi2_fit:.1f} < "
                f"{_FIT_MIN_CHI2_GAIN:g}; chi2 fit {chi2_fit:.1f}, flat {chi2_flat:.1f}): the fit did not reach a "
                f"dip, so its parameters {fitted} are not an estimate.",
                chi2_gain=chi2_flat - chi2_fit,
            )

    def _resolve_physical_bounds(self, param_names: list[str]) -> dict[str, tuple[float, float]]:
        """Resolve physical parameter bounds for the fit.

        Prefers ``signal_model.param_bounds_phys``, then the bounds stored by
        ``create()``, then per-parameter belief bounds. Raises if none are
        available, or if the resolved set is missing any model parameter.
        """
        unit_model = self.signal_model
        if hasattr(unit_model, "param_bounds_phys"):
            param_bounds_phys: dict[str, tuple[float, float]] = unit_model.param_bounds_phys
        elif self._parameter_bounds is not None:
            param_bounds_phys = self._parameter_bounds
        elif hasattr(self.belief, "physical_param_bounds"):
            param_bounds_phys = self.belief.physical_param_bounds
        else:
            raise RuntimeError(
                "No physical parameter bounds available for the model fit: "
                "signal_model.param_bounds_phys, create()-provided parameter_bounds, "
                "and belief.physical_param_bounds are all unavailable."
            )

        missing = [n for n in param_names if n not in param_bounds_phys]
        if missing:
            raise RuntimeError(f"physical_param_bounds is missing entries for: {missing}")
        # Snapshot: the SMC belief narrows signal_model.param_bounds_phys /
        # belief.physical_param_bounds in place during resampling — the fit
        # must not alias a dict that can shift under it.
        return dict(param_bounds_phys)

    # Parameters MAP-regularized in _fit_model, via _resolve_physical_priors below.
    # Deliberately narrow: hwhm_lorentz/sigma_inhom is a genuine degenerate
    # ridge (many combinations reproduce the same total width, so no amount of data
    # ever picks one over another — the prior can only ever act as a tie-breaker
    # among functionally-equivalent fits). Parameters that ARE identifiable from data
    # (zeeman_shift, hyperfine_shift, k_np, contrast_total) were tried here too and made things worse:
    # since curve_fit's multi-start races candidates by *total* residual including the
    # prior rows, a prior mean that's a noisy draw away from truth can make a
    # genuinely-worse-fitting candidate win the race outright, not just nudge the
    # final answer within one basin — a real regression (40-seed clean-signal median
    # |err| 52.9kHz -> 77.7kHz, mean 1915kHz -> 2568kHz) confirmed against a live run.
    _MAP_REGULARIZED_PARAMS = frozenset({"hwhm_lorentz", "sigma_inhom"})

    def _resolve_physical_priors(
        self,
        param_names: list[str],
        center_param: str,
        param_bounds_phys: dict[str, tuple[float, float]],
        names: frozenset[str] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Physical-unit Gaussian priors ``{name: (mean, std)}``.

        ``names`` restricts which parameters are returned; by default only
        ``_MAP_REGULARIZED_PARAMS`` (the MAP-penalty set). The fit's starting point passes
        every parameter name to read prior means for otherwise-unseeded parameters.

        These are the same priors SBED's belief is built from (the generator's
        ``bounds["_priors"]``, physical ``(mean, std)``). The SMC belief uses them to seed
        initial particles and rejuvenate during resampling — but neither mechanism helps
        ``curve_fit``, which has no Bayesian concept at all. Restricted to
        ``_MAP_REGULARIZED_PARAMS`` (see its comment) rather than every parameter with a
        prior — broader application actively hurt well-identified parameters.

        Source: the ``_priors`` entry the executor passes to the locator inside
        ``parameter_bounds`` (physical units). The sweep's own belief is a flat grid without
        priors, so reading ``self.belief.priors`` alone never found any in the real pipeline;
        that unit-space attribute is only a fallback for locators built with an SMC belief.

        ``center_param`` (zero_field_freq) is excluded even if a prior exists for it: localizing
        zero_field_freq from data is the entire point of the sweep, and its prior is a coarse
        "sin^2" shape rather than a Gaussian mean/std anyway.
        """
        allowed = self._MAP_REGULARIZED_PARAMS if names is None else names
        raw_phys = (self._parameter_bounds or {}).get("_priors")
        raw_belief = getattr(self.belief, "priors", None)
        if not raw_phys and not raw_belief:
            return {}
        phys_priors: dict[str, tuple[float, float]] = {}
        for name in param_names:
            if name == center_param or name not in allowed:
                continue
            if raw_phys and name in raw_phys:
                prior_val = raw_phys[name]
                to_phys = False
            elif raw_belief and name in raw_belief:
                prior_val = raw_belief[name]
                to_phys = True
            else:
                continue
            if not (isinstance(prior_val, tuple) and len(prior_val) >= 2) or prior_val[0] == "sin^2":
                continue
            mean, std = float(prior_val[0]), float(prior_val[1])
            if std <= 0:
                continue
            if to_phys:
                lo, hi = param_bounds_phys[name]
                mean, std = lo + mean * (hi - lo), std * (hi - lo)
            phys_priors[name] = (mean, std)
        return phys_priors

    def _smooth_for_peak_detection(self, ys: np.ndarray) -> tuple[np.ndarray, int]:
        """Smooth interior points only, to avoid edge artifacts from zero-padding.

        ``np.convolve(mode='same')`` zero-pads at boundaries, which artificially
        pulls edge values down and makes argmin pick the domain boundary instead
        of the actual dip.  The smoothed signal is only used for dip detection
        and contrast estimation, where edge effects don't matter because peaks
        are only detected in the interior.

        The window is capped at the number of samples spanning the model's
        minimum feature span — a *physical* scale.  The old ``n_pts // 30``
        window grew with sample count regardless of physics: a dense sweep over
        a wide domain produced a window many times wider than the dips
        themselves, smearing a whole hyperfine triplet into one blob so peak
        detection saw a single dip and the fit was seeded systematically wrong.

        Returns (smoothed, window).
        """
        n_pts = len(ys)
        window = max(3, n_pts // 30)
        min_span = self._signal_min_span or self._model_signal_min_span()
        domain_width = self._domain_hi - self._domain_lo
        if min_span is not None and min_span > 0 and domain_width > 0:
            samples_per_min_span = int(n_pts * min_span / domain_width)
            # Only cap if the theoretical minimum feature is actually resolvable
            # at this sampling density (>= a few samples across it). When min_span
            # is finer than the sweep's own per-sample resolution (e.g. a 140 kHz
            # worst-case minimum over 450 kHz/sample), samples_per_min_span
            # collapses to ~0 and the cap floors the window to 3 regardless of
            # this draw's *actual* dip width -- badly under-smoothing a dip that's
            # much wider than the theoretical minimum, which corrupts hwhm_est and
            # feeds find_peaks spurious noise-scale features instead of the real
            # dip edges (confirmed against a captured real sweep: a 4.6 MHz-wide
            # true dip produced hwhm_est=0.23 MHz and a zeeman_shift seed ~40% low
            # under the old unconditional cap). A cap that can't resolve anything
            # at this sampling density protects against nothing, so skip it.
            if samples_per_min_span >= 3:
                window = max(3, min(window, samples_per_min_span))
        smoothed = np.convolve(ys, np.ones(window) / window, mode="same") if window > 1 else ys
        return smoothed, window

    @staticmethod
    def _estimate_contrast(ys: np.ndarray, smoothed: np.ndarray, window: int) -> float:
        """Estimate the dip depth (baseline minus minimum) directly from the data.

        Seeds ``contrast_total`` close to the true value — a blind bound-midpoint
        initialization can be 6x off (e.g. 0.255 for a 0.04 dip), which makes
        convergence unreliable.  Uses interior smoothed points to reduce noise
        influence on the estimate.
        """
        n_pts = len(ys)
        half_win = window // 2
        interior = smoothed[half_win : n_pts - half_win] if n_pts > 2 * half_win else ys
        baseline = float(np.percentile(interior, 85))
        return max(0.0, baseline - float(interior.min()))

    @staticmethod
    def _estimate_hwhm(
        drive_freqs_unit: np.ndarray, smoothed: np.ndarray, contrast: float, domain_width: float
    ) -> float | None:
        """Estimate the deepest dip's half-width-at-half-max (HWHM) directly
        from the data, in the same units as this codebase's ``hwhm``
        parameter (see ``nv_center_lorentzian_eval``: ``omega`` in
        ``(x-freq)/omega`` is the HWHM scale, not the full width).

        Unlike `contrast_total`/contrast — whose value at the exact dip center is
        independent of width by construction (population-normalized
        reparametrization) — `hwhm` had no data-driven seed at all and
        always started at the bounds midpoint.  A bounds range can span 10-25x
        (e.g. 0.2-5 MHz), so a midpoint start can be far enough from the true
        width that curve_fit converges to a *compensating* (wrong width,
        wrong contrast) local minimum instead of the true one, even though
        zero_field_freq/splits/contrast were all seeded correctly.
        """
        if contrast <= 0:
            return None
        baseline = float(np.max(smoothed))
        half_level = baseline - 0.5 * contrast
        min_idx = int(np.argmin(smoothed))
        n = len(smoothed)
        left = min_idx
        while left > 0 and smoothed[left] < half_level:
            left -= 1
        right = min_idx
        while right < n - 1 and smoothed[right] < half_level:
            right += 1
        if right <= left:
            return None
        fwhm_norm = float(drive_freqs_unit[right] - drive_freqs_unit[left])
        return 0.5 * fwhm_norm * domain_width

    def _seed_zero_field_freq_and_splits(
        self,
        drive_freqs_unit: np.ndarray,
        ys: np.ndarray,
        smoothed: np.ndarray,
        contrast: float,
        domain_lo: float,
        domain_width: float,
        zs_idx: int | None,
        hyperfine_shift_idx: int | None,
        lo_bounds: list[float],
        hi_bounds: list[float],
    ) -> list[tuple[float, float | None, float | None]]:
        """Seed (zero_field_freq, zeeman half-separation, hyperfine shift) initial
        guesses for the fit, one tuple per plausible interpretation of the
        detected peaks.  The primary (most likely) seed comes first; all are
        raced by residual in ``_run_curve_fit_candidates``.

        This is the crucial step for multi-dip spectra: the raw argmin picks a
        *single* dip, but `zero_field_freq` is the *center* of the whole pattern —
        e.g. for a two-dip Zeeman spectrum that center sits ~zeeman_shift away
        from either dip, so an argmin init is systematically wrong and the fit
        diverges.

        The model can present 1, 2, 3, or 6 resolvable dips (see
        SignalModel.expected_dip_count): none, Zeeman-only, hyperfine-only, or
        both combined (2 Zeeman groups x 2 or 3 resolved hyperfine lines — see
        NVCenterLorentzianModel; an unresolved multiplet counts as 1 line).  We detect all peaks, keep
        the most prominent ones (bounded by how many the model expects, so
        stray noise peaks don't corrupt the grouping), then split them into
        Zeeman groups at the largest position gap — physically valid because
        zeeman_shift (up to 100 MHz) is normally much larger than the
        hyperfine shift (2-8.5 MHz) that separates lines *within* a group.

        For the hyperfine triplet (dips at freq-hyperfine_shift / freq / freq+hyperfine_shift with
        depths 1/k : 1 : k), the peak count is genuinely ambiguous — depths
        span k² so the shallow line often hides below the detection floor —
        and each count implies a *different* zero_field_freq:

        - 3 peaks: middle = freq, outer span = 2*hyperfine_shift (unambiguous).
        - 2 peaks: could be (center, deepest) → freq = left, hyperfine_shift = gap; or
          (outer pair) → freq = mid, hyperfine_shift = gap/2; or (shallow, center) →
          freq = right.  Seeding only one of these plants the fit in a wrong
          local minimum ~hyperfine_shift/2 away that gradient descent cannot escape (the
          basins are separated by many linewidths).
        - 1 peak: it is the *deepest* line, which for k>1 sits at freq+hyperfine_shift,
          not freq — so seed freq = argmin - hyperfine_shift over the hyperfine_shift grid, plus
          plain argmin for the k≈1 case.
        """
        from scipy.signal import find_peaks

        # Deepest raw point: use raw ys, not smoothed — np.convolve(mode='same')
        # zero-pads the boundaries and can drag the smoothed edge value below the
        # real dip, making argmin(smoothed) pick the domain edge.
        argmin_zero_field_freq = domain_lo + float(drive_freqs_unit[np.argmin(ys)]) * domain_width

        # Prominence floor must clear the (smoothed) noise so isolated fluctuations
        # aren't mistaken for dips.  Smoothing reduces noise by ~sqrt(window), but we
        # keep a conservative multiple of noise_std.
        prom_floor = max(0.3 * contrast, 4.0 * max(float(self._noise_std), 1e-9))
        peaks, props = find_peaks(-smoothed, prominence=prom_floor)
        n_expected_dips = self._expected_dip_count_from_model()
        if zs_idx is not None:
            # Never cap below the two Zeeman groups themselves. expected_dip_count()
            # now reports the model's real line structure (hyperfine lines x Zeeman
            # groups), so it can be trusted directly -- it used to be forced up to 6
            # here because every NV model secretly evaluated a fixed ¹⁴N triplet even
            # when configured without one, making the reported count an undercount.
            # An undercount is the dangerous direction: capping at 2 when the spectrum
            # really shows 6 keeps two sub-lines from the *same* Zeeman group instead
            # of one from each, seeding zeeman_shift tens of MHz off with no way for
            # curve_fit to recover.
            n_expected_dips = max(n_expected_dips, 2)

        if len(peaks) < 2:
            seeds: list[tuple[float, float | None, float | None]] = [(argmin_zero_field_freq, None, None)]
            if hyperfine_shift_idx is not None:
                # The lone detected dip is the deepest hyperfine line, at
                # freq+hyperfine_shift for k_np>1 — seed the center accordingly across
                # the hyperfine_shift grid (the true hyperfine_shift is unknown here).
                for h in np.linspace(
                    lo_bounds[hyperfine_shift_idx], hi_bounds[hyperfine_shift_idx], _FIT_ZEEMAN_STARTS
                ):
                    seeds.append((argmin_zero_field_freq - float(h), None, float(h)))
            if zs_idx is not None:
                # A single resolved peak is also the signature of near-merged
                # Zeeman dips (zeeman_shift too small to resolve at this
                # noise/resolution) — the blind bounds-midpoint start that
                # `half_sep=None` falls back to in make_p0 is then a bad
                # guess (e.g. ~50 MHz when the truth is <1 MHz) that can trap
                # the fit at an unrelated local minimum regardless of the true
                # small hyperfine_shift. Seed zeeman_shift≈0 explicitly alongside the
                # blind-midpoint version of every seed above.
                seeds = seeds + [(f, 0.0, h) for f, _, h in seeds]
            return seeds

        # Keep the most prominent peaks, capped at what the model expects
        # (at least 2), so stray noise peaks don't corrupt the grouping.
        keep = max(2, min(len(peaks), n_expected_dips))
        top_order = np.argsort(props["prominences"])[::-1][:keep]
        positions = np.sort(domain_lo + drive_freqs_unit[peaks[top_order]] * domain_width)

        if zs_idx is not None:
            # Split into the two Zeeman groups at the largest position gap —
            # used below only to resolve within-group hyperfine candidates.
            split_at = int(np.argmax(np.diff(positions)))
            group_lo = positions[: split_at + 1]
            group_hi = positions[split_at + 1 :]

            # zero_field_freq is the center of the *whole* two-dip Zeeman pattern:
            # each group sits at zero_field_freq +/- zeeman_shift, and any hyperfine
            # fine structure inside a group is symmetric about that group's
            # own center, so it never shifts where the pattern's outer edges
            # fall. The midpoint of the outermost detected peaks is therefore
            # a direct, physically-grounded estimate of zero_field_freq/zeeman_shift
            # — the same "generic doublet" logic used below when there's no
            # Zeeman splitting at all — and it doesn't depend on the
            # largest-gap grouping being correct.
            zero_field_freq_init = 0.5 * (float(positions[0]) + float(positions[-1]))
            half_sep = 0.5 * (float(positions[-1]) - float(positions[0]))

            if hyperfine_shift_idx is None:
                return [(zero_field_freq_init, half_sep, None)]

            # Each Zeeman group is itself a hyperfine triplet, subject to the
            # exact same ambiguity as the no-zeeman branch below: with only 2
            # of 3 lines resolved (the shallow k_np-suppressed line often
            # falls below the prominence floor), the within-group spread could
            # be the true hyperfine_shift (adjacent pair) or 2x the true hyperfine_shift (outer
            # pair) — picking only one seeds a wrong k_np/hyperfine_shift local minimum
            # gradient descent can't escape.  Offer every group's candidates;
            # zero_field_freq_init/half_sep stay fixed (from the outermost-peak midpoint
            # above) since the residual race only needs to sort out `hyperfine_shift`.
            hf_candidates: set[float] = set()
            for group in (group_lo, group_hi):
                if len(group) >= 3:
                    hf_candidates.add(round(float(group.max() - group.min()) / 2.0, 6))
                elif len(group) == 2:
                    gap = float(group[1] - group[0])
                    hf_candidates.add(round(gap, 6))
                    hf_candidates.add(round(0.5 * gap, 6))
            if not hf_candidates:
                return [(zero_field_freq_init, half_sep, None)]
            return [(zero_field_freq_init, half_sep, h) for h in sorted(hf_candidates)]

        if hyperfine_shift_idx is not None:
            if len(positions) >= 3:
                # Full triplet resolved: middle dip = zero_field_freq, outer span = 2*hyperfine_shift.
                return [
                    (float(positions[len(positions) // 2]), None, 0.5 * (float(positions[-1]) - float(positions[0])))
                ]
            # Two detected lines of a triplet — seed every consistent
            # assignment (see docstring); the residual race picks the winner.
            left, right = float(positions[0]), float(positions[1])
            gap = right - left
            return [
                (left, None, gap),  # (center, deepest): most likely for k_np > 1
                (0.5 * (left + right), None, 0.5 * gap),  # (outer pair, center hidden)
                (right, None, gap),  # (shallow, center)
            ]

        # Generic doublet: midpoint of the outermost pair.
        zero_field_freq_init = 0.5 * (float(positions[0]) + float(positions[-1]))
        half_sep = 0.5 * (float(positions[-1]) - float(positions[0]))
        return [(zero_field_freq_init, half_sep, None)]

    @staticmethod
    def _anchor_zeeman_seeds_to_fixed_center(
        seeds: list[tuple[float, float | None, float | None]],
        fixed_center_phys: float,
        domain_lo_phys: float,
        domain_hi_phys: float,
    ) -> list[tuple[float, float | None, float | None]]:
        """Give every seed a ``zeeman_shift`` measured from the fixed ``zero_field_freq``.

        Each seed's first entry is the detected centre of the dip pattern, but a fixed
        ``zero_field_freq`` is not a fit parameter, so ``make_p0`` discards it. The dip's position then only
        reaches the fit through ``zeeman_shift`` -- its distance from the fixed centre -- and a seed without one
        starts at the bounds midpoint, tens of linewidths from the dip, where the residual is flat and the fit
        never leaves. In the half-window domain ``[D, D + Delta]`` only the upper Zeeman group is visible, so
        the group centre is at ``D + zeeman_shift`` and the seed is exactly ``centre - D``.

        Seeds that already carry a ``zeeman_shift`` are kept when the domain straddles the fixed centre (both
        groups visible, so the outermost-peak half-separation is right); in a one-sided domain the
        outermost-peak midpoint is the lone group's centre, so there the offset replaces it. The explicit
        ``zeeman_shift = 0`` (merged-dips) seeds are kept.

        shape of ``seeds``: ``[(centre_phys, zeeman_shift_phys | None, hyperfine_shift_phys | None), ...]``.
        """
        one_sided = not (domain_lo_phys < fixed_center_phys < domain_hi_phys)
        anchored: list[tuple[float, float | None, float | None]] = []
        for centre, half_sep, hyperfine in seeds:
            if half_sep is None or (one_sided and half_sep != 0.0):
                half_sep = abs(centre - fixed_center_phys)
            anchored.append((centre, half_sep, hyperfine))
        return anchored

    def _build_fit_candidates(
        self,
        *,
        drive_freqs_unit: np.ndarray,
        ys: np.ndarray,
        domain_lo: float,
        domain_width: float,
        contrast: float,
        seeds: list[tuple[float, float | None, float | None]],
        zs_idx: int | None,
        hyperfine_shift_idx: int | None,
        lo_bounds: list[float],
        hi_bounds: list[float],
    ) -> list[tuple[float, float | None, float | None]]:
        """Build (freq, half_sep, hyperfine_shift) candidate initializations for curve_fit.

        All candidates are scored by residual once fit (lowest wins):
          1. dip-detection seeds — one per plausible peak interpretation
             (see ``_seed_zero_field_freq_and_splits``)
          2. deepest-point init as a cheap alternate
          3. (hyperfine shift unknown only) a small grid over hyperfine_shift — when both
             Zeeman and hyperfine splitting are present (the 6-dip case), the
             two Zeeman groups are usually resolved by peak detection but their
             internal hyperfine lines are often smoothed together, leaving the
             seeds without a hyperfine_shift value.  A single blind bounds-midpoint start
             for `hyperfine_shift` can then trap the fit in a wrong hyperfine_shift/k_np basin —
             this is missing information, not noise-corrupted detection, so
             it's added unconditionally (and kept small).
          4. (low SNR only) a coarse grid over zero_field_freq x zeeman_shift (or,
             for hyperfine-only models with no Zeeman splitting, x hyperfine_shift) — the
             safety net for low SNR, where dip detection is fooled by noise and
             a single start lands in a wrong local minimum.  Gated on SNR —
             cheap to compute *before* fitting — rather than on the fit
             residual, because at low SNR the residual gap between a right and
             a wrong fit is within the chi-square noise band and cannot tell
             them apart.  High-SNR spectra are solved by the smart starts in a
             few fits; only genuinely hard low-SNR spectra pay for the full
             multi-start grid.
        """
        zero_field_freq_init, half_sep, hyperfine_shift_init = seeds[0]
        argmin_zero_field_freq = domain_lo + float(drive_freqs_unit[np.argmin(ys)]) * domain_width
        candidates = list(seeds)
        candidates.append((argmin_zero_field_freq, half_sep, hyperfine_shift_init))

        if hyperfine_shift_idx is not None:
            # Unconditionally, not only when the seeds carry no hyperfine_shift: a detected
            # hyperfine_shift can itself be a wrong-basin reading. With a strongly asymmetric
            # triplet (k_np >~ 3, sub-dip depths 1/k² : 1/k : 1) each Zeeman group
            # reads as one dominant dip (its rightmost hyperfine line, at
            # group_center + hyperfine_shift) plus a faint shoulder — so dip detection seeds
            # the *zero_field_freq* offset by ~+hyperfine_shift while zeeman_shift stays correct,
            # and reads the shoulder spacing as a too-small hyperfine_shift. From that
            # correlated-wrong start curve_fit converges to a half-hyperfine_shift/
            # k_np-at-bound local optimum ~hyperfine_shift/2 off in zero_field_freq and never
            # visits the true basin. Pair each hyperfine_shift start h with the
            # dominant-line-corrected zero_field_freq (zero_field_freq_init - h) as well as the
            # uncorrected one; a few extra starts are cheap next to that failure
            # mode, and the residual race discards the wrong hypothesis.
            hf_grid0 = np.linspace(lo_bounds[hyperfine_shift_idx], hi_bounds[hyperfine_shift_idx], _FIT_ZEEMAN_STARTS)
            candidates += [(zero_field_freq_init, half_sep, float(h)) for h in hf_grid0]
            candidates += [(zero_field_freq_init - float(h), half_sep, float(h)) for h in hf_grid0]

        snr = contrast / max(float(self._noise_std), 1e-9)
        if snr < _FIT_GRID_SNR:
            zero_field_freq_grid = [
                domain_lo + (i + 0.5) / _FIT_ZERO_FIELD_FREQ_STARTS * domain_width
                for i in range(_FIT_ZERO_FIELD_FREQ_STARTS)
            ]
            if zs_idx is not None:
                zee_grid = np.linspace(lo_bounds[zs_idx], hi_bounds[zs_idx], _FIT_ZEEMAN_STARTS)
                candidates += [(f, float(z), hyperfine_shift_init) for f in zero_field_freq_grid for z in zee_grid]
            elif hyperfine_shift_idx is not None:
                hf_grid = np.linspace(
                    lo_bounds[hyperfine_shift_idx], hi_bounds[hyperfine_shift_idx], _FIT_ZEEMAN_STARTS
                )
                candidates += [(f, None, float(h)) for f in zero_field_freq_grid for h in hf_grid]
            else:
                candidates += [(f, None, None) for f in zero_field_freq_grid]

        return candidates

    @staticmethod
    def _zeeman_profile_starts(
        curve_fn: Callable[..., np.ndarray],
        xs_fit: np.ndarray,
        ys: np.ndarray,
        base_p0: list[float],
        zs_idx: int,
        width_idx: int | None,
        lo_bounds: list[float],
        hi_bounds: list[float],
        domain_width: float,
        noise_std: float,
    ) -> list[list[float]]:
        """Start vectors at the best minima of a chi-square profile over ``zeeman_shift`` (fixed centre).

        With ``zero_field_freq`` fixed, ``zeeman_shift`` is the only parameter that moves the dips along the
        scan axis, and chi-square is multimodal in it on a scale of one linewidth: a start more than about a
        linewidth from the dip sees a flat residual and stalls. When the dip sits at the domain edge, the
        smart seeds all say ``zeeman_shift = 0``, the bound, which stalls the same way.

        The profile evaluates ``curve_fn`` on a ``zeeman_shift`` grid with step half the smallest width tried
        (capped at ``_FIT_PROFILE_MAX_POINTS`` points over ``[lo, min(hi, domain_width)]``), at each width in
        ``_FIT_PROFILE_WIDTH_FRACTIONS`` x the width in ``base_p0``, every other parameter at ``base_p0``. The
        ``_FIT_PROFILE_STARTS`` lowest local minima of the per-``zeeman_shift`` best chi-square become starts.

        shape of ``xs_fit``: ``(n_rows,)`` physical drive_freq, ``n_rows >= len(ys)`` (prior rows after the data);
        shape of ``ys``: ``(n_pts,)`` measured signal; shape of ``base_p0``: ``(n_params,)`` physical units;
        returns ``[start_vector (n_params,), ...]``, empty when the ``zeeman_shift`` range has no extent.
        """
        n_pts = len(ys)
        if width_idx is None:
            widths: list[float | None] = [None]
            min_width = 0.0
        else:
            widths = sorted(
                {
                    float(np.clip(base_p0[width_idx] * f, lo_bounds[width_idx], hi_bounds[width_idx]))
                    for f in _FIT_PROFILE_WIDTH_FRACTIONS
                }
            )
            min_width = min(w for w in widths if w is not None)
        span = min(hi_bounds[zs_idx], domain_width) - lo_bounds[zs_idx]
        if span <= 0:
            return []
        step = max(0.5 * min_width, span / _FIT_PROFILE_MAX_POINTS)
        # Half a step in from the lower bound, so no start sits exactly on it (zero gradient there).
        zs_grid = lo_bounds[zs_idx] + step * (np.arange(int(np.ceil(span / step))) + 0.5)

        chi2 = np.empty((len(widths), len(zs_grid)))  # shape: (n_widths, n_zeeman_shift)
        for i, width in enumerate(widths):
            for j, zs in enumerate(zs_grid):
                p = list(base_p0)
                p[zs_idx] = float(zs)
                if width_idx is not None:
                    p[width_idx] = width
                chi2[i, j] = float(np.sum(((curve_fn(xs_fit, *p)[:n_pts] - ys) / noise_std) ** 2))
        best_chi2 = chi2.min(axis=0)  # shape: (n_zeeman_shift,)
        best_width = chi2.argmin(axis=0)  # shape: (n_zeeman_shift,) index into widths

        is_minimum = np.ones(len(zs_grid), dtype=bool)
        is_minimum[1:] &= best_chi2[1:] <= best_chi2[:-1]
        is_minimum[:-1] &= best_chi2[:-1] <= best_chi2[1:]
        minima = sorted(np.flatnonzero(is_minimum), key=lambda j: best_chi2[j])[:_FIT_PROFILE_STARTS]

        starts: list[list[float]] = []
        for j in minima:
            p = list(base_p0)
            p[zs_idx] = float(zs_grid[j])
            if width_idx is not None:
                p[width_idx] = widths[best_width[j]]
            starts.append(p)
        return starts

    def _fit_xtol(self, contrast: float, n_pts: int, domain_width: float, lw_guess: float) -> float:
        """Convergence tolerance (curve_fit xtol) derived from the expected
        zero_field_freq CRLB: refine until the zero_field_freq is pinned to
        ``_FIT_CRLB_TOL_FRAC`` of its Cramér-Rao floor.  Refining tighter than
        the CRLB is meaningless (it is below the information the data
        carries) and only burns iterations; refining looser leaves accuracy on
        the table.

        Using a rough pre-fit hwhm (bounds midpoint) and the data-driven
        contrast (contrast) is enough — the CRLB only sets the *scale* of the
        tolerance, not the estimate.  The optimizer runs with per-parameter
        ``x_scale`` = bounds width (see ``_run_curve_fit_candidates``), so
        steps are measured in bounds-width units: a zero_field_freq target of
        (CRLB × frac) Hz corresponds to a scaled tolerance of
        (CRLB × frac) / domain_width.
        """
        crlb_f = lorentzian_zero_field_freq_crlb(lw_guess, contrast, float(self._noise_std), n_pts, domain_width)
        if np.isfinite(crlb_f) and crlb_f > 0 and domain_width > 0:
            return float(np.clip(_FIT_CRLB_TOL_FRAC * crlb_f / domain_width, _FIT_XTOL_MIN, _FIT_XTOL_MAX))
        return 1e-8  # curve_fit default when the CRLB is unavailable

    @staticmethod
    def _run_curve_fit_candidates(
        curve_fn,
        drive_freqs_phys: np.ndarray,
        ys: np.ndarray,
        candidates: list[tuple[float, float | None, float | None]],
        make_p0,
        lo_bounds: list[float],
        hi_bounds: list[float],
        xtol: float,
        sigma: np.ndarray | None = None,
        early_stop_redchi2: float | None = None,
        data_noise_std: float = 1.0,
        late_starts: Callable[[], list[list[float]]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run curve_fit from each candidate start, keep the lowest-residual result.

        ``late_starts``, when given, returns extra full start vectors (``shape: (n_params,)`` each, physical
        units) that are raced after the candidates, and only when no candidate reached the noise floor
        (``early_stop_redchi2``), so a spectrum the smart seeds already solve never pays for them.

        ``early_stop_redchi2``, when set, ends the race at the first start whose reduced
        chi-square (chi-square / (n_rows - n_params), weighted by ``sigma`` or, when that is
        ``None``, by ``data_noise_std``) is at or below it: that fit is already at the noise
        floor, so the remaining starts cannot meaningfully improve it.

        ``x_scale`` = bounds width per parameter is essential: the raw
        parameter vector mixes a GHz-scale zero_field_freq with O(1) shape
        parameters (k_np, contrast_total), a ~1e9 conditioning spread.  Without
        scaling, the trust-region steps and the xtol termination test are
        dominated by the drive-frequency axis, so the optimizer stops while the
        *shape* parameters are still far from their optimum — zero_field_freq comes
        out fine but the fitted curve's depth/width visibly deviates from the
        data.

        ``sigma``, when given, carries the per-residual weight (data points at
        the measurement noise std, any appended MAP prior rows at their prior
        std — see ``_fit_model``) with ``absolute_sigma=True`` so the two
        differently-scaled residual types combine correctly instead of scipy's
        default reduced-chi-square rescale conflating them. Candidate racing
        uses the same weighting so "lowest residual" means the same thing the
        optimizer itself minimized.

        Raises if every candidate fails to converge.
        """
        from scipy.optimize import curve_fit

        x_scale = np.maximum(np.asarray(hi_bounds, dtype=float) - np.asarray(lo_bounds, dtype=float), 1e-12)
        sigma_kwargs = {} if sigma is None else {"sigma": sigma, "absolute_sigma": True}
        diff_step_kwargs = {"diff_step": _FIT_FLOAT32_DIFF_STEP} if NVISION_SWEEP_FIT_FLOAT32 else {}

        best_popt = None
        best_pcov = None
        best_resid = np.inf
        seen_p0: set[tuple[float, ...]] = set()

        def race_start(p0: list[float]) -> bool:
            """Fit from ``p0``, keep it if it is the best so far; True when it reached the noise floor."""
            nonlocal best_popt, best_pcov, best_resid
            # Starts that are the same vector give the identical fit (curve_fit is deterministic) -- e.g. a
            # drive-frequency grid when zero_field_freq is fixed, so make_p0 ignores it. Skipping repeats
            # cannot change the winner, only the run time.
            p0_key = tuple(p0)
            if p0_key in seen_p0:
                return False
            seen_p0.add(p0_key)
            try:
                popt, pcov = curve_fit(
                    curve_fn,
                    drive_freqs_phys,
                    ys,
                    p0=p0,
                    bounds=(lo_bounds, hi_bounds),
                    maxfev=_FIT_MAXFEV,
                    xtol=xtol,
                    x_scale=x_scale,
                    **sigma_kwargs,
                    **diff_step_kwargs,
                )
            except Exception:
                return False
            resid_vec = curve_fn(drive_freqs_phys, *popt) - ys
            if sigma is not None:
                resid_vec = resid_vec / sigma
            resid = float(np.sum(resid_vec**2))
            if resid < best_resid:
                best_resid = resid
                best_popt = popt
                best_pcov = pcov
            if early_stop_redchi2 is None:
                return False
            weights = sigma if sigma is not None else data_noise_std
            chi2 = float(np.sum(((curve_fn(drive_freqs_phys, *popt) - ys) / weights) ** 2))
            return chi2 / (len(ys) - len(popt)) <= early_stop_redchi2

        reached_floor = False
        for zero_field_freq_c, half_c, hf_c in candidates:
            if race_start(make_p0(zero_field_freq_c, half_c, hf_c)):
                reached_floor = True
                break
        if not reached_floor and late_starts is not None:
            # No smart start reached the noise floor: only now pay for the rescue starts.
            for p0 in late_starts():
                if race_start(p0):
                    break

        if best_popt is None:
            raise RuntimeError(f"curve_fit failed to converge from any of {len(candidates)} candidate starts")
        return best_popt, best_pcov

    def _report_fit_uncertainty(
        self,
        best_pcov: np.ndarray | None,
        center_idx: int,
        contrast: float,
        n_pts: int,
        domain_width: float,
        lw_guess: float,
    ) -> float:
        """Report uncertainty from the fit covariance, floored at the CRLB
        evaluated at the fitted parameters (the fit cannot beat the
        information floor).
        """
        fit_std = float("inf")
        if best_pcov is not None and np.all(np.isfinite(best_pcov)):
            var = float(best_pcov[center_idx, center_idx])
            if var > 0:
                fit_std = float(np.sqrt(var))
        contrast_fit = next(
            (self._fit_params_phys[n] for n in ("contrast_total", "c_max") if n in self._fit_params_phys),
            contrast,
        )
        crlb_at_fit = lorentzian_zero_field_freq_crlb(
            self._fit_params_phys.get("hwhm", lw_guess),
            contrast_fit,
            float(self._noise_std),
            n_pts,
            domain_width,
        )
        uncert_phys = max(fit_std, crlb_at_fit) if np.isfinite(crlb_at_fit) else fit_std
        if not np.isfinite(uncert_phys):
            uncert_phys = domain_width / max(1, n_pts - 1) * 0.5  # fallback: half a sweep step
        return uncert_phys

    def finalize(self) -> None:
        """Fit the physical model to the sweep and report the ``zero_field_freq``.

        The fit uses only the recorded measurements, never the belief (which was updated step by step
        and is what the stage metrics read); it supplies the reported estimate.

        A fit that finds no dip (``SweepFitNoDipError``) is recorded, not raised: ``fit_failed`` is set and no
        estimate is reported. Any other fit failure raises — see ``_fit_model``.

        No-ops if no measurements were taken (e.g. the run was skipped
        outright by the pre-run CRLB feasibility gate in ``executor.py``,
        which still calls ``finalize()`` on a freshly-created, unobserved
        locator instance).
        """
        if self.history.count == 0:
            return
        domain_width = self._domain_hi - self._domain_lo
        passes = self.passes_done
        try:
            if passes <= 1:
                zero_field_freq_phys, uncert_phys = self._fit_model(self.history.drive_freqs_unit, self.history.ys)
            else:
                # Fit the per-grid-point mean over the passes (the fit helpers expect one ordered pass), whose
                # noise is noise_std / sqrt(passes). Only whole passes exist: done() stops on a pass boundary.
                n = self.max_steps
                assert self.history.count == n * passes, "averaged sweep must hold whole passes"
                grid_unit = self.history.drive_freqs_unit[:n]
                mean_ys = self.history.ys.reshape(passes, n).mean(axis=0)
                single_pass_noise_std = self._noise_std
                self._noise_std = single_pass_noise_std / np.sqrt(passes)
                try:
                    zero_field_freq_phys, uncert_phys = self._fit_model(grid_unit, mean_ys)
                finally:
                    self._noise_std = single_pass_noise_std
        except SweepFitNoDipError as exc:
            # A measured outcome (no estimate, window stays the whole domain), not a crash of every repeat.
            self._fit_failed = True
            self._fit_chi2_gain = exc.chi2_gain
            return

        self._zero_field_freq_estimate_phys = zero_field_freq_phys
        self._zero_field_freq_uncert_phys = uncert_phys
        self._signal_found = True
        signal_max_span = self._signal_max_span or self._model_signal_max_span()
        half_width_phys = (
            min(domain_width / 2, 1.5 * signal_max_span) if signal_max_span is not None else domain_width * 0.2
        )
        self._acquisition_lo = max(self._domain_lo, zero_field_freq_phys - half_width_phys)
        self._acquisition_hi = min(self._domain_hi, zero_field_freq_phys + half_width_phys)

    def result(self) -> dict[str, float]:
        """Return dip-center estimate, uncertainty, and fit-derived sweep metrics."""
        res = super().result()
        # zero_field_freq is only an estimate when the fit estimated it; a fixed (known) center is not a result,
        # and reporting it would score the sweep on a parameter nobody measured (error and uncertainty 0).
        if self._center_fitted and self._zero_field_freq_estimate_phys is not None:
            res["zero_field_freq"] = self._zero_field_freq_estimate_phys
        if self._center_fitted and self._zero_field_freq_uncert_phys is not None:
            res["uncert"] = self._zero_field_freq_uncert_phys
        res.update(self._compute_sweep_metrics())
        # fit_failed: the sweep found no dip, so it reports no estimate (and fit_chi2_gain says how little it gained).
        res["fit_failed"] = self._fit_failed
        res["fit_chi2_gain"] = self._fit_chi2_gain if self._fit_failed else float("nan")
        return res

    def _compute_sweep_metrics(self) -> dict[str, float | int]:
        """Sweep efficiency metrics derived from the model fit.

        ``finalize()`` always produces a full least-squares fit (or raises),
        so the dip count and width are known exactly from the model and the
        fit — no need to re-detect dips from noisy sweep y-values.
        """
        domain_width = self._domain_hi - self._domain_lo
        expected_dips = self._expected_dip_count_from_model()

        dip_width = None
        if self._fit_params_phys is not None:
            dip_width = self._fit_params_phys.get("hwhm") or self._fit_params_phys.get("hwhm_lorentz")
        dip_width = dip_width or self._model_signal_min_span()

        total_dip_width = expected_dips * dip_width if dip_width is not None and dip_width > 0 else 0.0
        if total_dip_width > 0 and domain_width > 0:
            expected_uniform = 2.0 * domain_width / total_dip_width
        else:
            expected_uniform = float(self.max_steps)
        measurements_done = min(round(expected_uniform), self.max_steps)

        return {
            "measurements_done": measurements_done,
            "dips_detected": expected_dips,
            "total_dip_width": total_dip_width,
            "min_dip_width": dip_width or 0.0,
            "expected_uniform_points": expected_uniform,
            "sweep_efficiency": expected_uniform / max(measurements_done, 1),
        }

    def fit_mode_estimates(self) -> dict[str, float] | None:
        """Full fitted physical parameters from the model fit, or None if unavailable.

        This is the actual least-squares fit of the physical model to the sweep
        data — used by the visualization in preference to the SMC belief marginal
        mode, which is not resampled during the sweep and can collapse.
        """
        return dict(self._fit_params_phys) if self._fit_params_phys is not None else None

    def fitted_estimates(self) -> dict[str, float] | None:
        """The parameters the fit estimated (a fixed ``zero_field_freq`` is left out), or None before ``finalize()``."""
        if self._fit_params_phys is None:
            return None
        return {n: v for n, v in self._fit_params_phys.items() if n != self._center_param or self._center_fitted}

    def fit_mode_uncertainties(self) -> dict[str, float] | None:
        """Fit standard deviation of each fitted parameter (physical units; NaN = no covariance), or None."""
        return dict(self._fit_std_phys) if self._fit_std_phys is not None else None

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        return None
