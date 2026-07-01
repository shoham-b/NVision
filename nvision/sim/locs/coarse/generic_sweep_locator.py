"""Generic uniform sweep locator — a concrete SweepingLocator implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from nvision.belief.abstract_marginal import AbstractMarginalDistribution
from nvision.sim.locs.coarse.sweep_locator import SweepingLocator
from nvision.spectra.signal import SignalModel

if TYPE_CHECKING:
    pass

# Deferred observations are flushed to belief.batch_update() in chunks of this
# size at finalize().  batch_update evaluates the model once per chunk via the
# vectorized _many kernel instead of once per observation, while still letting
# the filter resample between chunks (same rationale as SOBOL_BATCH_CHUNK_SIZE).
NVISION_SWEEP_BATCH_CHUNK_SIZE: int = int(os.getenv("NVISION_SWEEP_BATCH_CHUNK_SIZE", "200"))


class GenericSweepLocator(SweepingLocator):
    """Uniform grid sweep locator with parabolic (r²) fit.

    Sweeps the full domain uniformly, then fits a quadratic to the region
    around the minimum to estimate the dip center. No refocusing or early
    stopping — every allocated step is used.

    Parameters
    ----------
    belief : AbstractMarginalDistribution
        Belief distribution (required by Locator parent class).
    signal_model : SignalModel
        Signal model for sweep detection.
    max_steps : int
        Number of sweep steps (use all of them).
    noise_std : float, default 0.01
        Estimated measurement noise standard deviation.
    noise_max_dev : float | None, default None
        Pre-computed maximum noise deviation for thresholding.
    signal_min_span : float | None, default None
        Minimum expected signal span for density calculation.
    signal_max_span : float | None, default None
        Maximum expected signal span for window sizing.
    scan_param : str | None, default None
        Parameter name being scanned.
    domain_lo : float, default 0.0
        Domain lower bound in physical units.
    domain_hi : float, default 1.0
        Domain upper bound in physical units.
    """

    @classmethod
    def create(
        cls,
        belief: AbstractMarginalDistribution,
        signal_model: SignalModel,
        max_steps: int,
        *,
        noise_std: float = 0.01,
        noise_max_dev: float | None = None,
        signal_min_span: float | None = None,
        signal_max_span: float | None = None,
        scan_param: str | None = None,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> GenericSweepLocator:
        domain_lo = kwargs.get("domain_lo")
        domain_hi = kwargs.get("domain_hi")
        
        if domain_lo is None or domain_hi is None:
            if parameter_bounds is not None:
                param_name = scan_param or (
                    signal_model.parameter_names()[0] if signal_model.parameter_names() else "peak_x"
                )
                if param_name in parameter_bounds:
                    if domain_lo is None:
                        domain_lo = parameter_bounds[param_name][0]
                    if domain_hi is None:
                        domain_hi = parameter_bounds[param_name][1]
        
        domain_lo = domain_lo if domain_lo is not None else 0.0
        domain_hi = domain_hi if domain_hi is not None else 1.0

        inst = cls(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
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
        scan_param: str | None = None,
        domain_lo: float = 0.0,
        domain_hi: float = 1.0,
    ):
        super().__init__(
            belief=belief,
            signal_model=signal_model,
            max_steps=max_steps,
            noise_std=noise_std,
            noise_max_dev=noise_max_dev,
            signal_min_span=signal_min_span,
            signal_max_span=signal_max_span,
            scan_param=scan_param,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
        )

        # No refocusing — sweep the entire domain.
        self._refocus_at = None

        # Fit results set by finalize() — reported via result().
        self._freq_estimate_phys: float | None = None
        self._freq_uncert_phys: float | None = None

        # Full fitted physical parameter vector from the model fit (set by
        # _try_model_fit).  Used to draw the actual fit in the visualization
        # instead of the SMC belief marginal mode.
        self._fit_params_phys: dict[str, float] | None = None

        # Buffer observations so we can batch-update the belief once at finalize()
        # instead of paying a full Bayesian update on every step.
        self._pending_obs: list = []

        # Parameter bounds stored by create() when provided as an argument.
        # Used by _try_model_fit() when signal_model has no param_bounds_phys.
        self._parameter_bounds: dict[str, tuple[float, float]] | None = None

        # Generate sweep points across the full domain.
        self._sweep_points = self._generate_sweep_points(max_steps)

    def _generate_sweep_points(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            return np.array([], dtype=float)
        return np.linspace(0.0, 1.0, n, dtype=float)

    def _generate_fallback_points(self, n: int) -> NDArray[np.float64]:
        if n <= 0:
            return np.array([], dtype=float)
        return (np.linspace(0.0, 1.0, n, dtype=float) + 0.5 / n) % 1.0

    def observe(self, obs) -> None:
        """Record observation without updating the belief on every step.

        The belief is only used for visualization, not for choosing the next
        point (all sweep positions are predetermined). We buffer all observations
        and do a single belief update in finalize() to avoid paying the full
        Bayesian posterior cost N times.
        """
        if self.step_count <= self.max_steps:
            self.history.append(obs)
            self.belief.last_obs = obs
            self._pending_obs.append(obs)

    def _should_refocus(self, step_count: int) -> int | None:
        return None

    def _regenerate_points(self, refocus_step: int, lo_norm: float, hi_norm: float) -> None:
        pass

    def _check_early_stop(self) -> bool:
        return False

    def _flush_pending_obs(self) -> None:
        """Flush deferred belief updates in vectorized chunks.

        One batched posterior update per chunk instead of a full
        per-observation update N times (the belief is not used for
        acquisition decisions during the sweep).
        """
        if not self._pending_obs:
            return
            
        mapped_obs = []
        is_unit_cube = type(self.belief.model).__name__ == "UnitCubeSignalModel"
        
        if is_unit_cube:
            mapped_obs = self._pending_obs
        else:
            from nvision.models.observation import Observation
            width = self._domain_hi - self._domain_lo
            for o in self._pending_obs:
                x_phys = self._domain_lo + o.x * width
                mapped_obs.append(Observation(
                    x=x_phys,
                    signal_value=o.signal_value,
                    noise_std=o.noise_std,
                    frequency_noise_model=o.frequency_noise_model,
                ))

        if hasattr(self.belief, "batch_update"):
            chunk = NVISION_SWEEP_BATCH_CHUNK_SIZE
            for i in range(0, len(mapped_obs), chunk):
                self.belief.batch_update(mapped_obs[i : i + chunk])
        else:
            for obs in mapped_obs:
                self.belief.update(obs)
        self._pending_obs.clear()

    def _try_model_fit(
        self, xs_norm: np.ndarray, ys: np.ndarray
    ) -> tuple[float, float] | None:
        """Fit the physical model to sweep data via least squares.

        Returns (freq_phys, uncert_phys) on success, None if unavailable or failed.
        The fit uses all sweep points jointly, which gives sub-step accuracy even
        when dips are partially merged — much better than centroid heuristics.
        """
        from scipy.optimize import curve_fit

        inner = self._inner_model()
        param_names = inner.parameter_names()
        scan_param = self._scan_param
        if scan_param not in param_names:
            return None

        # Resolve parameter bounds: prefer signal_model.param_bounds_phys, then
        # the bounds stored from create(), then per-parameter belief bounds.
        unit_model = self.signal_model
        if hasattr(unit_model, "param_bounds_phys"):
            param_bounds_phys: dict[str, tuple[float, float]] = unit_model.param_bounds_phys
        elif self._parameter_bounds is not None:
            param_bounds_phys = self._parameter_bounds
        elif hasattr(self.belief, "physical_param_bounds"):
            param_bounds_phys = self.belief.physical_param_bounds
        else:
            return None

        if not all(n in param_bounds_phys for n in param_names):
            return None

        domain_lo = self._domain_lo
        domain_hi = self._domain_hi
        domain_width = domain_hi - domain_lo
        xs_phys = domain_lo + xs_norm * domain_width

        scan_idx = param_names.index(scan_param)
        lo_bounds = [param_bounds_phys[n][0] for n in param_names]
        hi_bounds = [param_bounds_phys[n][1] for n in param_names]

        n_pts = len(xs_norm)

        # Smooth interior points only (avoid edge artifacts from zero-padding).
        # np.convolve mode='same' zero-pads at boundaries, which artificially pulls
        # edge values down and makes argmin pick the domain boundary instead of the
        # actual dip.  The smoothed signal is used for dip detection and c_total
        # estimation; edge effects there don't matter because we detect peaks in the
        # interior.
        window = max(3, n_pts // 30)
        smoothed = np.convolve(ys, np.ones(window) / window, mode="same") if window > 1 else ys

        # Estimate contrast directly from the data so that c_total starts close to
        # the true value.  Blind bound-midpoint initialization can be 6× off (e.g.
        # 0.255 for a 0.04 dip), which makes convergence unreliable.
        # Use interior smoothed points to reduce noise influence on c_total estimate.
        half_win = window // 2
        interior = smoothed[half_win : n_pts - half_win] if n_pts > 2 * half_win else ys
        baseline = float(np.percentile(interior, 85))
        dip_depth = max(0.0, baseline - float(interior.min()))

        # Detect the main dips to initialize `frequency` and (for Zeeman models)
        # `zeeman_split`.  This is the crucial step for multi-dip spectra: the raw
        # argmin picks a *single* dip, but `frequency` is the *center* of the whole
        # pattern.  For a two-dip Zeeman spectrum that center sits ~zeeman_split away
        # from either dip, so an argmin init is systematically wrong and the fit
        # diverges.  Taking the midpoint of the outermost detected dips gives the
        # center for single, triplet, and Zeeman patterns alike.
        from scipy.signal import find_peaks

        # Prominence floor must clear the (smoothed) noise so isolated fluctuations
        # aren't mistaken for dips.  Smoothing reduces noise by ~sqrt(window), but we
        # keep a conservative multiple of noise_std.
        prom_floor = max(0.3 * dip_depth, 4.0 * max(float(self._noise_std), 1e-9))
        peaks, props = find_peaks(-smoothed, prominence=prom_floor)
        if len(peaks) >= 2:
            # Use the two most prominent dips (the main pair for a Zeeman spectrum),
            # not the outermost — spurious edge peaks from noise would otherwise
            # inflate the separation.
            order = np.argsort(props["prominences"])[::-1]
            top = peaks[order[:2]]
            dip_positions = domain_lo + xs_norm[top] * domain_width
            lo_dip = float(dip_positions.min())
            hi_dip = float(dip_positions.max())
            freq_init = 0.5 * (lo_dip + hi_dip)
            half_sep = 0.5 * (hi_dip - lo_dip)
        else:
            # Single dip (or none resolved): frequency is the deepest point.
            # Use raw ys, not smoothed — np.convolve(mode='same') zero-pads the
            # boundaries and can drag the smoothed edge value below the real dip,
            # making argmin(smoothed) pick the domain edge.
            freq_init = domain_lo + float(xs_norm[np.argmin(ys)]) * domain_width
            half_sep = None

        ct_idx = param_names.index("c_total") if "c_total" in param_names else None
        zs_idx = param_names.index("zeeman_split") if "zeeman_split" in param_names else None

        def make_p0(freq_hz: float, half_sep_hz: float | None) -> list[float]:
            p0 = [(lo_bounds[i] + hi_bounds[i]) / 2.0 for i in range(len(param_names))]
            p0[scan_idx] = float(np.clip(freq_hz, domain_lo, domain_hi))
            if ct_idx is not None:
                p0[ct_idx] = float(np.clip(dip_depth, lo_bounds[ct_idx], hi_bounds[ct_idx]))
            if zs_idx is not None and half_sep_hz is not None:
                p0[zs_idx] = float(np.clip(half_sep_hz, lo_bounds[zs_idx], hi_bounds[zs_idx]))
            return p0

        # Candidate initializations.  The dip-detection init (primary) is correct
        # when detection succeeds; the alternates guard against noise fooling the
        # detection at low SNR.  We keep the fit with the lowest residual, so extra
        # candidates can only help.
        argmin_freq = domain_lo + float(xs_norm[np.argmin(ys)]) * domain_width
        candidates = [(freq_init, half_sep)]
        if half_sep is not None:
            # In case the two "most prominent" peaks were the wrong pair, also try
            # treating the deepest single dip as the pattern center.
            candidates.append((argmin_freq, half_sep))
        else:
            candidates.append((argmin_freq, None))

        def curve_fn(xs: np.ndarray, *params: float) -> np.ndarray:
            typed = inner.spec.unpack_params(list(params))
            return np.array([float(inner.compute(float(x), typed)) for x in xs])

        best_popt = None
        best_resid = np.inf
        for freq_c, half_c in candidates:
            try:
                popt, _ = curve_fit(
                    curve_fn,
                    xs_phys,
                    ys,
                    p0=make_p0(freq_c, half_c),
                    bounds=(lo_bounds, hi_bounds),
                    maxfev=1000,
                )
            except Exception:
                continue
            resid = float(np.sum((curve_fn(xs_phys, *popt) - ys) ** 2))
            if resid < best_resid:
                best_resid = resid
                best_popt = popt

        if best_popt is None:
            return None

        freq_phys = float(best_popt[scan_idx])
        if not (domain_lo <= freq_phys <= domain_hi):
            return None

        # Store the full fitted parameter vector so the visualization can draw the
        # actual fit instead of the (collapsed) SMC belief marginal mode.
        self._fit_params_phys = {n: float(v) for n, v in zip(param_names, best_popt)}

        step_phys = domain_width / max(1, n_pts - 1)
        return freq_phys, step_phys * 0.5

    def finalize(self) -> None:
        """Detect dips in the sweep and report the center frequency."""
        from scipy.signal import find_peaks

        self._flush_pending_obs()

        if self.history.count < 3:
            self._acquisition_lo = self._domain_lo
            self._acquisition_hi = self._domain_hi
            return

        xs = self.history.xs  # normalized [0, 1]
        ys = self.history.ys
        domain_width = self._domain_hi - self._domain_lo

        # Try a full model fit first — much more accurate than centroid heuristics.
        fit_result = self._try_model_fit(xs, ys)
        if fit_result is not None:
            freq_phys, uncert_phys = fit_result
            self._freq_estimate_phys = freq_phys
            self._freq_uncert_phys = uncert_phys
            self._signal_found = True
            signal_max_span = self._signal_max_span or self._model_signal_max_span()
            half_width_phys = (
                min(domain_width / 2, 1.5 * signal_max_span)
                if signal_max_span is not None
                else domain_width * 0.2
            )
            self._acquisition_lo = max(self._domain_lo, freq_phys - half_width_phys)
            self._acquisition_hi = min(self._domain_hi, freq_phys + half_width_phys)
            return

        signal_max_span = self._signal_max_span or self._model_signal_max_span()
        if signal_max_span is not None and domain_width > 0:
            half_width_norm = min(0.5, 1.5 * signal_max_span / domain_width)
        else:
            half_width_norm = 0.2

        n_pts = max(len(xs), 1)
        step = 1.0 / n_pts

        baseline = float(np.percentile(ys, 90))
        global_min = float(np.min(ys))
        dip_depth = max(1e-9, baseline - global_min)

        # Minimum separation between two dips: half the minimum signal span.
        signal_min_span = self._signal_min_span or self._model_signal_min_span()
        if signal_min_span is not None and domain_width > 0:
            min_sep_norm = 0.5 * signal_min_span / domain_width
        else:
            min_sep_norm = step * 2
        min_distance_samples = max(1, int(min_sep_norm / step))

        # Smoothing window must be narrower than the minimum expected dip separation
        # so that closely spaced dips (e.g. NV center triplet) are not merged.
        # Use at most (min_distance_samples - 1) steps, falling back to 1 if the
        # minimum separation is already 1 step.
        window = max(1, min(min_distance_samples - 1, n_pts // 100))
        smoothed = np.convolve(ys, np.ones(window) / window, mode="same") if window > 1 else ys

        # find_peaks works on local maxima; invert to find dips.
        # Use the larger of the depth-based and noise-based prominence floors so
        # that noise spikes (3σ) don't create false peaks on slowly-varying signals.
        noise_floor = 3.0 * max(float(self._noise_std), 1e-6) * max(baseline, 1e-9)
        prominence_threshold = max(0.25 * dip_depth, noise_floor)
        neg_smoothed = -smoothed
        peaks, _ = find_peaks(
            neg_smoothed,
            prominence=prominence_threshold,
            distance=min_distance_samples,
        )

        x0_norm: float
        sigma_x0_norm: float

        # How many dips does the underlying model expect?
        n_expected_dips: int = 1
        try:
            inner = self._inner_model()
            if hasattr(inner, "expected_dip_count"):
                n_expected_dips = int(inner.expected_dip_count())
        except Exception:
            pass

        # Weighted centroid of the absorption feature — used as fallback when dips
        # overlap and cannot be individually resolved.
        #
        # Subtract the noise floor from each weight before summing so that
        # baseline-noise fluctuations (which affect ~90 % of non-dip points when
        # using the 90th-percentile baseline) don't swamp the actual dip signal
        # and drag the centroid toward the domain center.
        # Points that are not significantly below baseline get weight=0.
        weights = np.maximum(0.0, (baseline - ys) - noise_floor)
        centroid_norm = float(np.sum(xs * weights) / weights.sum()) if weights.sum() > 0 else float(xs[np.argmin(ys)])

        if len(peaks) == 0:
            x0_norm = centroid_norm
            sigma_x0_norm = step * 3
        else:
            # Filter peaks to those within the signal span of the global deepest dip.
            deepest_peak = peaks[np.argmin(ys[peaks])]
            deepest_x = float(xs[deepest_peak])
            nearby_mask = np.abs(xs[peaks] - deepest_x) <= half_width_norm
            nearby = peaks[nearby_mask]

            # Sort by position (ascending frequency).
            nearby = nearby[np.argsort(xs[nearby])]

            if len(nearby) >= 3 and n_expected_dips >= 3:
                # Three or more dips from a triplet model: median position = center freq.
                x0_norm = float(xs[nearby[len(nearby) // 2]])
                sigma_x0_norm = step
            elif len(nearby) == 2 and n_expected_dips >= 3:
                # Two dips found from a triplet model.  With k_np>1 the deepest is
                # freq+split (rightmost) and the second deepest is freq (leftmost).
                # Taking the left peak gives the center directly.
                x0_norm = float(xs[nearby[0]])
                sigma_x0_norm = step
            elif len(nearby) >= 2:
                # Doublet or generic model: midpoint of the outermost pair.
                x0_norm = float((xs[nearby[0]] + xs[nearby[-1]]) / 2.0)
                sigma_x0_norm = step / np.sqrt(2)
            else:
                # Single peak — dips are merged.  Use centroid rather than the
                # global minimum to reduce the systematic bias toward freq+split.
                x0_norm = centroid_norm
                sigma_x0_norm = step * 2

        x0_norm = float(np.clip(x0_norm, 0.0, 1.0))

        if domain_width > 0:
            self._freq_estimate_phys = self._domain_lo + x0_norm * domain_width
            self._freq_uncert_phys = sigma_x0_norm * domain_width
        else:
            self._freq_estimate_phys = self._domain_lo
            self._freq_uncert_phys = None

        self._signal_found = True
        acq_lo_norm = max(0.0, x0_norm - half_width_norm)
        acq_hi_norm = min(1.0, x0_norm + half_width_norm)
        self._acquisition_lo = self._domain_lo + acq_lo_norm * domain_width
        self._acquisition_hi = self._domain_lo + acq_hi_norm * domain_width

    def result(self) -> dict[str, float]:
        """Return dip-center estimate and uncertainty."""
        res = super().result()
        if self._freq_estimate_phys is not None:
            res["frequency"] = self._freq_estimate_phys
        if self._freq_uncert_phys is not None:
            res["uncert"] = self._freq_uncert_phys
        return res

    def fit_mode_estimates(self) -> dict[str, float] | None:
        """Full fitted physical parameters from the model fit, or None if unavailable.

        This is the actual least-squares fit of the physical model to the sweep
        data — used by the visualization in preference to the SMC belief marginal
        mode, which is not resampled during the sweep and can collapse.
        """
        return dict(self._fit_params_phys) if self._fit_params_phys is not None else None

    def effective_initial_sweep_steps(self) -> int:
        return self.effective_step_count()

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        return None
