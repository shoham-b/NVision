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

# Per-fit evaluation budget.  curve_fit's default tolerances (~1.5e-8) converge in
# ~50-200 evaluations; a larger budget only matters if a start is far off.
_FIT_MAXFEV: int = int(os.getenv("NVISION_SWEEP_FIT_MAXFEV", "4000"))

# The frequency step tolerance (curve_fit xtol) is derived from the expected
# frequency CRLB rather than hard-coded: refine until the frequency is pinned to
# this fraction of its Cramér-Rao floor.  Refining tighter than the CRLB is
# meaningless (it is below the information the data carries) and only burns
# iterations; refining looser leaves accuracy on the table.  This mirrors how the
# sbed locator gates convergence on NVISION_FREQ_CRLB_SAFETY_FACTOR × CRLB — here
# we target CRLB × _FIT_CRLB_TOL_FRAC as the optimizer's stopping precision.
_FIT_CRLB_TOL_FRAC: float = float(os.getenv("NVISION_SWEEP_FIT_CRLB_TOL_FRAC", "0.1"))
# Clamp the resulting relative tolerance to a sane band so a degenerate CRLB
# estimate can't make the fit either never converge (too tight) or quit
# immediately (too loose).
_FIT_XTOL_MIN: float = float(os.getenv("NVISION_SWEEP_FIT_XTOL_MIN", "1e-10"))
_FIT_XTOL_MAX: float = float(os.getenv("NVISION_SWEEP_FIT_XTOL_MAX", "1e-6"))

# Multi-start grid sizes.  The fit is non-convex: at low SNR a single start lands
# in a wrong local minimum.  Restarting from a grid of (frequency × zeeman_split)
# and keeping the lowest-residual fit escapes those basins — this is where extra
# compute actually refines the result (more iterations per fit do not).  The grid
# must be fine enough: a too-coarse grid can miss the true basin and do worse than
# a smart single start.
_FIT_FREQ_STARTS: int = int(os.getenv("NVISION_SWEEP_FIT_FREQ_STARTS", "5"))
_FIT_ZEEMAN_STARTS: int = int(os.getenv("NVISION_SWEEP_FIT_ZEEMAN_STARTS", "4"))

# Peak-SNR (dip depth / noise_std) below which the multi-start grid is engaged.
# Above it, the two smart starts already converge to sub-MHz accuracy, so the grid
# (~20 extra fits) is skipped.  Set from empirical accuracy: at SNR>=6 the smart
# starts give sub-MHz median with no failures; below that a single start starts
# landing in wrong basins and the grid is needed.
_FIT_GRID_SNR: float = float(os.getenv("NVISION_SWEEP_FIT_GRID_SNR", "6.0"))


def _expected_frequency_crlb(linewidth: float, c_total: float, noise_std: float, n_obs: int, bandwidth: float) -> float:
    """Closed-form Cramér-Rao lower bound for frequency (physical Hz), uniform sweep.

    ``CRLB_f = sqrt(2 σ² Ω / (π a² ρ))`` with σ = noise std, Ω = linewidth,
    a = contrast (c_total), ρ = n_obs / bandwidth (measurements per Hz).  This is
    the same closed form the SMC belief exposes as ``crlb_frequency`` for the NV
    Lorentzian; duplicated here because the sweep fits a raw physical model with no
    belief to query.  Returns ``inf`` when inputs are degenerate.
    """
    if linewidth <= 0 or c_total <= 0 or noise_std <= 0 or bandwidth <= 0 or n_obs <= 0:
        return float("inf")
    rho = n_obs / bandwidth
    variance = (2.0 * noise_std**2 * linewidth) / (np.pi * c_total**2 * rho)
    return float(np.sqrt(max(variance, 0.0)))


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

        # Fit results set by finalize() — reported via result().
        self._freq_estimate_phys: float | None = None
        self._freq_uncert_phys: float | None = None

        # Full fitted physical parameter vector from the model fit (set by
        # _fit_model).  Used to draw the actual fit in the visualization
        # instead of the SMC belief marginal mode.
        self._fit_params_phys: dict[str, float] | None = None

        # Buffer observations so we can batch-update the belief once at finalize()
        # instead of paying a full Bayesian update on every step.
        self._pending_obs: list = []

        # Parameter bounds stored by create() when provided as an argument.
        # Used by _fit_model() when signal_model has no param_bounds_phys.
        self._parameter_bounds: dict[str, tuple[float, float]] | None = None

        # Generate sweep points across the full domain.
        self._sweep_points = self._generate_sweep_points(max_steps)

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
                mapped_obs.append(
                    Observation(
                        x=x_phys,
                        signal_value=o.signal_value,
                        noise_std=o.noise_std,
                        frequency_noise_model=o.frequency_noise_model,
                    )
                )

        if hasattr(self.belief, "batch_update"):
            chunk = NVISION_SWEEP_BATCH_CHUNK_SIZE
            for i in range(0, len(mapped_obs), chunk):
                self.belief.batch_update(mapped_obs[i : i + chunk])
        else:
            for obs in mapped_obs:
                self.belief.update(obs)
        self._pending_obs.clear()

    def _fit_model(self, xs_norm: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        """Fit the physical model to sweep data via least squares.

        Returns (freq_phys, uncert_phys).  The fit uses all sweep points
        jointly, which gives sub-step accuracy even when dips are partially
        merged — much better than centroid heuristics.  Raises if the model
        can't be fit; a `GenericSweepLocator` sweep is only useful *because*
        of this fit, so a failure here is a real problem to surface, not a
        case to silently paper over with a cruder estimate.
        """
        inner = self._inner_model()
        param_names = inner.parameter_names()
        scan_param = self._scan_param
        if scan_param not in param_names:
            raise ValueError(f"scan_param {scan_param!r} is not one of the model's parameters {param_names}")
        scan_idx = param_names.index(scan_param)
        # Contrast/amplitude parameter name varies by model family: "c_total"
        # (Lorentzian), "c_max" (saturation-Voigt), "dip_depth" (plain Voigt).
        ct_idx = next((param_names.index(n) for n in ("c_total", "c_max", "dip_depth") if n in param_names), None)
        zs_idx = param_names.index("zeeman_split") if "zeeman_split" in param_names else None
        split_idx = param_names.index("split") if "split" in param_names else None

        param_bounds_phys = self._resolve_physical_bounds(param_names)
        lo_bounds = [param_bounds_phys[n][0] for n in param_names]
        hi_bounds = [param_bounds_phys[n][1] for n in param_names]

        domain_lo, domain_hi = self._domain_lo, self._domain_hi
        domain_width = domain_hi - domain_lo
        xs_phys = domain_lo + xs_norm * domain_width
        n_pts = len(xs_norm)

        smoothed, window = self._smooth_for_peak_detection(ys)
        dip_depth = self._estimate_dip_depth(ys, smoothed, window)
        hwhm_est = self._estimate_hwhm(xs_norm, smoothed, dip_depth, domain_width)
        # Width parameter name varies by model family: "linewidth" (Lorentzian,
        # HWHM-scale) or "fwhm_total" (Voigt, full-width — 2x the HWHM).
        width_idx = next((param_names.index(n) for n in ("linewidth", "fwhm_total") if n in param_names), None)
        seeds = self._seed_frequency_and_splits(
            xs_norm, ys, smoothed, dip_depth, domain_lo, domain_width, zs_idx, split_idx, lo_bounds, hi_bounds
        )

        def make_p0(freq_hz: float, half_sep_hz: float | None, hf_split_hz: float | None) -> list[float]:
            p0 = [(lo_bounds[i] + hi_bounds[i]) / 2.0 for i in range(len(param_names))]
            p0[scan_idx] = float(np.clip(freq_hz, domain_lo, domain_hi))
            if ct_idx is not None:
                p0[ct_idx] = float(np.clip(dip_depth, lo_bounds[ct_idx], hi_bounds[ct_idx]))
            if width_idx is not None and hwhm_est is not None and hwhm_est > 0:
                width_val = hwhm_est if param_names[width_idx] == "linewidth" else 2.0 * hwhm_est
                p0[width_idx] = float(np.clip(width_val, lo_bounds[width_idx], hi_bounds[width_idx]))
            if zs_idx is not None and half_sep_hz is not None:
                p0[zs_idx] = float(np.clip(half_sep_hz, lo_bounds[zs_idx], hi_bounds[zs_idx]))
            if split_idx is not None and hf_split_hz is not None:
                p0[split_idx] = float(np.clip(hf_split_hz, lo_bounds[split_idx], hi_bounds[split_idx]))
            return p0

        candidates = self._build_fit_candidates(
            xs_norm=xs_norm,
            ys=ys,
            domain_lo=domain_lo,
            domain_width=domain_width,
            dip_depth=dip_depth,
            seeds=seeds,
            zs_idx=zs_idx,
            split_idx=split_idx,
            lo_bounds=lo_bounds,
            hi_bounds=hi_bounds,
        )

        if hwhm_est is not None and hwhm_est > 0:
            lw_guess = hwhm_est
        elif width_idx is not None:
            mid = (lo_bounds[width_idx] + hi_bounds[width_idx]) / 2.0
            lw_guess = mid if param_names[width_idx] == "linewidth" else mid / 2.0
        else:
            lw_guess = domain_width * 0.01
        xtol = self._fit_xtol(dip_depth, n_pts, domain_width, lw_guess)

        # Evaluate in float64 (scalar path).  The vectorized kernel uses float32,
        # whose ~1e-7 relative resolution is coarser than curve_fit's numerical
        # Jacobian step for a GHz-scale frequency (~tens of Hz), which zeroes out
        # the frequency gradient and stalls the fit.  float64 is required here.
        def curve_fn(xs: np.ndarray, *params: float) -> np.ndarray:
            typed = inner.spec.unpack_params(list(params))
            return np.array([float(inner.compute(float(x), typed)) for x in xs])

        best_popt, best_pcov = self._run_curve_fit_candidates(
            curve_fn, xs_phys, ys, candidates, make_p0, lo_bounds, hi_bounds, xtol
        )

        freq_phys = float(best_popt[scan_idx])
        if not (domain_lo <= freq_phys <= domain_hi):
            raise RuntimeError(f"fitted frequency {freq_phys} fell outside the domain [{domain_lo}, {domain_hi}]")

        # Store the full fitted parameter vector so the visualization can draw the
        # actual fit instead of the (collapsed) SMC belief marginal mode.
        self._fit_params_phys = {n: float(v) for n, v in zip(param_names, best_popt)}

        uncert_phys = self._report_fit_uncertainty(best_pcov, scan_idx, dip_depth, n_pts, domain_width, lw_guess)
        return freq_phys, uncert_phys

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
            window = max(3, min(window, samples_per_min_span))
        smoothed = np.convolve(ys, np.ones(window) / window, mode="same") if window > 1 else ys
        return smoothed, window

    @staticmethod
    def _estimate_dip_depth(ys: np.ndarray, smoothed: np.ndarray, window: int) -> float:
        """Estimate the dip depth (baseline minus minimum) directly from the data.

        Seeds ``c_total`` close to the true value — a blind bound-midpoint
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
        xs_norm: np.ndarray, smoothed: np.ndarray, dip_depth: float, domain_width: float
    ) -> float | None:
        """Estimate the deepest dip's half-width-at-half-max (HWHM) directly
        from the data, in the same units as this codebase's ``linewidth``
        parameter (see ``nv_center_lorentzian_eval``: ``omega`` in
        ``(x-freq)/omega`` is the HWHM scale, not the full width).

        Unlike `c_total`/contrast — whose value at the exact dip center is
        independent of width by construction (population-normalized
        reparametrization) — `linewidth` had no data-driven seed at all and
        always started at the bounds midpoint.  A bounds range can span 10-25x
        (e.g. 0.2-5 MHz), so a midpoint start can be far enough from the true
        width that curve_fit converges to a *compensating* (wrong width,
        wrong contrast) local minimum instead of the true one, even though
        frequency/splits/contrast were all seeded correctly.
        """
        if dip_depth <= 0:
            return None
        baseline = float(np.max(smoothed))
        half_level = baseline - 0.5 * dip_depth
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
        fwhm_norm = float(xs_norm[right] - xs_norm[left])
        return 0.5 * fwhm_norm * domain_width

    def _seed_frequency_and_splits(
        self,
        xs_norm: np.ndarray,
        ys: np.ndarray,
        smoothed: np.ndarray,
        dip_depth: float,
        domain_lo: float,
        domain_width: float,
        zs_idx: int | None,
        split_idx: int | None,
        lo_bounds: list[float],
        hi_bounds: list[float],
    ) -> list[tuple[float, float | None, float | None]]:
        """Seed (frequency, zeeman half-separation, hyperfine split) initial
        guesses for the fit, one tuple per plausible interpretation of the
        detected peaks.  The primary (most likely) seed comes first; all are
        raced by residual in ``_run_curve_fit_candidates``.

        This is the crucial step for multi-dip spectra: the raw argmin picks a
        *single* dip, but `frequency` is the *center* of the whole pattern —
        e.g. for a two-dip Zeeman spectrum that center sits ~zeeman_split away
        from either dip, so an argmin init is systematically wrong and the fit
        diverges.

        The model can present 1, 2, 3, or 6 resolvable dips (see
        SignalModel.expected_dip_count): none, Zeeman-only, hyperfine-only, or
        both combined (2 Zeeman groups x 3 hyperfine lines, each line resolved
        individually — see NVCenterLorentzianModel).  We detect all peaks, keep
        the most prominent ones (bounded by how many the model expects, so
        stray noise peaks don't corrupt the grouping), then split them into
        Zeeman groups at the largest position gap — physically valid because
        zeeman_split (up to 100 MHz) is normally much larger than the
        hyperfine split (2-8.5 MHz) that separates lines *within* a group.

        For the hyperfine triplet (dips at freq-split / freq / freq+split with
        depths 1/k : 1 : k), the peak count is genuinely ambiguous — depths
        span k² so the shallow line often hides below the detection floor —
        and each count implies a *different* frequency:

        - 3 peaks: middle = freq, outer span = 2*split (unambiguous).
        - 2 peaks: could be (center, deepest) → freq = left, split = gap; or
          (outer pair) → freq = mid, split = gap/2; or (shallow, center) →
          freq = right.  Seeding only one of these plants the fit in a wrong
          local minimum ~split/2 away that gradient descent cannot escape (the
          basins are separated by many linewidths).
        - 1 peak: it is the *deepest* line, which for k>1 sits at freq+split,
          not freq — so seed freq = argmin - split over the split grid, plus
          plain argmin for the k≈1 case.
        """
        from scipy.signal import find_peaks

        # Deepest raw point: use raw ys, not smoothed — np.convolve(mode='same')
        # zero-pads the boundaries and can drag the smoothed edge value below the
        # real dip, making argmin(smoothed) pick the domain edge.
        argmin_freq = domain_lo + float(xs_norm[np.argmin(ys)]) * domain_width

        # Prominence floor must clear the (smoothed) noise so isolated fluctuations
        # aren't mistaken for dips.  Smoothing reduces noise by ~sqrt(window), but we
        # keep a conservative multiple of noise_std.
        prom_floor = max(0.3 * dip_depth, 4.0 * max(float(self._noise_std), 1e-9))
        peaks, props = find_peaks(-smoothed, prominence=prom_floor)
        n_expected_dips = self._expected_dip_count_from_model()
        if zs_idx is not None and split_idx is not None:
            # Some models' expected_dip_count() assumes the nitrogen hyperfine
            # triplet is always merged into one dip per Zeeman group — but the
            # hyperfine splitting is a real physical constant
            # (NV_N14_HYPERFINE_SPLIT_HZ) baked into the model's evaluation
            # *even when with_hyperfine_splitting=False* (split/k_np just
            # become fixed instead of free parameters; see
            # NVCenterLorentzianModel.compute). Whether the 3 lines actually
            # resolve depends on the fitted linewidth vs. that fixed split,
            # not on whether hyperfine is a free parameter — a narrow-linewidth
            # draw (e.g. 500 kHz vs. ~2.16 MHz hyperfine spacing) resolves each
            # Zeeman group into 3 separate dips regardless. A cap of 2 here
            # then keeps two sub-lines from the *same* Zeeman group instead of
            # one from each, seeding zeeman_split tens of MHz off from the
            # true value with no way for curve_fit to recover. Never cap below
            # the full 6-line pattern whenever Zeeman splitting is a free
            # parameter, whether or not hyperfine is too.
            n_expected_dips = max(n_expected_dips, 6)

        if len(peaks) < 2:
            seeds: list[tuple[float, float | None, float | None]] = [(argmin_freq, None, None)]
            if split_idx is not None:
                # The lone detected dip is the deepest hyperfine line, at
                # freq+split for k_np>1 — seed the center accordingly across
                # the split grid (the true split is unknown here).
                for h in np.linspace(lo_bounds[split_idx], hi_bounds[split_idx], _FIT_ZEEMAN_STARTS):
                    seeds.append((argmin_freq - float(h), None, float(h)))
            if zs_idx is not None:
                # A single resolved peak is also the signature of near-merged
                # Zeeman dips (zeeman_split too small to resolve at this
                # noise/resolution) — the blind bounds-midpoint start that
                # `half_sep=None` falls back to in make_p0 is then a bad
                # guess (e.g. ~50 MHz when the truth is <1 MHz) that can trap
                # the fit at an unrelated local minimum regardless of the true
                # small split. Seed zeeman_split≈0 explicitly alongside the
                # blind-midpoint version of every seed above.
                seeds = seeds + [(f, 0.0, h) for f, _, h in seeds]
            return seeds

        # Keep the most prominent peaks, capped at what the model expects
        # (at least 2), so stray noise peaks don't corrupt the grouping.
        keep = max(2, min(len(peaks), n_expected_dips))
        top_order = np.argsort(props["prominences"])[::-1][:keep]
        positions = np.sort(domain_lo + xs_norm[peaks[top_order]] * domain_width)

        if zs_idx is not None:
            # Split into the two Zeeman groups at the largest position gap.
            split_at = int(np.argmax(np.diff(positions)))
            group_lo = positions[: split_at + 1]
            group_hi = positions[split_at + 1 :]
            center_lo = float(np.mean(group_lo))
            center_hi = float(np.mean(group_hi))
            freq_init = 0.5 * (center_lo + center_hi)
            half_sep = 0.5 * abs(center_hi - center_lo)

            if split_idx is None:
                return [(freq_init, half_sep, None)]

            # Each Zeeman group is itself a hyperfine triplet, subject to the
            # exact same ambiguity as the no-zeeman branch below: with only 2
            # of 3 lines resolved (the shallow k_np-suppressed line often
            # falls below the prominence floor), the within-group spread could
            # be the true split (adjacent pair) or 2x the true split (outer
            # pair) — picking only one seeds a wrong k_np/split local minimum
            # gradient descent can't escape.  Offer every group's candidates;
            # freq_init/half_sep stay fixed (from the group means) since a
            # partially-resolved group's mean is still a reasonable frequency
            # estimate, and the residual race sorts out `split`.
            hf_candidates: set[float] = set()
            for group in (group_lo, group_hi):
                if len(group) >= 3:
                    hf_candidates.add(round(float(group.max() - group.min()) / 2.0, 6))
                elif len(group) == 2:
                    gap = float(group[1] - group[0])
                    hf_candidates.add(round(gap, 6))
                    hf_candidates.add(round(0.5 * gap, 6))
            if not hf_candidates:
                return [(freq_init, half_sep, None)]
            return [(freq_init, half_sep, h) for h in sorted(hf_candidates)]

        if split_idx is not None:
            if len(positions) >= 3:
                # Full triplet resolved: middle dip = frequency, outer span = 2*split.
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
        freq_init = 0.5 * (float(positions[0]) + float(positions[-1]))
        half_sep = 0.5 * (float(positions[-1]) - float(positions[0]))
        return [(freq_init, half_sep, None)]

    def _build_fit_candidates(
        self,
        *,
        xs_norm: np.ndarray,
        ys: np.ndarray,
        domain_lo: float,
        domain_width: float,
        dip_depth: float,
        seeds: list[tuple[float, float | None, float | None]],
        zs_idx: int | None,
        split_idx: int | None,
        lo_bounds: list[float],
        hi_bounds: list[float],
    ) -> list[tuple[float, float | None, float | None]]:
        """Build (freq, half_sep, hf_split) candidate initializations for curve_fit.

        All candidates are scored by residual once fit (lowest wins):
          1. dip-detection seeds — one per plausible peak interpretation
             (see ``_seed_frequency_and_splits``)
          2. deepest-point init as a cheap alternate
          3. (hyperfine split unknown only) a small grid over split — when both
             Zeeman and hyperfine splitting are present (the 6-dip case), the
             two Zeeman groups are usually resolved by peak detection but their
             internal hyperfine lines are often smoothed together, leaving the
             seeds without a split value.  A single blind bounds-midpoint start
             for `split` can then trap the fit in a wrong split/k_np basin —
             this is missing information, not noise-corrupted detection, so
             it's added unconditionally (and kept small).
          4. (low SNR only) a coarse grid over frequency x zeeman_split (or,
             for hyperfine-only models with no Zeeman splitting, x split) — the
             safety net for low SNR, where dip detection is fooled by noise and
             a single start lands in a wrong local minimum.  Gated on SNR —
             cheap to compute *before* fitting — rather than on the fit
             residual, because at low SNR the residual gap between a right and
             a wrong fit is within the chi-square noise band and cannot tell
             them apart.  High-SNR spectra are solved by the smart starts in a
             few fits; only genuinely hard low-SNR spectra pay for the full
             multi-start grid.
        """
        freq_init, half_sep, hf_split_init = seeds[0]
        argmin_freq = domain_lo + float(xs_norm[np.argmin(ys)]) * domain_width
        candidates = list(seeds)
        candidates.append((argmin_freq, half_sep, hf_split_init))

        if split_idx is not None and all(s[2] is None for s in seeds):
            hf_grid0 = np.linspace(lo_bounds[split_idx], hi_bounds[split_idx], _FIT_ZEEMAN_STARTS)
            candidates += [(freq_init, half_sep, float(h)) for h in hf_grid0]

        snr = dip_depth / max(float(self._noise_std), 1e-9)
        if snr < _FIT_GRID_SNR:
            freq_grid = [domain_lo + (i + 0.5) / _FIT_FREQ_STARTS * domain_width for i in range(_FIT_FREQ_STARTS)]
            if zs_idx is not None:
                zee_grid = np.linspace(lo_bounds[zs_idx], hi_bounds[zs_idx], _FIT_ZEEMAN_STARTS)
                candidates += [(f, float(z), hf_split_init) for f in freq_grid for z in zee_grid]
            elif split_idx is not None:
                hf_grid = np.linspace(lo_bounds[split_idx], hi_bounds[split_idx], _FIT_ZEEMAN_STARTS)
                candidates += [(f, None, float(h)) for f in freq_grid for h in hf_grid]
            else:
                candidates += [(f, None, None) for f in freq_grid]

        return candidates

    def _fit_xtol(self, dip_depth: float, n_pts: int, domain_width: float, lw_guess: float) -> float:
        """Convergence tolerance (curve_fit xtol) derived from the expected
        frequency CRLB: refine until the frequency is pinned to
        ``_FIT_CRLB_TOL_FRAC`` of its Cramér-Rao floor.  Refining tighter than
        the CRLB is meaningless (it is below the information the data
        carries) and only burns iterations; refining looser leaves accuracy on
        the table.

        Using a rough pre-fit linewidth (bounds midpoint) and the data-driven
        contrast (dip_depth) is enough — the CRLB only sets the *scale* of the
        tolerance, not the estimate.  The optimizer runs with per-parameter
        ``x_scale`` = bounds width (see ``_run_curve_fit_candidates``), so
        steps are measured in bounds-width units: a frequency target of
        (CRLB × frac) Hz corresponds to a scaled tolerance of
        (CRLB × frac) / domain_width.
        """
        crlb_f = _expected_frequency_crlb(lw_guess, dip_depth, float(self._noise_std), n_pts, domain_width)
        if np.isfinite(crlb_f) and crlb_f > 0 and domain_width > 0:
            return float(np.clip(_FIT_CRLB_TOL_FRAC * crlb_f / domain_width, _FIT_XTOL_MIN, _FIT_XTOL_MAX))
        return 1e-8  # curve_fit default when the CRLB is unavailable

    @staticmethod
    def _run_curve_fit_candidates(
        curve_fn,
        xs_phys: np.ndarray,
        ys: np.ndarray,
        candidates: list[tuple[float, float | None, float | None]],
        make_p0,
        lo_bounds: list[float],
        hi_bounds: list[float],
        xtol: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run curve_fit from each candidate start, keep the lowest-residual result.

        ``x_scale`` = bounds width per parameter is essential: the raw
        parameter vector mixes a GHz-scale frequency with O(1) shape
        parameters (k_np, c_total), a ~1e9 conditioning spread.  Without
        scaling, the trust-region steps and the xtol termination test are
        dominated by the frequency axis, so the optimizer stops while the
        *shape* parameters are still far from their optimum — frequency comes
        out fine but the fitted curve's depth/width visibly deviates from the
        data.

        Raises if every candidate fails to converge.
        """
        from scipy.optimize import curve_fit

        x_scale = np.maximum(np.asarray(hi_bounds, dtype=float) - np.asarray(lo_bounds, dtype=float), 1e-12)

        best_popt = None
        best_pcov = None
        best_resid = np.inf
        for freq_c, half_c, hf_c in candidates:
            try:
                popt, pcov = curve_fit(
                    curve_fn,
                    xs_phys,
                    ys,
                    p0=make_p0(freq_c, half_c, hf_c),
                    bounds=(lo_bounds, hi_bounds),
                    maxfev=_FIT_MAXFEV,
                    xtol=xtol,
                    x_scale=x_scale,
                )
            except Exception:
                continue
            resid = float(np.sum((curve_fn(xs_phys, *popt) - ys) ** 2))
            if resid < best_resid:
                best_resid = resid
                best_popt = popt
                best_pcov = pcov

        if best_popt is None:
            raise RuntimeError(f"curve_fit failed to converge from any of {len(candidates)} candidate starts")
        return best_popt, best_pcov

    def _report_fit_uncertainty(
        self,
        best_pcov: np.ndarray | None,
        scan_idx: int,
        dip_depth: float,
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
            var = float(best_pcov[scan_idx, scan_idx])
            if var > 0:
                fit_std = float(np.sqrt(var))
        contrast_fit = next(
            (self._fit_params_phys[n] for n in ("c_total", "c_max", "dip_depth") if n in self._fit_params_phys),
            dip_depth,
        )
        crlb_at_fit = _expected_frequency_crlb(
            self._fit_params_phys.get("linewidth", lw_guess),
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
        """Fit the physical model to the sweep and report the center frequency.

        Order matters: the fit must run BEFORE the belief flush. The SMC
        belief's batch update can resample and auto-narrow the frequency
        bounds *in place* on the shared ``signal_model.param_bounds_phys``
        (see ``UnitCubeSMCMarginalDistribution.narrow_scan_parameter_physical_bounds``)
        — and a batch-updated belief is collapsed, so it narrows to the wrong
        window using the global RNG. Fitting first keeps the fit anchored to
        the original physical bounds; the flush only exists so visualizations
        can show a belief.

        Raises if the fit fails — see ``_fit_model``.
        """
        domain_width = self._domain_hi - self._domain_lo
        freq_phys, uncert_phys = self._fit_model(self.history.xs, self.history.ys)

        self._flush_pending_obs()

        self._freq_estimate_phys = freq_phys
        self._freq_uncert_phys = uncert_phys
        self._signal_found = True
        signal_max_span = self._signal_max_span or self._model_signal_max_span()
        half_width_phys = (
            min(domain_width / 2, 1.5 * signal_max_span) if signal_max_span is not None else domain_width * 0.2
        )
        self._acquisition_lo = max(self._domain_lo, freq_phys - half_width_phys)
        self._acquisition_hi = min(self._domain_hi, freq_phys + half_width_phys)

    def result(self) -> dict[str, float]:
        """Return dip-center estimate, uncertainty, and fit-derived sweep metrics."""
        res = super().result()
        if self._freq_estimate_phys is not None:
            res["frequency"] = self._freq_estimate_phys
        if self._freq_uncert_phys is not None:
            res["uncert"] = self._freq_uncert_phys
        res.update(self._compute_sweep_metrics())
        return res

    def _compute_sweep_metrics(self) -> dict[str, float | int]:
        """Sweep efficiency metrics derived from the model fit.

        ``finalize()`` always produces a full least-squares fit (or raises),
        so the dip count and width are known exactly from the model and the
        fit — no need to re-detect dips from noisy sweep y-values the way a
        locator without a fit (e.g. ``StagedSobolSweepLocator``) has to.
        """
        domain_width = self._domain_hi - self._domain_lo
        expected_dips = self._expected_dip_count_from_model()

        dip_width = None
        if self._fit_params_phys is not None:
            dip_width = self._fit_params_phys.get("linewidth") or self._fit_params_phys.get("fwhm_total")
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

    def bayesian_focus_window(self) -> tuple[float, float] | None:
        return None
