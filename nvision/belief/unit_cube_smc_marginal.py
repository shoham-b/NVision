"""SMC belief with particles on ``[0, 1]`` and physical-scale public summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nvision.belief.abstract_marginal import ParameterValues
from nvision.belief.coordinate import RescaleMap
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.models.observation import Observation
from nvision.spectra.noise_model import NoiseSignalModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


class UnitCubeNoiseSignalModelWrapper(NoiseSignalModel):
    """Wrap a physical NoiseSignalModel to map unit-cube particles to physical space for likelihood evaluation."""

    def __init__(self, inner: NoiseSignalModel, physical_param_bounds: dict[str, tuple[float, float]]):
        self.inner = inner
        self.physical_param_bounds = physical_param_bounds

    @property
    def spec(self):
        return self.inner.spec

    def composite_log_likelihood(
        self,
        predicted: np.ndarray,
        residuals: np.ndarray,
        noise_param_arrays: Sequence[np.ndarray],
        sigma_epistemic: float,
    ) -> np.ndarray:
        phys_arrays = []
        names = self.inner.spec.names
        for name, arr in zip(names, noise_param_arrays, strict=True):
            lo, hi = self.physical_param_bounds[name]
            phys_arr = lo + arr * (hi - lo)
            phys_arrays.append(phys_arr)
        return self.inner.composite_log_likelihood(predicted, residuals, phys_arrays, sigma_epistemic)


@dataclass
class UnitCubeSMCMarginalDistribution(SMCMarginalDistribution):
    """Like :class:`SMCMarginalDistribution` but particles live on ``[0, 1]``.

    The ``model`` must be a :class:`UnitCubeSignalModel` mapping unit coordinates to
    the inner physical model. Internal particles and uncertainties are in normalized
    space so acquisition and ``converged()`` thresholds apply uniformly across
    parameters.

    :meth:`estimates` and :meth:`uncertainty` are returned in **physical** units for
    metrics, plotting, and comparison.
    """

    physical_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    physical_x_bounds: tuple[float, float] = (0.0, 1.0)

    @property
    def physical_param_bounds(self) -> dict[str, tuple[float, float]]:  # type: ignore[override]  # noqa: F811
        if hasattr(self, "_physical_param_bounds"):
            return self._physical_param_bounds
        return {}

    @physical_param_bounds.setter
    def physical_param_bounds(self, value: dict[str, tuple[float, float]]) -> None:
        self._physical_param_bounds = value

    @property
    def _rescale_maps(self) -> dict[str, RescaleMap]:
        """Return a ``RescaleMap`` per parameter using the *original* physical bounds.

        For the frequency axis this uses ``_original_physical_x_bounds`` — the
        full domain set at construction and never narrowed — so that
        ``freq_rescale.to_phys(o.x)`` always converts the stored [0, 1]
        observation coordinate correctly, even after the focus window has
        shrunk ``physical_param_bounds["frequency"]``.

        For all other parameters the current ``physical_param_bounds`` is
        used (those bounds are never narrowed).
        """
        if not hasattr(self, "_original_physical_x_bounds"):
            raise RuntimeError(
                "UnitCubeSMCMarginalDistribution._rescale_maps called before "
                "__post_init__ set _original_physical_x_bounds."
            )
        phys = self.physical_param_bounds
        # "frequency" is always the probe/measurement x-axis, independent of
        # whether it's also a free/inferred particle dimension (it may not be,
        # e.g. NVCenterVoigtModel(with_fixed_frequency=True)) -- so its rescale
        # map is built unconditionally rather than gated on self._param_names.
        lo, hi = self._original_physical_x_bounds
        maps: dict[str, RescaleMap] = {"frequency": RescaleMap(lo=float(lo), hi=float(hi))}
        for name in self._param_names:
            if name == "frequency":
                continue
            if name in phys:
                lo, hi = phys[name]
            else:
                raise RuntimeError(
                    f"UnitCubeSMCMarginalDistribution is missing physical_param_bounds "
                    f"for parameter '{name}'. All parameters must have physical bounds at construction."
                )
            maps[name] = RescaleMap(lo=float(lo), hi=float(hi))
        return maps

    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeSMCMarginalDistribution requires a UnitCubeSignalModel")

        if self.noise_model is not None and not isinstance(self.noise_model, UnitCubeNoiseSignalModelWrapper):
            self.noise_model = UnitCubeNoiseSignalModelWrapper(self.noise_model, self.physical_param_bounds)

        # Ensure all parameters (including noise) are in unit space [0, 1].
        # We must detect noise parameters here because super().__post_init__
        # expects them to be in parameter_bounds before it iterates over _param_names.
        all_names = list(self.model.parameter_names())
        if self.noise_model is not None:
            all_names.extend(n for n in self.noise_model.spec.names if n not in all_names)

        self.parameter_bounds = {name: (0.0, 1.0) for name in all_names}
        # "frequency" is always the probe/measurement x-axis (needed by the base
        # class's __post_init__ to size the global candidate grid) even when it's
        # fixed and therefore absent from all_names/model.parameter_names(). This
        # doesn't reintroduce it as a particle dimension -- _param_names below is
        # set directly from model.parameter_names(), not from this dict's keys.
        if "frequency" not in self.parameter_bounds and "frequency" in self.physical_param_bounds:
            self.parameter_bounds["frequency"] = (0.0, 1.0)
        super().__post_init__()
        self._original_physical_x_bounds = self.physical_x_bounds

    def expected_information_gain(self, candidates: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """Override to normalize physical candidates to [0, 1] for the UnitCube model."""
        lo, hi = self.physical_x_bounds
        unit_candidates = (candidates - lo) / (hi - lo)
        return super().expected_information_gain(unit_candidates, noise_std=noise_std)

    def get_candidates(self) -> np.ndarray:
        """Return candidates in **physical** frequency space.

        The base class stores candidates in unit [0, 1] space (matching internal
        particles). The locator and ``expected_information_gain`` both operate in
        physical space, so we convert here before returning.
        """
        lo, hi = self.physical_x_bounds
        return lo + self._current_candidates.astype(np.float64) * (hi - lo)

    def _generate_epoch_candidates(self) -> None:
        """Generate candidates in unit space.

        The base class now uses internal _estimates_unit() which correctly
        returns [0, 1] values even for this subclass.
        """
        super()._generate_epoch_candidates()

    def estimates(self) -> dict[str, float]:
        raw = super().estimates()
        return {
            k: (
                self._to_physical(k, v)
                if (k != "noise_sigma" or not getattr(self, "_use_rao_blackwell_noise", False))
                else v
            )
            for k, v in raw.items()
        }

    def mode_estimates(self) -> dict[str, float]:
        raw = super().mode_estimates()
        return {
            k: (
                self._to_physical(k, v)
                if (k != "noise_sigma" or not getattr(self, "_use_rao_blackwell_noise", False))
                else v
            )
            for k, v in raw.items()
        }

    def _to_physical(self, name: str, u: float) -> float:
        lo, hi = self.physical_param_bounds[name]
        return lo + float(u) * (hi - lo)

    def _fim_param_values(self) -> dict[str, float]:
        """Unit-cube estimates — ``self.model`` is the unit-cube wrapper (see base docstring)."""
        return super().estimates()

    def crlb_per_param(self) -> dict[str, float]:
        """Marginal CRLB per parameter in **physical** units.

        The cumulative FIM is accumulated in unit-cube coordinates (that is the
        space ``self.model`` and ``obs.x`` live in), so its CRLBs come out as
        unit-cube stds and are rescaled by each parameter's physical range here
        — the same conversion :meth:`_empirical_uncertainty` applies, so the two
        are directly comparable by callers such as the CRLB early-stop.
        """
        raw = super().crlb_per_param()
        out: dict[str, float] = {}
        for name, std in raw.items():
            if name in self.physical_param_bounds:
                lo, hi = self.physical_param_bounds[name]
                out[name] = std * (hi - lo)
            else:
                out[name] = std
        return out

    def _empirical_uncertainty(self) -> ParameterValues[float]:
        raw = super()._empirical_uncertainty()
        data = {}
        for name, u in raw.items():
            if name == "noise_sigma" and getattr(self, "_use_rao_blackwell_noise", False):
                data[name] = u
            elif name in self.physical_param_bounds:
                data[name] = u * (self.physical_param_bounds[name][1] - self.physical_param_bounds[name][0])
            else:
                data[name] = u
        return ParameterValues.from_mapping(list(raw.keys()), data)

    def uncertainty(self) -> ParameterValues[float]:
        return self._empirical_uncertainty()

    def crlb_frequency(self) -> float:
        """Analytical CRLB for frequency (physical Hz) for the NV Lorentzian/Voigt models.

        Uses the closed-form Fisher result for uniform sampling of a single
        population-normalized dip of height-normalized shape ``V`` (peak 1) and
        amplitude ``a``:
        ``I(f) = (ρ/σ²)·a²·J``, ``CRLB_f = sqrt(σ² / (ρ·a²·J))``
        where σ = noise std, ρ = n_obs / bandwidth (measurements per Hz), and
        ``J = ∫(V'(x))² dx`` depends only on the lineshape.

        For a Lorentzian of HWHM Ω, ``J = π/(4Ω)`` — verified by direct numerical
        integration of ``(∂S/∂f)²`` against the coded ``nv_center_lorentzian_eval``
        (confirms the ``4`` in the denominator here, not ``2``).

        Returns ``math.inf`` for unsupported models or before any observations.
        """
        from nvision.spectra.nv_center import NVCenterLorentzianModel, NVCenterSaturationVoigtModel, NVCenterVoigtModel

        inner = getattr(self.model, "inner", None)

        if self._obs_count == 0 or self.last_obs is None:
            return math.inf

        noise_std = float(self.last_obs.noise_std)
        if noise_std <= 0:
            return math.inf

        lo, hi = self._original_physical_x_bounds
        bandwidth = hi - lo
        if bandwidth <= 0:
            return math.inf

        rho = self._obs_count / bandwidth  # measurements per Hz

        if isinstance(inner, NVCenterLorentzianModel):
            ests = self.estimates()  # physical-space values
            linewidth = ests.get("linewidth")
            c_total = ests.get("c_total")
            if linewidth is None or c_total is None or c_total <= 0 or linewidth <= 0:
                return math.inf
            variance = (4.0 * noise_std**2 * linewidth) / (math.pi * c_total**2 * rho)
            return math.sqrt(max(variance, 0.0))

        if isinstance(inner, NVCenterSaturationVoigtModel):
            from nvision.spectra.numba_kernels import _pv_factors
            from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar

            ests = self.estimates()  # physical-space values
            saturation = ests.get("saturation")
            sigma_inhom = ests.get("sigma_inhom")
            if saturation is None or sigma_inhom is None:
                return math.inf

            fwhm_total, lorentz_frac, c_total = _saturation_voigt_reparam_scalar(
                saturation, sigma_inhom, NV_SATURATION_C_MAX
            )
            if fwhm_total <= 0 or c_total <= 0:
                return math.inf

            elf, egf, nhs, gamma2, has_gamma, has_sigma = _pv_factors(fwhm_total, lorentz_frac)
            # J = ∫(V')²dx for the height-normalized pseudo-Voigt V, dropping the
            # Lorentzian x Gaussian cross-term (single-dip approximation, matching
            # the Lorentzian branch above). This underestimates J for genuinely
            # mixed lineshapes, i.e. is a conservative (larger) CRLB — verified
            # numerically to be exact in the sigma_inhom -> 0 (pure Lorentzian) limit.
            j_lorentz = (elf**2) * math.pi / (4.0 * gamma2**2.5) if has_gamma else 0.0
            j_gauss = (egf**2) * math.sqrt(-math.pi * nhs / 2.0) if has_sigma and nhs < 0 else 0.0
            j_total = j_lorentz + j_gauss
            if j_total <= 0:
                return math.inf

            variance = noise_std**2 / (rho * c_total**2 * j_total)
            return math.sqrt(max(variance, 0.0))

        if isinstance(inner, NVCenterVoigtModel):
            from nvision.spectra.numba_kernels import _pv_factors
            from nvision.spectra.nv_center import _voigt_reparam_scalar

            ests = self.estimates()  # physical-space values
            homogeneous_linewidth = ests.get("homogeneous_linewidth")
            sigma_inhom = ests.get("sigma_inhom")
            c_total = ests.get("c_total")
            if homogeneous_linewidth is None or sigma_inhom is None or c_total is None:
                return math.inf
            if homogeneous_linewidth <= 0 or c_total <= 0:
                return math.inf

            fwhm_total, lorentz_frac = _voigt_reparam_scalar(homogeneous_linewidth, sigma_inhom)
            if fwhm_total <= 0:
                return math.inf

            elf, egf, nhs, gamma2, has_gamma, has_sigma = _pv_factors(fwhm_total, lorentz_frac)
            # Same J = ∫(V')²dx single-dip approximation as the saturation-Voigt branch
            # above (shared pseudo-Voigt shape function and now the same c_total
            # amplitude convention too).
            j_lorentz = (elf**2) * math.pi / (4.0 * gamma2**2.5) if has_gamma else 0.0
            j_gauss = (egf**2) * math.sqrt(-math.pi * nhs / 2.0) if has_sigma and nhs < 0 else 0.0
            j_total = j_lorentz + j_gauss
            if j_total <= 0:
                return math.inf

            variance = noise_std**2 / (rho * c_total**2 * j_total)
            return math.sqrt(max(variance, 0.0))

        return math.inf

    def reported_uncertainty(self) -> ParameterValues[float]:
        """Physical uncertainty, floored so it never claims impossible precision.

        Two independent floors:

        * ``frequency`` — ``K × crlb_frequency()`` (closed-form), where K is
          ``NVISION_FREQ_CRLB_SAFETY_FACTOR``, the same factor the locator uses to
          raise the frequency convergence threshold, so at convergence the reported
          σ equals the convergence ceiling.
        * every other parameter — its own **marginal** CRLB from the cumulative
          FIM (``crlb_per_param()``, i.e. ``sqrt(diag(pinv(FIM)))``, so nuisance
          parameters are profiled out rather than held fixed).

        The second floor exists because the particle spread alone understates the
        uncertainty exactly where it matters most. When two parameters trade off
        along a near-flat ridge — ``zeeman_split`` against the width pair below the
        dip-resolution threshold is the standard case — the SMC posterior can be
        narrow and *wrong*, while the marginal CRLB correctly blows up because the
        FIM is near-singular in that direction. Measured over 120 mixed repeats
        (56 deliberately degenerate), flooring moves the median ``error / reported σ``
        for ``zeeman_split`` from 1.21 to 0.90 and cuts the fraction beyond 3σ from
        8.3% to 2.5%; ``c_total`` goes 1.61 -> 0.96 and 18.3% -> 3.3%. The width pair
        becomes ~2-3× conservative (medians 1.03/1.38 -> 0.36/0.49) because both sit
        *in* the degenerate direction even though their sum stays well determined --
        an accepted trade: overstating precision is the dangerous direction.

        No safety factor is applied to the per-parameter floor deliberately: the CRLB
        is already a hard lower bound on any unbiased estimator's variance, so 1× is
        the principled choice, and 4× (matching K) over-corrects badly (medians drop
        to 0.09-0.25). Control-flow paths must use :meth:`uncertainty` instead.
        """
        from nvision.sim.defaults import NVISION_FREQ_CRLB_SAFETY_FACTOR

        data = dict(self.uncertainty().items())
        crlb = self.crlb_frequency()
        if math.isfinite(crlb) and "frequency" in data:
            data["frequency"] = max(data["frequency"], NVISION_FREQ_CRLB_SAFETY_FACTOR * crlb)

        for name, floor in self.crlb_per_param().items():
            if name != "frequency" and name in data and math.isfinite(floor) and floor > 0:
                data[name] = max(data[name], floor)
        return ParameterValues.from_mapping(list(data.keys()), data)

    def converged(self, threshold: float) -> bool:
        # Check convergence in unit [0, 1] space. Not used by the SBED/Sobol
        # locators (they gate on _target_params_converged), but kept consistent
        # for any belief-level convergence query.
        raw_uncertainties = super()._empirical_uncertainty()
        return all(u < threshold for u in raw_uncertainties.values())

    def covariance_matrix(self) -> np.ndarray:
        """Return the physical-scale covariance matrix."""
        raw_cov = super().covariance_matrix()
        ranges = np.array(
            [self.physical_param_bounds[name][1] - self.physical_param_bounds[name][0] for name in self._param_names]
        )
        return raw_cov * np.outer(ranges, ranges)

    def sample(self, n: int) -> ParameterValues[np.ndarray]:
        return super().sample(n)

    def narrow_scan_parameter_physical_bounds(self, param_name: str, new_lo: float, new_hi: float) -> None:
        """Shrink physical bounds for ``param_name`` and remap unit particles (see grid variant)."""
        if param_name not in self.physical_param_bounds:
            raise KeyError(param_name)
        old_lo, old_hi = self.physical_param_bounds[param_name]
        w_old = old_hi - old_lo
        if w_old <= 0:
            return

        lo_orig, hi_orig = self._original_physical_x_bounds
        nl = float(max(min(new_lo, new_hi), lo_orig))
        nh = float(min(max(new_lo, new_hi), hi_orig))
        if nh <= nl:
            return

        sync_x = self.physical_x_bounds == (old_lo, old_hi)
        w_new = nh - nl

        j = self._param_names.index(param_name)
        u_col = self._particles[:, j]
        f = old_lo + u_col * w_old
        u_new = (f - nl) / w_new

        out_of_bounds = (u_new < 0.0) | (u_new > 1.0)
        n_out = int(np.sum(out_of_bounds))
        if n_out > 0:
            in_bounds_indices = np.where(~out_of_bounds)[0]
            if len(in_bounds_indices) > 0:
                replacement_indices = np.random.choice(in_bounds_indices, size=n_out, replace=True)
                # Copy entire particle rows for out-of-bounds particles to preserve joint parameter dependencies
                self._particles[out_of_bounds] = self._particles[replacement_indices]
                # Re-evaluate u_col for the narrowed scan parameter
                u_col = self._particles[:, j]
                f = old_lo + u_col * w_old
                u_new = (f - nl) / w_new
            else:
                import logging

                logging.getLogger(__name__).warning(
                    "All particles are out of bounds during narrow_scan_parameter_physical_bounds (%s, %s). "
                    "Rejecting narrowing proposal to protect filter stability.",
                    nl,
                    nh,
                )
                return

        # Verify that all particles are now within [0, 1] with a tiny tolerance for numerical precision
        tol = 1e-7
        if np.any((u_new < -tol) | (u_new > 1.0 + tol)):
            raise ValueError(
                "Particles out of bounds after resampling in "
                f"narrow_scan_parameter_physical_bounds: min {np.min(u_new)}, max {np.max(u_new)}"
            )
        self._particles[:, j] = np.clip(u_new, 0.0, 1.0)
        self._belief_version += 1

        self.model.narrow_physical_interval_for_param(param_name, nl, nh, update_x_axis=sync_x)
        self.physical_param_bounds[param_name] = (nl, nh)
        if sync_x:
            self.physical_x_bounds = (nl, nh)

    def _resample(self) -> None:
        """Perform systematic resampling and automatically narrow the frequency bounds.

        Only the scan axis (``frequency``) is narrowed. All other parameters keep
        their full physical range so the posterior can freely re-explore them once
        the true frequency is located — their optimal values may differ from the
        initial posterior mode.
        """
        super()._resample()

        # Identify the scan parameter (almost always "frequency"). Must also be a
        # free particle dimension (self._param_names) -- "frequency" stays in
        # physical_param_bounds even when fixed (it's still the probe x-axis),
        # but there's nothing to narrow from particle spread if it's not inferred.
        scan_param = "frequency" if "frequency" in self._param_names else None
        if scan_param is None:
            for name, bounds in self.physical_param_bounds.items():
                if name in self._param_names and bounds == self.physical_x_bounds:
                    scan_param = name
                    break
        if scan_param is None:
            return  # Nothing to narrow — no free frequency axis found.

        lo_phys, hi_phys = self.physical_param_bounds[scan_param]
        cur_width = hi_phys - lo_phys
        if cur_width <= 0:
            return

        # Setup dip detection parameters (lineshape-agnostic effective HWHM estimate)
        estimates = self.estimates()
        if "linewidth" in estimates:
            omega_phys = float(estimates["linewidth"])
        elif "saturation" in estimates and "sigma_inhom" in estimates:
            from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar

            fwhm_total, _, _ = _saturation_voigt_reparam_scalar(
                estimates["saturation"], estimates["sigma_inhom"], NV_SATURATION_C_MAX
            )
            omega_phys = fwhm_total / 2.0
        else:
            omega_phys = float(estimates.get("fwhm_total", 2.0e6)) / 2.0
        omega_phys = max(omega_phys, 1.0e5)  # at least 100 kHz
        zeeman_hat_phys = float(estimates.get("zeeman_split", 0.0))

        # --- Active Boundary-Escape Guard ----------------------------------
        # If particles are piling up near the unit boundaries [0.0, 1.0], it
        # means the true frequency is outside the currently narrowed window.
        # We proactively expand the physical bounds in that direction.
        j = self._param_names.index(scan_param)
        u_vals = self._particles[:, j]

        left_piling = float(np.mean(u_vals < 0.05)) > 0.15
        right_piling = float(np.mean(u_vals > 0.95)) > 0.15

        lo_orig, hi_orig = self._original_physical_x_bounds
        import logging
        import os

        if left_piling and lo_phys > lo_orig:
            expansion = max(cur_width, 10.0 * omega_phys + 2.0 * zeeman_hat_phys)
            new_lo = max(lo_phys - expansion, lo_orig)
            new_hi = hi_phys
            logging.getLogger(__name__).debug(
                "SMC particles piling up at left boundary (%.2f%%). "
                "Expanding physical scan window to the left: [%.6f, %.6f] GHz",
                float(np.mean(u_vals < 0.05)) * 100,
                new_lo / 1e9,
                new_hi / 1e9,
            )
            self._last_expansion_step = self._step_count
            self.narrow_scan_parameter_physical_bounds(scan_param, new_lo, new_hi)
            self._cached_cov = None
            self._cov_step = -1
            self._generate_epoch_candidates()
            return

        if right_piling and hi_phys < hi_orig:
            expansion = max(cur_width, 10.0 * omega_phys + 2.0 * zeeman_hat_phys)
            new_lo = lo_phys
            new_hi = min(hi_phys + expansion, hi_orig)
            logging.getLogger(__name__).debug(
                "SMC particles piling up at right boundary (%.2f%%). "
                "Expanding physical scan window to the right: [%.6f, %.6f] GHz",
                float(np.mean(u_vals > 0.95)) * 100,
                new_lo / 1e9,
                new_hi / 1e9,
            )
            self._last_expansion_step = self._step_count
            self.narrow_scan_parameter_physical_bounds(scan_param, new_lo, new_hi)
            self._cached_cov = None
            self._cov_step = -1
            self._generate_epoch_candidates()
            return

        # --- Narrowing Delay Safeguard -------------------------------------
        # Delay narrowing until we have completed a minimum number of global
        # measurements (default 8 steps) to resolve multi-modal hyperfine peak
        # ambiguity and ensure we never focus on the wrong place.
        min_narrowing_steps = int(os.getenv("NVISION_MIN_STEPS_BEFORE_NARROWING", "8"))
        if self._step_count < min_narrowing_steps:
            return

        # Also delay narrowing if we recently expanded the bounds to allow exploration
        last_exp = getattr(self, "_last_expansion_step", -1)
        if last_exp >= 0 and (self._step_count - last_exp) < min_narrowing_steps:
            return

        # --- Unified Active-Range Union Focusing (Zero Fallbacks) -----------
        # Compute believed active frequency range for each particle i:
        # [left_i, right_i] = [f_i - split_i - k * omega_i, f_i + split_i + k * omega_i]
        j_freq = self._param_names.index(scan_param)
        u_freq = self._particles[:, j_freq]
        freq_phys = lo_phys + u_freq * cur_width

        if "split" in self._param_names:
            j_split = self._param_names.index("split")
            u_split = self._particles[:, j_split]
            lo_s, hi_s = self.physical_param_bounds["split"]
            split_phys = lo_s + u_split * (hi_s - lo_s)
        else:
            split_phys = np.zeros_like(freq_phys)

        if "zeeman_split" in self._param_names:
            j_zeeman = self._param_names.index("zeeman_split")
            u_zeeman = self._particles[:, j_zeeman]
            lo_z, hi_z = self.physical_param_bounds["zeeman_split"]
            zeeman_phys = lo_z + u_zeeman * (hi_z - lo_z)
        else:
            zeeman_phys = np.zeros_like(freq_phys)

        if "linewidth" in self._param_names:
            j_line = self._param_names.index("linewidth")
            u_line = self._particles[:, j_line]
            lo_l, hi_l = self.physical_param_bounds["linewidth"]
            linewidth_phys = lo_l + u_line * (hi_l - lo_l)
        elif "saturation" in self._param_names and "sigma_inhom" in self._param_names:
            from nvision.spectra.nv_center import _saturation_voigt_reparam

            j_sat = self._param_names.index("saturation")
            j_sig = self._param_names.index("sigma_inhom")
            lo_sat, hi_sat = self.physical_param_bounds["saturation"]
            lo_sig, hi_sig = self.physical_param_bounds["sigma_inhom"]
            saturation_phys = lo_sat + self._particles[:, j_sat] * (hi_sat - lo_sat)
            sigma_inhom_phys = lo_sig + self._particles[:, j_sig] * (hi_sig - lo_sig)
            # c_max doesn't affect fwhm_total; a placeholder array satisfies the vectorized signature.
            fwhm_total, _, _ = _saturation_voigt_reparam(
                saturation_phys, sigma_inhom_phys, np.ones_like(saturation_phys)
            )
            linewidth_phys = fwhm_total / 2.0
        elif "homogeneous_linewidth" in self._param_names:
            j_hl = self._param_names.index("homogeneous_linewidth")
            u_hl = self._particles[:, j_hl]
            lo_hl, hi_hl = self.physical_param_bounds["homogeneous_linewidth"]
            linewidth_phys = lo_hl + u_hl * (hi_hl - lo_hl)
            if "sigma_inhom" in self._param_names:
                j_sig = self._param_names.index("sigma_inhom")
                u_sig = self._particles[:, j_sig]
                lo_sig, hi_sig = self.physical_param_bounds["sigma_inhom"]
                sigma_inhom_phys = lo_sig + u_sig * (hi_sig - lo_sig)
                linewidth_phys = linewidth_phys + math.sqrt(2.0 * math.log(2.0)) * sigma_inhom_phys
        elif "fwhm_total" in self._param_names:
            j_fwhm = self._param_names.index("fwhm_total")
            u_fwhm = self._particles[:, j_fwhm]
            lo_l, hi_l = self.physical_param_bounds["fwhm_total"]
            linewidth_phys = (lo_l + u_fwhm * (hi_l - lo_l)) / 2.0
        else:
            linewidth_phys = np.full_like(freq_phys, 2.0e6)

        cover_factor = float(os.getenv("NVISION_SMC_FOCUSING_COVER_FACTOR", "3.0"))
        # Rejuvenation fraction is max 5%. By using the 5th/95th percentiles, we
        # safely skip the uniform background particles while minimally eating into
        # the true dense clusters (which span ~100s of kHz, so losing 5% of their mass
        # barely moves the boundary).
        tail_percentile = 5.0

        # Per particle, the active range is exactly symmetric about freq_i: both dips
        # (or the hyperfine triplet) reach the same offset in either direction. Pool that
        # offset across *all* particles into one two-sided quantile rather than computing
        # independent one-sided quantiles of (freq_i - offset_i) and (freq_i + offset_i) --
        # the latter effectively estimates the same offset distribution twice from disjoint
        # tail subsets, and can produce an asymmetric window even though the underlying
        # physics guarantees symmetry. The frequency-center term still uses its own
        # percentiles (not the mean) so a wide or not-yet-unimodal frequency posterior is
        # still respected.
        offset_phys = zeeman_phys + split_phys + cover_factor * linewidth_phys
        offset_q95 = float(np.percentile(offset_phys, 100.0 - tail_percentile))

        freq_lo = float(np.percentile(freq_phys, tail_percentile))
        freq_hi = float(np.percentile(freq_phys, 100.0 - tail_percentile))

        new_lo = freq_lo - offset_q95
        new_hi = freq_hi + offset_q95

        # Floor: a single narrowing step must not undershoot the plausible
        # dip span, even if the frequency marginal has already collapsed.
        min_width = 2.0 * offset_q95
        if (new_hi - new_lo) < min_width:
            center = 0.5 * (new_hi + new_lo)
            new_lo = center - min_width / 2.0
            new_hi = center + min_width / 2.0

        lo_orig, hi_orig = self._original_physical_x_bounds
        new_lo = max(new_lo, lo_orig)
        new_hi = min(new_hi, hi_orig)
        if new_hi <= new_lo:
            return

        # Only apply if the window actually shrinks by at least 5%.
        min_narrowing_fraction = 0.05
        if (cur_width - (new_hi - new_lo)) / cur_width < min_narrowing_fraction:
            return

        self.narrow_scan_parameter_physical_bounds(scan_param, new_lo, new_hi)

        # Clear unit covariance cache so it's recomputed for the new unit particles.
        self._cached_cov = None
        self._cov_step = -1
        # Regenerate the epoch candidates using the updated physical frequency bounds.
        self._generate_epoch_candidates()

    def update(self, obs: Observation) -> None:
        lo_orig, hi_orig = self._original_physical_x_bounds
        lo_curr, hi_curr = self.physical_x_bounds

        if (lo_orig != lo_curr) or (hi_orig != hi_curr):
            x_phys = lo_orig + obs.x * (hi_orig - lo_orig)
            x_curr_unit = (x_phys - lo_curr) / (hi_curr - lo_curr)
            from dataclasses import replace

            obs_eval = replace(obs, x=float(x_curr_unit))
        else:
            obs_eval = obs

        super().update(obs_eval)
        # Restore the original-frame x in the history buffer and last_obs
        # (the rescaled obs_eval.x was recorded by super().update()).
        self._obs_x_arr[self._obs_count - 1] = obs.x
        self.last_obs = obs

    def batch_update(self, observations: list[Observation]) -> None:
        # NOTE: super().batch_update() appends to self._observations; do not
        # extend here as well or every observation is stored twice.
        lo_orig, hi_orig = self._original_physical_x_bounds
        lo_curr, hi_curr = self.physical_x_bounds

        observations_eval = []
        if (lo_orig != lo_curr) or (hi_orig != hi_curr):
            from dataclasses import replace

            for obs in observations:
                x_phys = lo_orig + obs.x * (hi_orig - lo_orig)
                x_curr_unit = (x_phys - lo_curr) / (hi_curr - lo_curr)
                observations_eval.append(replace(obs, x=float(x_curr_unit)))
        else:
            observations_eval = observations

        super().batch_update(observations_eval)
        # Restore the original-frame x values in the history buffer and last_obs
        # (the rescaled observations_eval x values were recorded by super()).
        n_obs = len(observations)
        if n_obs:
            self._obs_x_arr[self._obs_count - n_obs : self._obs_count] = [o.x for o in observations]
            self.last_obs = observations[-1]

    def copy(self) -> UnitCubeSMCMarginalDistribution:
        dist = UnitCubeSMCMarginalDistribution(
            model=self.model,
            parameter_bounds=self.parameter_bounds.copy(),
            num_particles=self.num_particles,
            ess_threshold=self.ess_threshold,
            a_param=self.a_param,
            last_obs=self.last_obs,
            noise_model=self.noise_model,
            auto_resample=self.auto_resample,
            resample_delay=self.resample_delay,
            physical_param_bounds=dict(self.physical_param_bounds),
            physical_x_bounds=self.physical_x_bounds,
            priors=self.priors,
            min_exploration_frac=self.min_exploration_frac,
            tempering_factor=self.tempering_factor,
            noise_discount_factor=self.noise_discount_factor,
            noise_prior_strength=self.noise_prior_strength,
            skip_state_init=True,
        )
        # Grids depend only on bounds, which are identical — share by reference
        # (consumers rebind on narrowing/resample, never mutate in place).
        dist._global_grid = self._global_grid
        dist._current_candidates = self._current_candidates
        dist._param_names = self._param_names.copy()
        dist._particles = self._particles.copy(order="K")  # preserve F-order layout
        dist._weights = self._weights.copy()
        dist._step_count = self._step_count
        dist.resampled = self.resampled
        dist._original_physical_x_bounds = self._original_physical_x_bounds
        dist._obs_x_arr = self._obs_x_arr.copy()
        dist._obs_y_arr = self._obs_y_arr.copy()
        dist._obs_count = self._obs_count
        dist._use_rao_blackwell_noise = getattr(self, "_use_rao_blackwell_noise", False)
        if getattr(self, "_use_rao_blackwell_noise", False):
            dist._noise_alphas = self._noise_alphas.copy()
            dist._noise_betas = self._noise_betas.copy()
        if hasattr(self, "_dip_centers"):
            dist._dip_centers = list(self._dip_centers)
        return dist
