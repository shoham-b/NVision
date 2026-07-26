"""Belief builders — callables that construct a MarginalDistribution.

A builder is any callable with the signature::

    (parameter_bounds?, **grid_config) -> AbstractMarginalDistribution

No base class required.  The acquisition locators accept one and call it
at creation time.

All Bayesian builders below use a **unit cube** in parameter space: each
marginal prior is uniform on ``[0, 1]``, while
:class:`~nvision.spectra.unit_cube_model.UnitCubeSignalModel` maps probe position
and parameters into physical units for forward-model likelihood evaluation.
That keeps acquisition / convergence thresholds comparable across parameters
while predictions stay on the same scale as measured signals.
"""

from __future__ import annotations

from collections.abc import Mapping

from nvision.belief.smc_marginal import (
    NVISION_SMC_A_PARAM,
    NVISION_SMC_ESS_THRESHOLD,
    NVISION_SMC_MIN_EXPLORATION_FRAC,
    NVISION_SMC_NUM_PARTICLES,
    NVISION_SMC_TEMPERING_FACTOR,
)
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.sim.gen.nv_center_generator import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
)
from nvision.spectra.noise_model import NoiseSignalModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


def nv_lineshape_for_model(model: object) -> str:
    """Map a true-signal model instance to the ``lineshape`` name :func:`nv_center_smc_belief` expects.

    Baseline builders (Sobol/SimpleSweep, in ``runner/executor.py`` and
    ``runner/plots.py``) construct their own belief from scratch rather than going
    through ``CombinationGrid.strategies_for()``'s ``nv_smc_config``, so without this
    they silently got ``nv_center_smc_belief``'s "lorentzian" default -- mismatching
    the actual generated signal for voigt/saturation_voigt runs and either crashing
    the fit (SimpleSweep) or running Bayesian updates with the wrong forward model
    (Sobol).
    """
    from nvision.spectra.nv_center import NVCenterSaturationVoigtModel, NVCenterVoigtModel

    if isinstance(model, NVCenterSaturationVoigtModel):
        return "saturation_voigt"
    if isinstance(model, NVCenterVoigtModel):
        return "voigt"
    return "lorentzian"


def nv_center_smc_belief(  # noqa: C901
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    num_particles: int = NVISION_SMC_NUM_PARTICLES,
    ess_threshold: float = NVISION_SMC_ESS_THRESHOLD,
    a_param: float = NVISION_SMC_A_PARAM,
    noise_model: NoiseSignalModel | None = None,
    min_exploration_frac: float = NVISION_SMC_MIN_EXPLORATION_FRAC,
    tempering_factor: float = NVISION_SMC_TEMPERING_FACTOR,
    with_hyperfine_splitting: bool = False,
    with_zeeman_splitting: bool = True,
    with_fixed_frequency: bool = True,
    lineshape: str = "lorentzian",
    **_extra: object,
) -> UnitCubeSMCMarginalDistribution:
    """NV-center belief: **unit** parameter particles, **physical** signal model.

    By default uses Zeeman splitting (two dips). Set ``with_zeeman_splitting=False``
    for a single-dip model. Set ``with_hyperfine_splitting=True`` to also infer split and k_np.
    ``with_fixed_frequency=True`` (default) fixes the center frequency at the known
    zero-field-splitting constant instead of treating it as a free/inferred particle
    dimension (see ``NVCenterLorentzianModel``); set to ``False`` to infer it.

    ``lineshape`` selects the signal model:

    * ``"lorentzian"`` (default) — :class:`~nvision.spectra.nv_center.NVCenterLorentzianModel`.
    * ``"voigt"`` — :class:`~nvision.spectra.nv_center.NVCenterVoigtModel`, inferring
      physically-decomposed ``homogeneous_linewidth``/``sigma_inhom`` (reparameterized to
      the kernel-native ``fwhm_total``/``lorentz_frac`` internally) and population-normalized
      ``c_total``. Respects ``with_hyperfine_splitting``/``with_zeeman_splitting``.
    * ``"saturation_voigt"`` — :class:`~nvision.spectra.nv_center.NVCenterSaturationVoigtModel`,
      which replaces the lumped linewidth with two physically distinct, separately
      inferred broadening parameters: ``saturation`` (drive power, sets the
      homogeneous/power-broadened width *and* the realized contrast together via
      the saturation law) and ``sigma_inhom`` (independent inhomogeneous/Gaussian
      width). Respects ``with_hyperfine_splitting``/``with_zeeman_splitting``.
    """
    from nvision.spectra.nv_center import (
        NVCenterLorentzianModel,
        NVCenterSaturationVoigtModel,
        NVCenterVoigtModel,
        nv_center_lorentzian_bounds_for_domain,
        nv_center_saturation_voigt_bounds_for_domain,
        nv_center_voigt_bounds_for_domain,
    )

    if lineshape == "saturation_voigt":
        model = NVCenterSaturationVoigtModel(
            with_hyperfine_splitting=with_hyperfine_splitting,
            with_zeeman_splitting=with_zeeman_splitting,
            with_fixed_frequency=with_fixed_frequency,
        )
        merged_bounds = nv_center_saturation_voigt_bounds_for_domain(
            DEFAULT_NV_CENTER_FREQ_X_MIN,
            DEFAULT_NV_CENTER_FREQ_X_MAX,
            with_hyperfine_splitting=with_hyperfine_splitting,
            with_zeeman_splitting=with_zeeman_splitting,
        )
    elif lineshape == "voigt":
        model = NVCenterVoigtModel(
            with_hyperfine_splitting=with_hyperfine_splitting,
            with_zeeman_splitting=with_zeeman_splitting,
            with_fixed_frequency=with_fixed_frequency,
        )
        merged_bounds = nv_center_voigt_bounds_for_domain(
            DEFAULT_NV_CENTER_FREQ_X_MIN,
            DEFAULT_NV_CENTER_FREQ_X_MAX,
            with_hyperfine_splitting=with_hyperfine_splitting,
            with_zeeman_splitting=with_zeeman_splitting,
        )
    else:
        model = NVCenterLorentzianModel(
            with_hyperfine_splitting=with_hyperfine_splitting,
            with_zeeman_splitting=with_zeeman_splitting,
            with_fixed_frequency=with_fixed_frequency,
        )
        merged_bounds = nv_center_lorentzian_bounds_for_domain(
            DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX,
            with_hyperfine_splitting=with_hyperfine_splitting,
            with_zeeman_splitting=with_zeeman_splitting,
        )

    if parameter_bounds:
        for name in merged_bounds:
            if name in parameter_bounds and parameter_bounds[name][1] > parameter_bounds[name][0]:
                merged_bounds[name] = parameter_bounds[name]
        # Preserve priors if passed in
        if "_priors" in parameter_bounds:
            merged_bounds["_priors"] = parameter_bounds["_priors"]

    # Enforce dip_depth floor so the posterior cannot collapse to "flat signal".
    if "dip_depth" in merged_bounds:
        d_lo, d_hi = merged_bounds["dip_depth"]
        merged_bounds["dip_depth"] = (max(float(d_lo), 0.05), float(d_hi))

    # Merge noise parameter bounds
    if noise_model is not None:
        noise_spec = noise_model.spec
        for name in noise_spec.names:
            if parameter_bounds and name in parameter_bounds:
                merged_bounds[name] = parameter_bounds[name]
            else:
                merged_bounds[name] = noise_spec.bounds[name]

    # Extract priors if available
    phys_priors = merged_bounds.pop("_priors", None)
    unit_priors = None
    if phys_priors:
        unit_priors = {}
        for name, prior_val in phys_priors.items():
            if name in merged_bounds:
                if isinstance(prior_val, tuple) and len(prior_val) >= 2 and prior_val[0] == "sin^2":
                    unit_priors[name] = prior_val
                else:
                    mu, std = prior_val
                    lo, hi = merged_bounds[name]
                    unit_mu = (mu - lo) / (hi - lo)
                    unit_std = std / (hi - lo)
                    unit_priors[name] = (float(unit_mu), float(unit_std))

    x_phys = merged_bounds["frequency"]
    wrapped = UnitCubeSignalModel(model, merged_bounds, x_phys)

    return UnitCubeSMCMarginalDistribution(
        model=wrapped,
        parameter_bounds={name: (0.0, 1.0) for name in merged_bounds},
        num_particles=num_particles,
        ess_threshold=ess_threshold,
        a_param=a_param,
        noise_model=noise_model,
        physical_param_bounds=merged_bounds,
        physical_x_bounds=x_phys,
        priors=unit_priors,
        min_exploration_frac=min_exploration_frac,
        tempering_factor=tempering_factor,
    )
