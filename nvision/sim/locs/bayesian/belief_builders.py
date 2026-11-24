"""Belief builders — callables that construct a MarginalDistribution.

A builder is any callable with the signature::

    (parameter_bounds?, **grid_config) -> AbstractMarginalDistribution

No base class required.  The acquisition locators accept one and call it
at creation time.

All Bayesian builders below use a **unit cube** in parameter space: each
marginal prior is uniform on ``[0, 1]``, while
:class:`~nvision.spectra.unit_cube_model.UnitCubeSignalModel` maps drive frequencies
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
    PRIOR_BROAD_FRACTION,
    SMCMarginalDistribution,
)
from nvision.sim.gen.nv_center_generator import (
    DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
    DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
)
from nvision.spectra.drive import DriveSetting
from nvision.spectra.noise_model import NoiseSignalModel
from nvision.spectra.unit_cube import UnitCubeSignalModel

# Half-width, in prior standard deviations, of the box a known parameter is confined to (:func:`apply_known_prior`).
# 4 sigma keeps 99.99 % of the normal's mass, so the truncation is negligible; it also keeps the prior's broad
# (uniform) particles inside that box.
KNOWN_PRIOR_BOUND_SIGMAS: float = 4.0

# Line-shape parameters a known shape fixes, whichever of them the belief has: the contrast and the line widths
# (``hwhm`` for a Lorentzian belief, ``hwhm_lorentz``/``sigma_inhom`` for a Voigt one).
KNOWN_SHAPE_WIDTH_PARAMS: tuple[str, ...] = ("hwhm", "hwhm_lorentz", "sigma_inhom")
KNOWN_SHAPE_PARAMS: tuple[str, ...] = ("contrast_total", *KNOWN_SHAPE_WIDTH_PARAMS)

# How well a known shape is known (Bayesian-SBED-KnownShape): its prior std as a fraction of the true contrast and of
# the true total HWHM. The lab holds them exactly; 1 % keeps the box non-empty while the widths' and the contrast's
# share of the predictive variance (the acquisition score) stays negligible against the centre's.
KNOWN_SHAPE_STD_FRACTION: float = 0.01


def apply_known_prior(bounds: dict[str, object], name: str, value_phys: float, std_phys: float) -> None:
    """Make *bounds* carry a known prior ``N(value_phys, std_phys)`` on parameter *name*, in place.

    The bound becomes ``value +- KNOWN_PRIOR_BOUND_SIGMAS * std`` intersected with the parameter's own bound (its
    support: a width cannot go below 0), and ``bounds["_priors"][name] = (value, std)``. The ``"_priors"`` dict is
    copied, never mutated, since callers share it. *bounds* is the physical bounds dict of
    :func:`nv_center_smc_belief` (Hz for widths, dimensionless for the contrast).

    Raises ``ValueError`` if *name* is not in *bounds*, ``std_phys <= 0``, the value lies outside the bound, or the
    box would be empty.
    """
    if name not in bounds:
        raise ValueError(
            f"A known {name!r} needs a model with that free parameter; "
            f"this one has {sorted(k for k in bounds if not k.startswith('_'))}."
        )
    v_phys = float(value_phys)
    std_phys = float(std_phys)
    if not std_phys > 0.0:
        raise ValueError(f"Known {name} std must be > 0, got {std_phys!r}.")
    lo_phys, hi_phys = (float(b) for b in bounds[name])
    if not lo_phys <= v_phys <= hi_phys:
        raise ValueError(f"Known {name} {v_phys} lies outside its bound ({lo_phys}, {hi_phys}).")
    box_lo_phys = max(lo_phys, v_phys - KNOWN_PRIOR_BOUND_SIGMAS * std_phys)
    box_hi_phys = min(hi_phys, v_phys + KNOWN_PRIOR_BOUND_SIGMAS * std_phys)
    if not box_hi_phys > box_lo_phys:
        raise ValueError(f"Known {name} {v_phys} +- {std_phys} leaves an empty box ({box_lo_phys}, {box_hi_phys}).")
    bounds[name] = (box_lo_phys, box_hi_phys)
    priors = dict(bounds.get("_priors") or {})
    priors[name] = (v_phys, std_phys)
    bounds["_priors"] = priors


def apply_known_shape(bounds: dict[str, object], known_shape_phys: Mapping[str, tuple[float, float]]) -> None:
    """Apply :func:`apply_known_prior` for every entry of *known_shape_phys* the belief has, in place.

    *known_shape_phys*: ``{name: (value_phys, std_phys)}`` over :data:`KNOWN_SHAPE_PARAMS`; it may name widths under
    both the Lorentzian and the Voigt names, and only those *bounds* carries are applied. Raises ``ValueError`` on a
    name outside :data:`KNOWN_SHAPE_PARAMS`, or when the contrast or every width is left free (a half-known shape is
    not what was asked for).
    """
    unknown = sorted(set(known_shape_phys) - set(KNOWN_SHAPE_PARAMS))
    if unknown:
        raise ValueError(f"known_shape_phys names {unknown}, not line-shape parameters {KNOWN_SHAPE_PARAMS}.")
    applied = [name for name in known_shape_phys if name in bounds]
    if "contrast_total" not in applied or not any(name in applied for name in KNOWN_SHAPE_WIDTH_PARAMS):
        raise ValueError(
            f"A known shape needs the contrast and a width in the belief; given {sorted(known_shape_phys)}, "
            f"belief has {sorted(k for k in bounds if not k.startswith('_'))}."
        )
    for name in applied:
        apply_known_prior(bounds, name, *known_shape_phys[name])


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
    from nvision.spectra.nv_center_drive import NVCenterCWDriveModel, NVCenterPulsedDriveModel

    if isinstance(model, NVCenterCWDriveModel):
        return "cw_drive"
    if isinstance(model, NVCenterPulsedDriveModel):
        return "pulsed_drive"
    if isinstance(model, NVCenterSaturationVoigtModel):
        return "saturation_voigt"
    if isinstance(model, NVCenterVoigtModel):
        return "voigt"
    return "lorentzian"


def nv_center_smc_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    noise_model: NoiseSignalModel,
    num_particles: int = NVISION_SMC_NUM_PARTICLES,
    ess_threshold: float = NVISION_SMC_ESS_THRESHOLD,
    a_param: float = NVISION_SMC_A_PARAM,
    min_exploration_frac: float = NVISION_SMC_MIN_EXPLORATION_FRAC,
    prior_broad_fraction: float = PRIOR_BROAD_FRACTION,
    tempering_factor: float = NVISION_SMC_TEMPERING_FACTOR,
    hyperfine: str = "unresolved",
    infer_hyperfine: bool = False,
    with_zeeman_splitting: bool = True,
    with_fixed_zero_field_freq: bool = True,
    lineshape: str = "lorentzian",
    seed: int | None = None,
    fixed_drive: DriveSetting | None = None,
    voigt_bounds_as_lorentzian: bool = False,
    known_shape_phys: Mapping[str, tuple[float, float]] | None = None,
) -> SMCMarginalDistribution:
    """NV-center belief: **unit** parameter particles, **physical** signal model.

    By default uses Zeeman splitting (two dips) with the hyperfine structure
    *unresolved* -- one merged dip per group, which is what a typical NV
    hwhm gives. The belief's ``hyperfine``/``infer_hyperfine`` must match
    how the true signal was generated: modelling a resolved ¹⁴N triplet against
    a signal whose lines are merged (or vice versa) makes the locator hunt for
    peaks the data never showed. Set ``hyperfine="n14"``/``"n15"`` when the lines
    really are resolved, and ``infer_hyperfine=True`` to additionally infer
    ``hyperfine_shift`` and ``k_np`` rather than fixing them to the isotope's coupling.
    ``with_fixed_zero_field_freq`` (default ``True``, matching every ``NVCenter*Model``'s own
    default) treats the zero-field ``zero_field_freq`` as a known instrument constant
    rather than a free particle dimension -- pass ``False`` to infer it instead (e.g.
    for tests/locators that specifically exercise zero_field_freq estimation).

    ``lineshape`` selects the signal model:

    * ``"lorentzian"`` (default) — :class:`~nvision.spectra.nv_center.NVCenterLorentzianModel`.
    * ``"voigt"`` — :class:`~nvision.spectra.nv_center.NVCenterVoigtModel`, inferring
      physically-decomposed ``hwhm_lorentz``/``sigma_inhom`` (reparameterized to
      the kernel-native ``fwhm_total``/``lorentz_frac`` internally) and population-normalized
      ``contrast_total``. Respects ``hyperfine``/``infer_hyperfine``/``with_zeeman_splitting``.
    * ``"saturation_voigt"`` — :class:`~nvision.spectra.nv_center.NVCenterSaturationVoigtModel`,
      which replaces the lumped hwhm with two physically distinct, separately
      inferred broadening parameters: ``saturation`` (drive power, sets the
      homogeneous/power-broadened width *and* the realized contrast together via
      the saturation law) and ``sigma_inhom`` (independent inhomogeneous/Gaussian
      width). Respects ``hyperfine``/``infer_hyperfine``/``with_zeeman_splitting``.
    * ``"cw_drive"`` / ``"pulsed_drive"`` — the drive-controlled CW and pulsed models of
      :mod:`nvision.spectra.nv_center_drive` (inferring ``drive_amp_sat`` / ``rabi_per_amp``).
      Respect ``hyperfine``/``with_zeeman_splitting``; ``infer_hyperfine`` is not supported.

    ``fixed_drive`` (drive-controlled lineshapes only) builds the belief on the model's fixed-drive
    view: for locators that do not choose the drive, every measurement is taken at that one drive.
    Without it a drive-controlled belief needs every observation's drive (drive-choosing locators).

    ``voigt_bounds_as_lorentzian`` (``lineshape="lorentzian"`` only): ``parameter_bounds`` come from a Voigt signal
    with no Gaussian part (the SNR x window study at Gaussian share 0), which is exactly a Lorentzian whose ``hwhm`` is
    the Voigt's ``hwhm_lorentz``. That width's bounds and prior are read under the Lorentzian name; the Voigt's
    ``sigma_inhom`` is not a belief dimension. Raises if the bounds carry no ``hwhm_lorentz``.

    ``known_shape_phys`` = ``{name: (value, std)}`` declares the line shape known, as the lab's Bayesian ESR runs it:
    the contrast and the widths the belief has get the narrow normal priors of :func:`apply_known_shape`, applied after
    the Voigt-to-Lorentzian renaming so the belief's own names are matched. ``None`` (default) keeps the bounds and
    priors given in *parameter_bounds*.
    """
    if noise_model is None:
        raise ValueError(
            "nv_center_smc_belief requires a noise model: the belief always infers the noise level through "
            "its conjugate prior (experiments without noise cannot be run with a Bayesian belief)."
        )

    from nvision.spectra.nv_center import (
        NVCenterLorentzianModel,
        NVCenterSaturationVoigtModel,
        NVCenterVoigtModel,
        nv_center_lorentzian_bounds_for_domain,
        nv_center_saturation_voigt_bounds_for_domain,
        nv_center_voigt_bounds_for_domain,
    )

    if lineshape in ("cw_drive", "pulsed_drive"):
        from nvision.spectra.nv_center_drive import (
            NVCenterCWDriveModel,
            NVCenterPulsedDriveModel,
            nv_center_drive_bounds_for_domain,
        )

        if infer_hyperfine:
            raise ValueError(f"lineshape {lineshape!r} does not support infer_hyperfine=True.")
        drive_model_cls = NVCenterCWDriveModel if lineshape == "cw_drive" else NVCenterPulsedDriveModel
        model = drive_model_cls(
            hyperfine=hyperfine,
            with_zeeman_splitting=with_zeeman_splitting,
            with_fixed_zero_field_freq=with_fixed_zero_field_freq,
        )
        merged_bounds = nv_center_drive_bounds_for_domain(
            "cw" if lineshape == "cw_drive" else "pulsed",
            DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
            DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
            hyperfine=hyperfine,
            with_zeeman_splitting=with_zeeman_splitting,
        )
        if fixed_drive is not None:
            model = model.at_drive(fixed_drive)
    elif fixed_drive is not None:
        raise ValueError(f"fixed_drive is only meaningful for a drive-controlled lineshape; got {lineshape!r}.")
    elif lineshape == "saturation_voigt":
        model = NVCenterSaturationVoigtModel(
            hyperfine=hyperfine,
            infer_hyperfine=infer_hyperfine,
            with_zeeman_splitting=with_zeeman_splitting,
            with_fixed_zero_field_freq=with_fixed_zero_field_freq,
        )
        merged_bounds = nv_center_saturation_voigt_bounds_for_domain(
            DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
            DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
            hyperfine=hyperfine,
            infer_hyperfine=infer_hyperfine,
            with_zeeman_splitting=with_zeeman_splitting,
        )
    elif lineshape == "voigt":
        model = NVCenterVoigtModel(
            hyperfine=hyperfine,
            infer_hyperfine=infer_hyperfine,
            with_zeeman_splitting=with_zeeman_splitting,
            with_fixed_zero_field_freq=with_fixed_zero_field_freq,
        )
        merged_bounds = nv_center_voigt_bounds_for_domain(
            DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
            DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
            hyperfine=hyperfine,
            infer_hyperfine=infer_hyperfine,
            with_zeeman_splitting=with_zeeman_splitting,
        )
    else:
        model = NVCenterLorentzianModel(
            hyperfine=hyperfine,
            infer_hyperfine=infer_hyperfine,
            with_zeeman_splitting=with_zeeman_splitting,
            with_fixed_zero_field_freq=with_fixed_zero_field_freq,
        )
        merged_bounds = nv_center_lorentzian_bounds_for_domain(
            DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
            DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
            hyperfine=hyperfine,
            infer_hyperfine=infer_hyperfine,
            with_zeeman_splitting=with_zeeman_splitting,
        )

    if voigt_bounds_as_lorentzian:
        if lineshape != "lorentzian":
            raise ValueError(f"voigt_bounds_as_lorentzian needs lineshape='lorentzian', got {lineshape!r}.")
        if not parameter_bounds or "hwhm_lorentz" not in parameter_bounds:
            raise ValueError("voigt_bounds_as_lorentzian needs Voigt parameter bounds carrying 'hwhm_lorentz'.")
        parameter_bounds = {("hwhm" if k == "hwhm_lorentz" else k): v for k, v in parameter_bounds.items()}
        if "_priors" in parameter_bounds:
            parameter_bounds["_priors"] = {
                ("hwhm" if k == "hwhm_lorentz" else k): v for k, v in parameter_bounds["_priors"].items()
            }

    if parameter_bounds:
        for name in merged_bounds:
            if name in parameter_bounds and parameter_bounds[name][1] > parameter_bounds[name][0]:
                merged_bounds[name] = parameter_bounds[name]
        # Preserve priors if passed in
        if "_priors" in parameter_bounds:
            merged_bounds["_priors"] = parameter_bounds["_priors"]

    if known_shape_phys is not None:
        apply_known_shape(merged_bounds, known_shape_phys)

    # Merge noise parameter bounds
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

    # The NV bound builders make the zero_field_freq prior range span exactly the drive-frequency axis.
    drive_freq_bounds_phys = merged_bounds["zero_field_freq"]
    wrapped = UnitCubeSignalModel(model, merged_bounds, drive_freq_bounds_phys)

    return SMCMarginalDistribution(
        model=wrapped,
        parameter_bounds={name: (0.0, 1.0) for name in merged_bounds},
        num_particles=num_particles,
        ess_threshold=ess_threshold,
        a_param=a_param,
        noise_model=noise_model,
        physical_param_bounds=merged_bounds,
        drive_freq_bounds_phys=drive_freq_bounds_phys,
        priors=unit_priors,
        min_exploration_frac=min_exploration_frac,
        prior_broad_fraction=prior_broad_fraction,
        tempering_factor=tempering_factor,
        seed=seed,
    )
