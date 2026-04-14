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

import numpy as np

from nvision.belief.grid_marginal import GridParameter
from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.sim.gen.core_generators import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
)
from nvision.spectra import CompositePeakModel, GaussianModel, LorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


def _merge_phys_specs(
    specs: list[tuple[str, tuple[float, float], int]],
    overrides: Mapping[str, tuple[float, float]] | None,
) -> list[tuple[str, tuple[float, float], int]]:
    out: list[tuple[str, tuple[float, float], int]] = []
    for name, bounds, n in specs:
        lo, hi = bounds
        if overrides:
            inj = overrides.get(name)
            if inj is not None and inj[1] > inj[0]:
                lo, hi = float(inj[0]), float(inj[1])
        out.append((name, (lo, hi), n))
    return out


def _unit_cube_belief_from_specs(
    *,
    model,
    parameter_bounds: Mapping[str, tuple[float, float]] | None,
    specs: list[tuple[str, tuple[float, float], int]],
    x_param_name: str = "frequency",
) -> UnitCubeGridMarginalDistribution:
    merged = _merge_phys_specs(specs, parameter_bounds)
    phys = {name: b for name, b, _ in merged}
    x_phys = phys[x_param_name]
    wrapped = UnitCubeSignalModel(model, phys, x_phys)
    parameters = [
        GridParameter(
            name=name,
            bounds=(0.0, 1.0),
            grid=np.linspace(0.0, 1.0, n),
            posterior=np.ones(n, dtype=np.float64) / n,
        )
        for name, _, n in merged
    ]
    return UnitCubeGridMarginalDistribution(
        model=wrapped,
        parameters=parameters,
        physical_param_bounds=phys,
        physical_x_bounds=x_phys,
    )


def one_peak_gaussian_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_grid_freq: int = 120,
    n_grid_width: int = 80,
    n_grid_depth: int = 60,
    n_grid_background: int = 60,
    **_extra: object,
) -> UnitCubeGridMarginalDistribution:
    specs = [
        ("frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("sigma", (5e6, 100e6), n_grid_width),
        ("dip_depth", (0.1, 1.4), n_grid_depth),
        ("background", (0.0, 0.5), n_grid_background),
    ]
    return _unit_cube_belief_from_specs(
        model=GaussianModel(),
        parameter_bounds=parameter_bounds,
        specs=specs,
        x_param_name="frequency",
    )


def one_peak_lorentzian_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_grid_freq: int = 120,
    n_grid_width: int = 80,
    n_grid_depth: int = 60,
    n_grid_background: int = 60,
    **_extra: object,
) -> UnitCubeGridMarginalDistribution:
    specs = [
        ("frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("linewidth", (5e6, 100e6), n_grid_width),
        ("dip_depth", (0.05, 1.5), n_grid_depth),
        ("background", (0.5, 1.2), n_grid_background),
    ]
    return _unit_cube_belief_from_specs(
        model=LorentzianModel(),
        parameter_bounds=parameter_bounds,
        specs=specs,
        x_param_name="frequency",
    )


def two_peak_gaussian_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_grid_freq: int = 96,
    n_grid_width: int = 64,
    n_grid_depth: int = 48,
    n_grid_background: int = 48,
    **_extra: object,
) -> UnitCubeGridMarginalDistribution:
    model = CompositePeakModel([("peak1", GaussianModel()), ("peak2", GaussianModel())])
    specs = [
        ("peak1_frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("peak1_sigma", (5e6, 100e6), n_grid_width),
        ("peak1_dip_depth", (0.1, 1.4), n_grid_depth),
        ("peak1_background", (0.0, 0.5), n_grid_background),
        ("peak2_frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("peak2_sigma", (5e6, 100e6), n_grid_width),
        ("peak2_dip_depth", (0.1, 1.4), n_grid_depth),
        ("peak2_background", (0.0, 0.5), n_grid_background),
    ]
    return _unit_cube_belief_from_specs(
        model=model,
        parameter_bounds=parameter_bounds,
        specs=specs,
        x_param_name="peak1_frequency",
    )


def two_peak_lorentzian_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_grid_freq: int = 96,
    n_grid_width: int = 64,
    n_grid_depth: int = 48,
    n_grid_background: int = 48,
    **_extra: object,
) -> UnitCubeGridMarginalDistribution:
    model = CompositePeakModel([("peak1", LorentzianModel()), ("peak2", LorentzianModel())])
    specs = [
        ("peak1_frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("peak1_linewidth", (5e6, 100e6), n_grid_width),
        ("peak1_dip_depth", (0.05, 1.5), n_grid_depth),
        ("peak1_background", (0.0, 0.5), n_grid_background),
        ("peak2_frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("peak2_linewidth", (5e6, 100e6), n_grid_width),
        ("peak2_dip_depth", (0.05, 1.5), n_grid_depth),
        ("peak2_background", (0.0, 0.5), n_grid_background),
    ]
    return _unit_cube_belief_from_specs(
        model=model,
        parameter_bounds=parameter_bounds,
        specs=specs,
        x_param_name="peak1_frequency",
    )


def nv_center_one_peak_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_grid_freq: int = 500,
    n_grid_linewidth: int = 80,
    n_grid_depth: int = 100,
    n_grid_background: int = 60,
    **_extra: object,
) -> UnitCubeGridMarginalDistribution:
    """NV-center single-peak (zero-field) belief: 4 parameters, no split or k_np."""
    from nvision.spectra.nv_center import (
        NVCenterOnePeakLorentzianModel,
        nv_center_one_peak_lorentzian_bounds_for_domain,
    )

    model = NVCenterOnePeakLorentzianModel()
    phys = nv_center_one_peak_lorentzian_bounds_for_domain(DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX)
    base_specs: list[tuple[str, tuple[float, float], int]] = [
        ("frequency", phys["frequency"], n_grid_freq),
        ("linewidth", phys["linewidth"], n_grid_linewidth),
        ("dip_depth", phys["dip_depth"], n_grid_depth),
        ("background", phys["background"], n_grid_background),
    ]
    return _unit_cube_belief_from_specs(
        model=model,
        parameter_bounds=parameter_bounds,
        specs=base_specs,
        x_param_name="frequency",
    )


def nv_center_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_grid_freq: int = 500,
    n_grid_linewidth: int = 80,
    n_grid_fwhm_total: int = 80,
    n_grid_lorentz_frac: int = 60,
    n_grid_split: int = 80,
    n_grid_k_np: int = 60,
    n_grid_depth: int = 100,
    n_grid_background: int = 60,
    **_extra: object,
) -> UnitCubeGridMarginalDistribution:
    """NV-center belief: **unit** parameter grids, **physical** signal model.

    Automatically detects if it should use NVCenterLorentzianModel or NVCenterVoigtModel
    based on the presence of 'fwhm_total' in the required parameter set.
    """
    from nvision.spectra.nv_center import (
        NVCenterLorentzianModel,
        NVCenterVoigtModel,
        nv_center_lorentzian_bounds_for_domain,
        nv_center_voigt_bounds_for_domain,
    )

    is_voigt = "fwhm_total" in (parameter_bounds or {}) or n_grid_fwhm_total != 80 or "lorentz_frac" in _extra

    if is_voigt:
        model = NVCenterVoigtModel()
        phys = nv_center_voigt_bounds_for_domain(DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX)
        base_specs: list[tuple[str, tuple[float, float], int]] = [
            ("frequency", phys["frequency"], n_grid_freq),
            ("fwhm_total", phys["fwhm_total"], n_grid_fwhm_total),
            ("lorentz_frac", phys["lorentz_frac"], n_grid_lorentz_frac),
            ("split", phys["split"], n_grid_split),
            ("k_np", phys["k_np"], n_grid_k_np),
            ("dip_depth", phys["dip_depth"], n_grid_depth),
            ("background", phys["background"], n_grid_background),
        ]
    else:
        model = NVCenterLorentzianModel()
        phys = nv_center_lorentzian_bounds_for_domain(DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX)
        base_specs = [
            ("frequency", phys["frequency"], n_grid_freq),
            ("linewidth", phys["linewidth"], n_grid_linewidth),
            ("split", phys["split"], n_grid_split),
            ("k_np", phys["k_np"], n_grid_k_np),
            ("dip_depth", phys["dip_depth"], n_grid_depth),
            ("background", phys["background"], n_grid_background),
        ]

    return _unit_cube_belief_from_specs(
        model=model,
        parameter_bounds=parameter_bounds,
        specs=base_specs,
        x_param_name="frequency",
    )


def nv_center_one_peak_smc_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    num_particles: int = 5000,
    jitter_scale: float = 0.05,
    ess_threshold: float = 0.5,
    **_extra: object,
) -> UnitCubeSMCMarginalDistribution:
    """NV-center single-peak (zero-field) belief: 4 parameters, SMC particles."""
    from nvision.spectra.nv_center import (
        NVCenterOnePeakLorentzianModel,
        nv_center_one_peak_lorentzian_bounds_for_domain,
    )

    model = NVCenterOnePeakLorentzianModel()
    merged_bounds = nv_center_one_peak_lorentzian_bounds_for_domain(
        DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX
    )

    if parameter_bounds:
        for name in merged_bounds:
            if name in parameter_bounds and parameter_bounds[name][1] > parameter_bounds[name][0]:
                merged_bounds[name] = parameter_bounds[name]

    # Enforce dip_depth floor so the posterior cannot collapse to "flat signal".
    if "dip_depth" in merged_bounds:
        d_lo, d_hi = merged_bounds["dip_depth"]
        merged_bounds["dip_depth"] = (max(float(d_lo), 0.05), float(d_hi))

    x_phys = merged_bounds["frequency"]
    wrapped = UnitCubeSignalModel(model, merged_bounds, x_phys)

    return UnitCubeSMCMarginalDistribution(
        model=wrapped,
        parameter_bounds={name: (0.0, 1.0) for name in merged_bounds},
        num_particles=num_particles,
        jitter_scale=jitter_scale,
        ess_threshold=ess_threshold,
        physical_param_bounds=merged_bounds,
        physical_x_bounds=x_phys,
    )


def nv_center_smc_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    num_particles: int = 5000,
    jitter_scale: float = 0.05,
    ess_threshold: float = 0.5,
    **_extra: object,
) -> UnitCubeSMCMarginalDistribution:
    """NV-center belief: **unit** parameter particles, **physical** signal model."""
    from nvision.spectra.nv_center import (
        NVCenterLorentzianModel,
        NVCenterVoigtModel,
        nv_center_lorentzian_bounds_for_domain,
        nv_center_voigt_bounds_for_domain,
    )

    is_voigt = "fwhm_lorentz" in (parameter_bounds or {})

    if is_voigt:
        model = NVCenterVoigtModel()
        merged_bounds = nv_center_voigt_bounds_for_domain(DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX)
    else:
        model = NVCenterLorentzianModel()
        merged_bounds = nv_center_lorentzian_bounds_for_domain(
            DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX
        )

    if parameter_bounds:
        for name in merged_bounds:
            if name in parameter_bounds and parameter_bounds[name][1] > parameter_bounds[name][0]:
                merged_bounds[name] = parameter_bounds[name]

    # Enforce dip_depth floor so the posterior cannot collapse to "flat signal".
    if "dip_depth" in merged_bounds:
        d_lo, d_hi = merged_bounds["dip_depth"]
        merged_bounds["dip_depth"] = (max(float(d_lo), 0.05), float(d_hi))

    x_phys = merged_bounds["frequency"]
    wrapped = UnitCubeSignalModel(model, merged_bounds, x_phys)

    return UnitCubeSMCMarginalDistribution(
        model=wrapped,
        parameter_bounds={name: (0.0, 1.0) for name in merged_bounds},
        num_particles=num_particles,
        jitter_scale=jitter_scale,
        ess_threshold=ess_threshold,
        physical_param_bounds=merged_bounds,
        physical_x_bounds=x_phys,
    )
