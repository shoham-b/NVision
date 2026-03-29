"""Belief builders — callables that construct a BeliefDistribution.

A builder is any callable with the signature::

    (parameter_bounds?, **grid_config) -> AbstractBeliefDistribution

No base class required.  The acquisition locators accept one and call it
at creation time.

All Bayesian builders below use a **unit cube** in parameter space: each
marginal prior is uniform on ``[0, 1]``, while
:class:`~nvision.signal.unit_cube_model.UnitCubeSignalModel` maps probe position
and parameters into physical units for forward-model likelihood evaluation.
That keeps acquisition / convergence thresholds comparable across parameters
while predictions stay on the same scale as measured signals.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from nvision.belief.grid_belief import GridParameter
from nvision.belief.unit_cube_grid_belief import UnitCubeGridBeliefDistribution
from nvision.belief.unit_cube_smc_belief import UnitCubeSMCBeliefDistribution
from nvision.signal import CompositePeakModel, GaussianModel, LorentzianModel
from nvision.signal.unit_cube import UnitCubeSignalModel
from nvision.sim.gen.core_generators import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    nv_center_lorentzian_bounds_for_domain,
)


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
) -> UnitCubeGridBeliefDistribution:
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
    return UnitCubeGridBeliefDistribution(
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
    n_grid_amplitude: int = 60,
    n_grid_background: int = 60,
    **_extra: object,
) -> UnitCubeGridBeliefDistribution:
    specs = [
        ("frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("sigma", (5e6, 100e6), n_grid_width),
        ("amplitude", (0.1, 1.4), n_grid_amplitude),
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
    n_grid_amplitude: int = 60,
    n_grid_background: int = 60,
    **_extra: object,
) -> UnitCubeGridBeliefDistribution:
    amp_hi = (0.2 * (3.1e9 - 2.6e9)) ** 2
    # Keep a non-zero minimum contrast so inference does not collapse to
    # "flat signal" and overfit measurement noise.
    amp_lo = 0.01 * amp_hi
    specs = [
        ("frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("linewidth", (5e6, 100e6), n_grid_width),
        ("amplitude", (amp_lo, amp_hi), n_grid_amplitude),
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
    n_grid_amplitude: int = 48,
    n_grid_background: int = 48,
    **_extra: object,
) -> UnitCubeGridBeliefDistribution:
    model = CompositePeakModel([("peak1", GaussianModel()), ("peak2", GaussianModel())])
    specs = [
        ("peak1_frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("peak1_sigma", (5e6, 100e6), n_grid_width),
        ("peak1_amplitude", (0.1, 1.4), n_grid_amplitude),
        ("peak1_background", (0.0, 0.5), n_grid_background),
        ("peak2_frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("peak2_sigma", (5e6, 100e6), n_grid_width),
        ("peak2_amplitude", (0.1, 1.4), n_grid_amplitude),
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
    n_grid_amplitude: int = 48,
    n_grid_background: int = 48,
    **_extra: object,
) -> UnitCubeGridBeliefDistribution:
    model = CompositePeakModel([("peak1", LorentzianModel()), ("peak2", LorentzianModel())])
    amp_hi = (0.2 * (3.1e9 - 2.6e9)) ** 2
    # Keep non-zero peak contrast so Bayesian updates cannot explain data with
    # two effectively flat peaks plus noise.
    amp_lo = 0.01 * amp_hi
    specs = [
        ("peak1_frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("peak1_linewidth", (5e6, 100e6), n_grid_width),
        ("peak1_amplitude", (amp_lo, amp_hi), n_grid_amplitude),
        ("peak1_background", (0.0, 0.5), n_grid_background),
        ("peak2_frequency", (2.6e9, 3.1e9), n_grid_freq),
        ("peak2_linewidth", (5e6, 100e6), n_grid_width),
        ("peak2_amplitude", (amp_lo, amp_hi), n_grid_amplitude),
        ("peak2_background", (0.0, 0.5), n_grid_background),
    ]
    return _unit_cube_belief_from_specs(
        model=model,
        parameter_bounds=parameter_bounds,
        specs=specs,
        x_param_name="peak1_frequency",
    )


def nv_center_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    n_grid_freq: int = 120,
    n_grid_linewidth: int = 60,
    n_grid_split: int = 60,
    n_grid_k_np: int = 40,
    n_grid_amplitude: int = 40,
    n_grid_background: int = 40,
    **_extra: object,
) -> UnitCubeGridBeliefDistribution:
    """NV-center Lorentzian belief: **unit** parameter grids, **physical** signal model.

    Default physical ranges match :func:`~nvision.sim.gen.core_generators.nv_center_lorentzian_bounds_for_domain`
    (same domain as :class:`~nvision.sim.gen.core_generators.NVCenterCoreGenerator`). Runs from the experiment
    runner merge in ``parameter_bounds`` from each generated :class:`~nvision.signal.signal.TrueSignal``.
    """
    from nvision.signal.nv_center import NVCenterLorentzianModel

    phys = nv_center_lorentzian_bounds_for_domain(DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX)
    # Keep non-zero NV contrast so the posterior cannot collapse to "flat signal".
    amp_lo, amp_hi = phys["amplitude"]
    amp_floor = 0.01 * amp_hi
    phys = {**phys, "amplitude": (max(amp_lo, amp_floor), amp_hi)}
    base_specs: list[tuple[str, tuple[float, float], int]] = [
        ("frequency", phys["frequency"], n_grid_freq),
        ("linewidth", phys["linewidth"], n_grid_linewidth),
        ("split", phys["split"], n_grid_split),
        ("k_np", phys["k_np"], n_grid_k_np),
        ("amplitude", phys["amplitude"], n_grid_amplitude),
        ("background", phys["background"], n_grid_background),
    ]
    return _unit_cube_belief_from_specs(
        model=NVCenterLorentzianModel(),
        parameter_bounds=parameter_bounds,
        specs=base_specs,
        x_param_name="frequency",
    )


def nv_center_smc_belief(
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    *,
    num_particles: int = 1000,
    jitter_scale: float = 0.05,
    ess_threshold: float = 0.5,
    **_extra: object,
) -> UnitCubeSMCBeliefDistribution:
    """NV-center Lorentzian belief: **unit** parameter particles, **physical** signal model."""
    from nvision.signal.nv_center import NVCenterLorentzianModel

    merged_bounds = nv_center_lorentzian_bounds_for_domain(DEFAULT_NV_CENTER_FREQ_X_MIN, DEFAULT_NV_CENTER_FREQ_X_MAX)

    if parameter_bounds:
        for name in merged_bounds:
            if name in parameter_bounds and parameter_bounds[name][1] > parameter_bounds[name][0]:
                merged_bounds[name] = parameter_bounds[name]

    # Enforce amplitude floor even when caller injects parameter bounds.
    amp_lo, amp_hi = merged_bounds["amplitude"]
    amp_floor = 0.01 * amp_hi
    merged_bounds["amplitude"] = (max(float(amp_lo), float(amp_floor)), float(amp_hi))

    x_phys = merged_bounds["frequency"]
    wrapped = UnitCubeSignalModel(NVCenterLorentzianModel(), merged_bounds, x_phys)

    return UnitCubeSMCBeliefDistribution(
        model=wrapped,
        parameter_bounds={name: (0.0, 1.0) for name in merged_bounds},
        num_particles=num_particles,
        jitter_scale=jitter_scale,
        ess_threshold=ess_threshold,
        physical_param_bounds=merged_bounds,
        physical_x_bounds=x_phys,
    )
