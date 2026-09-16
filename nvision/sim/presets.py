"""Preset generators, noises, and constants for NVision simulations.

This module replaces the generator/noise definitions that used to live in
``nvision.sim.cases`` so they can be imported without pulling in the
``RunCase`` / ``RunGroup`` machinery.
"""

from __future__ import annotations

from nvision.models.noise import (
    CompositeNoise,
    CompositeOverFrequencyNoise,
)
from nvision.noises import (
    OverFrequencyGaussianNoise,
)
from nvision.noises.drift import DriftProcess, DriftSpec
from nvision.sim.defaults import (
    NVISION_DEFAULT_LOC_MAX_STEPS,
)

from .gen.nv_center_generator import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    NVCenterCoreGenerator,
)

# Single source for ``nvision run`` / ``nvision render`` defaults.
DEFAULT_LOC_MAX_STEPS = NVISION_DEFAULT_LOC_MAX_STEPS


def _fmt_width(hz: float) -> str:
    return f"w{hz / 1e6:.2f}MHz"


def _fmt_contrast(c: float) -> str:
    return f"c{c:.2f}"


# Generators: NV Center variants
# Now using core architecture with TrueSignal and explicit SignalModels
def generators_basic() -> list[tuple[str, object]]:
    from nvision.sim.defaults import (
        NVISION_SIGNAL_CONTRAST,
        NVISION_SIGNAL_LINEWIDTH,
        NVISION_SIGNAL_SWEEP_MAX,
        NVISION_SIGNAL_SWEEP_MIN,
        NVISION_SIGNAL_SWEEP_PARAM,
        NVISION_SIGNAL_SWEEP_STEPS,
    )

    if NVISION_SIGNAL_SWEEP_PARAM in ("width", "contrast"):
        import numpy as np

        if NVISION_SIGNAL_SWEEP_MIN is None or NVISION_SIGNAL_SWEEP_MAX is None:
            raise ValueError(
                "NVISION_SIGNAL_SWEEP_MIN and NVISION_SIGNAL_SWEEP_MAX must be set "
                f"when NVISION_SIGNAL_SWEEP_PARAM={NVISION_SIGNAL_SWEEP_PARAM!r}"
            )
        sweep_values = np.linspace(NVISION_SIGNAL_SWEEP_MIN, NVISION_SIGNAL_SWEEP_MAX, NVISION_SIGNAL_SWEEP_STEPS)

        generators: list[tuple[str, object]] = []
        for value in sweep_values:
            value = float(value)
            if NVISION_SIGNAL_SWEEP_PARAM == "width":
                linewidth, c_total = value, NVISION_SIGNAL_CONTRAST
            else:
                linewidth, c_total = NVISION_SIGNAL_LINEWIDTH, value
            name = "NVCenter-lorentzian"
            if linewidth is not None:
                name += f"-{_fmt_width(linewidth)}"
            if c_total is not None:
                name += f"-{_fmt_contrast(c_total)}"
            generators.append(
                (
                    name,
                    NVCenterCoreGenerator(
                        x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                        x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                        variant="lorentzian",
                        linewidth=linewidth,
                        c_total=c_total,
                    ),
                )
            )
        return generators

    if NVISION_SIGNAL_LINEWIDTH is not None or NVISION_SIGNAL_CONTRAST is not None:
        name = "NVCenter-lorentzian"
        if NVISION_SIGNAL_LINEWIDTH is not None:
            name += f"-{_fmt_width(NVISION_SIGNAL_LINEWIDTH)}"
        if NVISION_SIGNAL_CONTRAST is not None:
            name += f"-{_fmt_contrast(NVISION_SIGNAL_CONTRAST)}"
        return [
            (
                name,
                NVCenterCoreGenerator(
                    x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                    x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                    variant="lorentzian",
                    linewidth=NVISION_SIGNAL_LINEWIDTH,
                    c_total=NVISION_SIGNAL_CONTRAST,
                ),
            )
        ]

    return [
        # NV Center generators - different variants
        (
            "NVCenter-lorentzian",
            NVCenterCoreGenerator(
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="lorentzian"
            ),
        ),
        (
            "NVCenter-voigt",
            NVCenterCoreGenerator(
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="voigt"
            ),
        ),
        # Selectable inhomogeneous-broadening levels on the Voigt model. inhom-0 is
        # pure Lorentzian (lorentz_frac=1.0 -> zero Gaussian/inhomogeneous width);
        # inhom-low/high add increasing inhomogeneous broadening, which tends to wash
        # any hyperfine fine structure into a single unresolved dip.
        (
            "NVCenter-inhom-0",
            NVCenterCoreGenerator(
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                variant="voigt",
                lorentz_frac=1.0,
            ),
        ),
        (
            "NVCenter-inhom-low",
            NVCenterCoreGenerator(
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                variant="voigt",
                lorentz_frac=0.85,
            ),
        ),
        (
            "NVCenter-inhom-high",
            NVCenterCoreGenerator(
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                variant="voigt",
                lorentz_frac=0.55,
            ),
        ),
    ]


def param_grid_generators(variant: str = "lorentzian") -> list[tuple[str, object]]:
    """Full width x contrast Cartesian grid of named (lorentzian/voigt) generators.

    Not used by the SBED run-groups by default anymore (see
    :func:`saturation_voigt_param_grid_generators`); kept for direct/manual
    lorentzian studies. Independent of ``generators_basic()``/the
    ``NVISION_SIGNAL_*`` single-axis sweep used by ad-hoc ``nvision run``
    flags, so it doesn't change plain-run behavior.
    """
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_SBED_CONTRAST_MAX,
        NVISION_SBED_CONTRAST_MIN,
        NVISION_SBED_CONTRAST_STEPS,
        NVISION_SBED_WIDTH_MAX,
        NVISION_SBED_WIDTH_MIN,
        NVISION_SBED_WIDTH_STEPS,
    )

    widths = np.linspace(NVISION_SBED_WIDTH_MIN, NVISION_SBED_WIDTH_MAX, NVISION_SBED_WIDTH_STEPS)
    contrasts = np.linspace(NVISION_SBED_CONTRAST_MIN, NVISION_SBED_CONTRAST_MAX, NVISION_SBED_CONTRAST_STEPS)

    generators: list[tuple[str, object]] = []
    for width in widths:
        for contrast in contrasts:
            width, contrast = float(width), float(contrast)
            name = f"NVCenter-{variant}-{_fmt_width(width)}-{_fmt_contrast(contrast)}"
            generators.append(
                (
                    name,
                    NVCenterCoreGenerator(
                        x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                        x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                        variant=variant,
                        linewidth=width,
                        c_total=contrast,
                        # Preserve this grid's historical 6-dip Voigt behavior: a ¹⁴N
                        # hyperfine triplet resolved per Zeeman group, with split/k_np
                        # drawn per repeat and inferred. Every other generator now
                        # defaults to hyperfine="unresolved" (the lines merge into one
                        # dip), so this grid has to ask for the triplet explicitly --
                        # combinations.py's strategies_for() sets the matching
                        # belief-side override for this grid's name pattern.
                        hyperfine="n14" if variant == "voigt" else "unresolved",
                        infer_hyperfine=(variant == "voigt"),
                    ),
                )
            )
    return generators


def _fmt_sigma_inhom(hz: float) -> str:
    return f"si{hz / 1e6:.2f}MHz"


def voigt_sigma_inhom_param_grid_generators() -> list[tuple[str, object]]:
    """Full width x contrast x sigma_inhom Cartesian grid of named plain-Voigt generators.

    Unlike :func:`param_grid_generators` (variant="voigt"), which sweeps only
    width x contrast and leaves the inhomogeneous (Gaussian) broadening either
    fixed or randomized per repeat, this grid makes ``sigma_inhom`` (the same
    physical Hz-scale inhomogeneous width :func:`saturation_voigt_param_grid_generators`
    sweeps) an explicit, selectable third axis, passed straight through as
    ``NVCenterVoigtModel``'s own ``sigma_inhom`` parameter (see
    :class:`~nvision.sim.gen.nv_center_generator.NVCenterCoreGenerator`).
    Reuses the same ``NVISION_SBED_SIGMA_INHOM_*`` range as the saturation-Voigt
    grid so the two lineshapes' inhomogeneous-broadening axis is directly
    comparable.
    """
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_SBED_CONTRAST_MAX,
        NVISION_SBED_CONTRAST_MIN,
        NVISION_SBED_CONTRAST_STEPS,
        NVISION_SBED_SIGMA_INHOM_MAX,
        NVISION_SBED_SIGMA_INHOM_MIN,
        NVISION_SBED_SIGMA_INHOM_STEPS,
        NVISION_SBED_WIDTH_MAX,
        NVISION_SBED_WIDTH_MIN,
        NVISION_SBED_WIDTH_STEPS,
    )

    widths = np.linspace(NVISION_SBED_WIDTH_MIN, NVISION_SBED_WIDTH_MAX, NVISION_SBED_WIDTH_STEPS)
    contrasts = np.linspace(NVISION_SBED_CONTRAST_MIN, NVISION_SBED_CONTRAST_MAX, NVISION_SBED_CONTRAST_STEPS)
    sigma_inhoms = np.linspace(
        NVISION_SBED_SIGMA_INHOM_MIN, NVISION_SBED_SIGMA_INHOM_MAX, NVISION_SBED_SIGMA_INHOM_STEPS
    )

    generators: list[tuple[str, object]] = []
    for width in widths:
        for contrast in contrasts:
            for sigma_inhom in sigma_inhoms:
                width, contrast, sigma_inhom = float(width), float(contrast), float(sigma_inhom)
                name = f"NVCenter-voigt-{_fmt_width(width)}-{_fmt_contrast(contrast)}-{_fmt_sigma_inhom(sigma_inhom)}"
                generators.append(
                    (
                        name,
                        NVCenterCoreGenerator(
                            x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                            x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                            variant="voigt",
                            linewidth=width,
                            c_total=contrast,
                            sigma_inhom=sigma_inhom,
                        ),
                    )
                )
    return generators


def saturation_voigt_param_grid_generators() -> list[tuple[str, object]]:
    """Full contrast x sigma_inhom grid of named saturation-Voigt generators.

    Default generator set for the SBED run-groups in ``run_groups.py``.

    Sweeps target **contrast** directly (the same ``NVISION_SBED_CONTRAST_*`` range
    ``param_grid_generators()`` uses for lorentzian/voigt, so all three lineshape
    study grids move contrast by the same amount per step and are directly
    comparable) and ``sigma_inhom`` (independent inhomogeneous/Gaussian width,
    uncoupled from contrast). ``saturation`` is *solved* per grid point from the
    target contrast by inverting the saturation law ``C = c_max * s/(1+s)``:
    ``s = C / (c_max - C)``, with ``c_max`` held fixed at
    :data:`NVISION_SBED_C_MAX`. This replaces an earlier design that swept
    ``saturation`` linearly: because ``s/(1+s)`` saturates quickly, most of a
    linear ``saturation`` grid barely moved contrast at all (78% of the total
    contrast change happened in the first of 5 steps). Sweeping contrast directly
    makes every grid point move the signal by a comparable, intended amount.

    ``c_max`` is not passed to :class:`NVCenterCoreGenerator` (no such field --
    see its docstring): it's a fixed model constant
    (:data:`~nvision.spectra.nv_center.NV_SATURATION_C_MAX`), not a per-repeat
    value, and it reads the identical ``NVISION_SBED_C_MAX`` env var/default so
    the two can never drift apart.
    """
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_SBED_C_MAX,
        NVISION_SBED_CONTRAST_MAX,
        NVISION_SBED_CONTRAST_MIN,
        NVISION_SBED_CONTRAST_STEPS,
        NVISION_SBED_SIGMA_INHOM_MAX,
        NVISION_SBED_SIGMA_INHOM_MIN,
        NVISION_SBED_SIGMA_INHOM_STEPS,
    )

    if NVISION_SBED_CONTRAST_MAX >= NVISION_SBED_C_MAX:
        raise ValueError(
            f"NVISION_SBED_CONTRAST_MAX ({NVISION_SBED_CONTRAST_MAX}) must be strictly less "
            f"than NVISION_SBED_C_MAX ({NVISION_SBED_C_MAX}) -- contrast can only approach "
            "c_max asymptotically (C = c_max*s/(1+s)), never reach or exceed it."
        )

    contrasts = np.linspace(NVISION_SBED_CONTRAST_MIN, NVISION_SBED_CONTRAST_MAX, NVISION_SBED_CONTRAST_STEPS)
    sigma_inhoms = np.linspace(
        NVISION_SBED_SIGMA_INHOM_MIN, NVISION_SBED_SIGMA_INHOM_MAX, NVISION_SBED_SIGMA_INHOM_STEPS
    )

    generators: list[tuple[str, object]] = []
    for contrast in contrasts:
        for sigma_inhom in sigma_inhoms:
            contrast, sigma_inhom = float(contrast), float(sigma_inhom)
            saturation = contrast / (NVISION_SBED_C_MAX - contrast)
            name = f"NVCenter-saturation_voigt-{_fmt_contrast(contrast)}-{_fmt_sigma_inhom(sigma_inhom)}"
            generators.append(
                (
                    name,
                    NVCenterCoreGenerator(
                        x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                        x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                        variant="saturation_voigt",
                        saturation=saturation,
                        sigma_inhom=sigma_inhom,
                    ),
                )
            )
    return generators


# Noise tiers


def noises_single_each() -> list[tuple[str, CompositeNoise | None]]:
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_NOISE_GAUSS_STEPS,
        NVISION_NOISE_MAX_GAUSS,
    )

    noises = []
    if NVISION_NOISE_GAUSS_STEPS > 1:
        sigmas = np.linspace(0.0, NVISION_NOISE_MAX_GAUSS, NVISION_NOISE_GAUSS_STEPS)
    elif NVISION_NOISE_GAUSS_STEPS == 1:
        sigmas = [NVISION_NOISE_MAX_GAUSS]
    else:
        sigmas = []

    for sigma in sigmas:
        sigma_val = float(round(sigma, 4))
        noises.append(
            (
                f"Gauss({sigma_val})",
                CompositeNoise(
                    over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma_val)])
                ),
            )
        )

    return noises


def sbed_study_noises() -> list[tuple[str, CompositeNoise | None]]:
    """Noise grid for the SBED run-groups — swept like width/contrast via its own
    dedicated (smaller) range, independent of the generic NVISION_NOISE_* config
    used by plain ``nvision run``."""
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_SBED_NOISE_MAX,
        NVISION_SBED_NOISE_MIN,
        NVISION_SBED_NOISE_STEPS,
    )

    if NVISION_SBED_NOISE_STEPS > 1:
        sigmas = np.linspace(NVISION_SBED_NOISE_MIN, NVISION_SBED_NOISE_MAX, NVISION_SBED_NOISE_STEPS)
    elif NVISION_SBED_NOISE_STEPS == 1:
        sigmas = [NVISION_SBED_NOISE_MAX]
    else:
        sigmas = []

    noises = []
    for sigma in sigmas:
        sigma_val = float(round(sigma, 4))
        noises.append(
            (
                f"Gauss({sigma_val})",
                CompositeNoise(
                    over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma_val)])
                ),
            )
        )

    return noises


# Drift scenarios
#
# Assumed duration of one simulated shot. It turns the physical timescales below into
# shots, so a strategy that takes more shots is exposed to proportionally more drift
# (SBED at 200 shots spans ~100 s; SimpleSobol at up to 10000 shots, ~80 min). Not part
# of the noise name, so changing it would silently reuse cached results: add a scenario
# instead.
DRIFT_SHOT_DURATION_S = 0.5

# NdFeB remanence temperature coefficient (about -0.12 %/K): a permanent magnet's field,
# and so the Zeeman splitting, tracks the same temperature that moves the center.
_NDFEB_TEMPCO_PER_K = -0.0012


def drift_scenarios() -> list[DriftSpec]:
    """Named drift scenarios, from negligible to deliberately beyond realistic.

    Center shift is -74 kHz/K; splitting shift is 2.8 kHz/mG, plus -0.12 %/K of the
    splitting for a permanent magnet. The simulated splitting is drawn from 0-60 MHz
    (B up to ~21 G), so at a mid-range 30 MHz, 1 K moves a permanent magnet's splitting
    ~36 kHz — about half the center's shift.
    """
    dt = DRIFT_SHOT_DURATION_S
    return [
        # Temperature-controlled mount, laser long warmed up, permanent magnet.
        # Center wander ~1.5 kHz, splitting ~1.5-2 kHz: a near-zero control.
        DriftSpec(
            label="stable",
            shot_duration_s=dt,
            temperature_k=DriftProcess(ou_sigma=0.02, ou_tau_s=1800.0),
            field_mg=DriftProcess(ou_sigma=0.5, ou_tau_s=600.0),
            magnet_tempco_per_k=_NDFEB_TEMPCO_PER_K,
        ),
        # Uncontrolled room: slow wander plus a 20-minute air-conditioning cycle, occasional
        # small field jumps from nearby equipment. Center ~20-40 kHz, splitting ~10-20 kHz.
        DriftSpec(
            label="lab",
            shot_duration_s=dt,
            temperature_k=DriftProcess(ou_sigma=0.3, ou_tau_s=1200.0, periodic_amplitude=0.2, periodic_period_s=1200.0),
            field_mg=DriftProcess(ou_sigma=2.0, ou_tau_s=600.0, step_rate_per_s=1 / 1800, step_sigma=3.0),
            magnet_tempco_per_k=_NDFEB_TEMPCO_PER_K,
        ),
        # Measuring before thermal equilibrium (laser/microwave just switched on): the
        # sample warms 4 K with a 5-minute time constant. Center -300 kHz at full warm-up
        # (~-85 kHz within an SBED-length run); a 30 MHz permanent-magnet splitting -145 kHz.
        DriftSpec(
            label="warmup",
            shot_duration_s=dt,
            temperature_k=DriftProcess(warmup_amplitude=4.0, warmup_tau_s=300.0, ou_sigma=0.1, ou_tau_s=1200.0),
            field_mg=DriftProcess(ou_sigma=0.5, ou_tau_s=600.0),
            magnet_tempco_per_k=_NDFEB_TEMPCO_PER_K,
        ),
        # Electromagnet: coils heat and the field sags 40 mG (0.4% of 10 G) over 15 minutes,
        # with supply wander and switching jumps. Splitting -110 kHz at full sag; center ~7 kHz.
        DriftSpec(
            label="electromagnet",
            shot_duration_s=dt,
            temperature_k=DriftProcess(ou_sigma=0.1, ou_tau_s=1200.0),
            field_mg=DriftProcess(
                warmup_amplitude=-40.0,
                warmup_tau_s=900.0,
                ou_sigma=4.0,
                ou_tau_s=300.0,
                step_rate_per_s=1 / 900,
                step_sigma=5.0,
            ),
            magnet_tempco_per_k=0.0,
        ),
        # Beyond realistic, to find where each strategy breaks: 15 K warm-up (center
        # -1.1 MHz), 200 mG field sag (splitting -560 kHz), frequent field jumps.
        DriftSpec(
            label="stress",
            shot_duration_s=dt,
            temperature_k=DriftProcess(
                warmup_amplitude=15.0,
                warmup_tau_s=300.0,
                ou_sigma=1.0,
                ou_tau_s=600.0,
                periodic_amplitude=1.0,
                periodic_period_s=600.0,
            ),
            field_mg=DriftProcess(
                warmup_amplitude=-200.0,
                warmup_tau_s=300.0,
                ou_sigma=20.0,
                ou_tau_s=300.0,
                step_rate_per_s=1 / 300,
                step_sigma=30.0,
            ),
            magnet_tempco_per_k=_NDFEB_TEMPCO_PER_K,
        ),
    ]


def drift_scenario(label: str) -> DriftSpec | None:
    for spec in drift_scenarios():
        if spec.label == label:
            return spec
    return None


def drift_noise_name(sigma: float, label: str) -> str:
    return f"Gauss({sigma})+Drift({label})"


def drift_study_noises() -> list[tuple[str, CompositeNoise | None]]:
    """A no-drift control plus every drift scenario, all at NVISION_DRIFT_GAUSS_SIGMA."""
    from nvision.sim.defaults import NVISION_DRIFT_GAUSS_SIGMA

    sigma = float(round(NVISION_DRIFT_GAUSS_SIGMA, 4))
    noises: list[tuple[str, CompositeNoise | None]] = [(f"Gauss({sigma})", gauss_with_drift(sigma, None))]
    for spec in drift_scenarios():
        noises.append((drift_noise_name(sigma, spec.label), gauss_with_drift(sigma, spec)))
    return noises


def gauss_with_drift(sigma: float, drift: DriftSpec | None) -> CompositeNoise:
    return CompositeNoise(
        over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma)]),
        drift=drift,
    )
