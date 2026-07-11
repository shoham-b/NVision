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
            NVCenterCoreGenerator(x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="lorentzian"),
        ),
        (
            "NVCenter-voigt",
            NVCenterCoreGenerator(x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="voigt"),
        ),
        # Selectable inhomogeneous-broadening levels on the Voigt model. inhom-0 is
        # pure Lorentzian (lorentz_frac=1.0 -> zero Gaussian/inhomogeneous width);
        # inhom-low/high add increasing inhomogeneous broadening, which tends to wash
        # the hyperfine triplet fine structure into a single unresolved dip.
        (
            "NVCenter-inhom-0",
            NVCenterCoreGenerator(
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="voigt", lorentz_frac=1.0
            ),
        ),
        (
            "NVCenter-inhom-low",
            NVCenterCoreGenerator(
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="voigt", lorentz_frac=0.85
            ),
        ),
        (
            "NVCenter-inhom-high",
            NVCenterCoreGenerator(
                x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant="voigt", lorentz_frac=0.55
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
                        x_min=DEFAULT_NV_CENTER_FREQ_X_MIN, x_max=DEFAULT_NV_CENTER_FREQ_X_MAX, variant=variant, linewidth=width, c_total=contrast
                    ),
                )
            )
    return generators


def _fmt_sigma_inhom(hz: float) -> str:
    return f"si{hz / 1e6:.2f}MHz"


def _fmt_lorentz_frac(frac: float) -> str:
    return f"lf{frac:.2f}"


def voigt_lorentz_frac_param_grid_generators() -> list[tuple[str, object]]:
    """Full width x contrast x lorentz_frac Cartesian grid of named plain-Voigt generators.

    Unlike :func:`param_grid_generators` (variant="voigt"), which sweeps only
    width x contrast and leaves the Lorentzian/Gaussian mixing ratio either
    fixed or randomized per repeat, this grid makes ``lorentz_frac`` (the
    Lorentzian share of the total FWHM -- see
    :data:`~nvision.sim.gen.nv_center_generator.NVCenterCoreGenerator.lorentz_frac`)
    an explicit, selectable third axis, so the UI can offer the same
    width/contrast/inhomogeneous-broadening selection that
    :func:`saturation_voigt_param_grid_generators` offers for the
    saturation-Voigt lineshape.
    """
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_SBED_CONTRAST_MAX,
        NVISION_SBED_CONTRAST_MIN,
        NVISION_SBED_CONTRAST_STEPS,
        NVISION_SBED_LORENTZ_FRAC_MAX,
        NVISION_SBED_LORENTZ_FRAC_MIN,
        NVISION_SBED_LORENTZ_FRAC_STEPS,
        NVISION_SBED_WIDTH_MAX,
        NVISION_SBED_WIDTH_MIN,
        NVISION_SBED_WIDTH_STEPS,
    )

    widths = np.linspace(NVISION_SBED_WIDTH_MIN, NVISION_SBED_WIDTH_MAX, NVISION_SBED_WIDTH_STEPS)
    contrasts = np.linspace(NVISION_SBED_CONTRAST_MIN, NVISION_SBED_CONTRAST_MAX, NVISION_SBED_CONTRAST_STEPS)
    lorentz_fracs = np.linspace(
        NVISION_SBED_LORENTZ_FRAC_MIN, NVISION_SBED_LORENTZ_FRAC_MAX, NVISION_SBED_LORENTZ_FRAC_STEPS
    )

    generators: list[tuple[str, object]] = []
    for width in widths:
        for contrast in contrasts:
            for lorentz_frac in lorentz_fracs:
                width, contrast, lorentz_frac = float(width), float(contrast), float(lorentz_frac)
                name = (
                    f"NVCenter-voigt-{_fmt_width(width)}-{_fmt_contrast(contrast)}"
                    f"-{_fmt_lorentz_frac(lorentz_frac)}"
                )
                generators.append(
                    (
                        name,
                        NVCenterCoreGenerator(
                            x_min=DEFAULT_NV_CENTER_FREQ_X_MIN,
                            x_max=DEFAULT_NV_CENTER_FREQ_X_MAX,
                            variant="voigt",
                            linewidth=width,
                            c_total=contrast,
                            lorentz_frac=lorentz_frac,
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
