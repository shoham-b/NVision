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
    DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
    DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
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
        NVISION_SIGNAL_HWHM,
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
                hwhm, contrast_total = value, NVISION_SIGNAL_CONTRAST
            else:
                hwhm, contrast_total = NVISION_SIGNAL_HWHM, value
            name = "NVCenter-lorentzian"
            if hwhm is not None:
                name += f"-{_fmt_width(hwhm)}"
            if contrast_total is not None:
                name += f"-{_fmt_contrast(contrast_total)}"
            generators.append(
                (
                    name,
                    NVCenterCoreGenerator(
                        drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                        drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                        variant="lorentzian",
                        hwhm=hwhm,
                        contrast_total=contrast_total,
                    ),
                )
            )
        return generators

    if NVISION_SIGNAL_HWHM is not None or NVISION_SIGNAL_CONTRAST is not None:
        name = "NVCenter-lorentzian"
        if NVISION_SIGNAL_HWHM is not None:
            name += f"-{_fmt_width(NVISION_SIGNAL_HWHM)}"
        if NVISION_SIGNAL_CONTRAST is not None:
            name += f"-{_fmt_contrast(NVISION_SIGNAL_CONTRAST)}"
        return [
            (
                name,
                NVCenterCoreGenerator(
                    drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                    drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                    variant="lorentzian",
                    hwhm=NVISION_SIGNAL_HWHM,
                    contrast_total=NVISION_SIGNAL_CONTRAST,
                ),
            )
        ]

    return [
        # NV Center generators - different variants
        (
            "NVCenter-lorentzian",
            NVCenterCoreGenerator(
                drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                variant="lorentzian",
            ),
        ),
        (
            "NVCenter-voigt",
            NVCenterCoreGenerator(
                drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                variant="voigt",
            ),
        ),
        # Selectable inhomogeneous-broadening levels on the Voigt model. inhom-0 is
        # pure Lorentzian (lorentz_frac=1.0 -> zero Gaussian/inhomogeneous width);
        # inhom-low/high add increasing inhomogeneous broadening, which tends to wash
        # any hyperfine fine structure into a single unresolved dip.
        (
            "NVCenter-inhom-0",
            NVCenterCoreGenerator(
                drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                variant="voigt",
                lorentz_frac=1.0,
            ),
        ),
        (
            "NVCenter-inhom-low",
            NVCenterCoreGenerator(
                drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                variant="voigt",
                lorentz_frac=0.85,
            ),
        ),
        (
            "NVCenter-inhom-high",
            NVCenterCoreGenerator(
                drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
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
                        drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                        drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                        variant=variant,
                        hwhm=width,
                        contrast_total=contrast,
                        # Preserve this grid's historical 6-dip Voigt behavior: a ¹⁴N
                        # hyperfine triplet resolved per Zeeman group, with hyperfine_shift/k_np
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


def _fmt_hyperfine(hyperfine: str) -> str:
    return f"hf{hyperfine}"


def voigt_sigma_inhom_param_grid_generators() -> list[tuple[str, object]]:
    """Full width x contrast x sigma_inhom x hyperfine Cartesian grid of named plain-Voigt generators.

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

    ``hyperfine`` (``NVISION_SBED_HYPERFINE_VALUES``, default
    ``unresolved``/``n14``/``n15``) is a fourth, categorical axis: which nitrogen
    isotope's hyperfine structure the true signal resolves. It's encoded in the
    generator name (``-hf<value>``) rather than left to the class default, so
    each isotope gets its own selectable set of generators instead of silently
    always using ``NVCenterCoreGenerator``'s ``"unresolved"`` default --
    ``CombinationGrid.strategies_for()`` parses it back out of the name to keep
    the locator's belief in sync (see that function's docstring for why a
    generator/belief hyperfine mismatch is a real, silent bug and not a
    cosmetic detail). ``infer_hyperfine`` stays ``False``: like sigma_inhom and
    contrast, the isotope is a fixed, known grid coordinate here, not something
    the locator has to infer per repeat.
    """
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_SBED_CONTRAST_MAX,
        NVISION_SBED_CONTRAST_MIN,
        NVISION_SBED_CONTRAST_STEPS,
        NVISION_SBED_HYPERFINE_VALUES,
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
                for hyperfine in NVISION_SBED_HYPERFINE_VALUES:
                    name = (
                        f"NVCenter-voigt-{_fmt_width(width)}-{_fmt_contrast(contrast)}-"
                        f"{_fmt_sigma_inhom(sigma_inhom)}-{_fmt_hyperfine(hyperfine)}"
                    )
                    generators.append(
                        (
                            name,
                            NVCenterCoreGenerator(
                                drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                                drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                                variant="voigt",
                                hwhm=width,
                                contrast_total=contrast,
                                sigma_inhom=sigma_inhom,
                                hyperfine=hyperfine,
                            ),
                        )
                    )
    return generators


def _fmt_window(window_fwhm: float) -> str:
    return f"fw{window_fwhm:g}"


def voigt_snr_window_param_grid_generators() -> list[tuple[str, object]]:
    """Window-in-FWHM x sigma_inhom x hyperfine grid of named plain-Voigt generators (the SNR study's signals).

    The grid is posed in the dimensionless numbers that decide how hard the search is, rather than in Hz:

    * **Window / FWHM** (``fw<N>`` in the name): the drive-frequency window is ``N`` times the signal's total FWHM.
      The linewidth itself is held fixed (``NVISION_SBED_FIXED_HWHM`` plus the ``sigma_inhom`` axis), so it is the
      window that is stretched. That keeps everything with a physical scale (hyperfine offsets, the absolute
      convergence thresholds, the candidate step) meaningful across the axis, which sweeping the width in Hz
      against a fixed window did not. The window is ``[D, D + N * fwhm_total]``, and the Zeeman range follows it
      (:func:`~nvision.spectra.nv_center.max_zeeman_shift_for_window`).
    * **SNR** is not a generator axis: every generator carries the same ``contrast_total``
      (``NVISION_SBED_SNR_CONTRAST``), and :func:`snr_study_noises` sets the noise to ``contrast_total / SNR``.

    ``sigma_inhom`` and ``hyperfine`` are the same axes as in :func:`voigt_sigma_inhom_param_grid_generators`.
    """
    import numpy as np

    from nvision.sim.defaults import (
        NVISION_SBED_FIXED_HWHM,
        NVISION_SBED_HYPERFINE_VALUES,
        NVISION_SBED_SIGMA_INHOM_MAX,
        NVISION_SBED_SIGMA_INHOM_MIN,
        NVISION_SBED_SIGMA_INHOM_STEPS,
        NVISION_SBED_SNR_CONTRAST,
        NVISION_SBED_WINDOW_FWHM_VALUES,
    )
    from nvision.spectra.nv_center import NV_ZERO_FIELD_SPLITTING_HZ, _voigt_reparam_scalar

    sigma_inhoms = np.linspace(
        NVISION_SBED_SIGMA_INHOM_MIN, NVISION_SBED_SIGMA_INHOM_MAX, NVISION_SBED_SIGMA_INHOM_STEPS
    )

    generators: list[tuple[str, object]] = []
    for window_fwhm in NVISION_SBED_WINDOW_FWHM_VALUES:
        for sigma_inhom in sigma_inhoms:
            sigma_inhom = float(sigma_inhom)
            fwhm_total, _ = _voigt_reparam_scalar(NVISION_SBED_FIXED_HWHM, sigma_inhom)
            for hyperfine in NVISION_SBED_HYPERFINE_VALUES:
                name = (
                    f"NVCenter-voigt-{_fmt_window(window_fwhm)}-"
                    f"{_fmt_sigma_inhom(sigma_inhom)}-{_fmt_hyperfine(hyperfine)}"
                )
                generators.append(
                    (
                        name,
                        NVCenterCoreGenerator(
                            drive_freq_min_phys=NV_ZERO_FIELD_SPLITTING_HZ,
                            drive_freq_max_phys=NV_ZERO_FIELD_SPLITTING_HZ + window_fwhm * fwhm_total,
                            variant="voigt",
                            hwhm=NVISION_SBED_FIXED_HWHM,
                            contrast_total=NVISION_SBED_SNR_CONTRAST,
                            sigma_inhom=sigma_inhom,
                            hyperfine=hyperfine,
                            window_fwhm=window_fwhm,
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
                        drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                        drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
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


def snr_noise_name(snr: float, drift_label: str | None = None) -> str:
    """The name of the noise level of a given SNR: ``SNR(20)`` (``SNR(20)+Drift(label)`` with a drift scenario)."""
    name = f"SNR({snr:g})"
    return name if drift_label is None else f"{name}+Drift({drift_label})"


def snr_study_noises() -> list[tuple[str, CompositeNoise | None]]:
    """Noise grid of the SNR studies: one ``SNR(x)`` per level of ``NVISION_SBED_SNR_VALUES``.

    A run is identified by its SNR; the Gaussian sigma (``contrast / SNR``) is derived per signal when the
    combination is resolved (``CombinationGrid.resolve``), so the objects returned here are at the reference contrast
    ``NVISION_SBED_SNR_CONTRAST``, only for listing. The ``drive-control`` group uses the same levels, where the SNR is
    nominal: the contrast of a drive-controlled signal comes from the drive.
    """
    from nvision.sim.defaults import NVISION_SBED_SNR_CONTRAST, NVISION_SBED_SNR_VALUES

    noises = []
    for snr in NVISION_SBED_SNR_VALUES:
        if snr <= 0:
            raise ValueError(f"SNR levels must be positive, got {snr}")
        noises.append((snr_noise_name(snr), gauss_with_drift(NVISION_SBED_SNR_CONTRAST / snr, None)))
    return noises


# Drift scenarios
#
# Assumed duration of one simulated shot. It turns the physical timescales below into
# shots, so a strategy that takes more shots is exposed to proportionally more drift
# (SBED at 200 shots spans ~100 s; a sweep at thousands of shots, ~80 min). Not part
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


def drift_noise_name(snr: float, label: str) -> str:
    return snr_noise_name(snr, label)


def drift_study_noises() -> list[tuple[str, CompositeNoise | None]]:
    """A no-drift control plus every drift scenario, all at one SNR (``NVISION_DRIFT_SNR``).

    Named ``SNR(x)`` / ``SNR(x)+Drift(label)``; as in :func:`snr_study_noises` the sigma is derived per signal at
    resolve time and the objects here are at the reference contrast.
    """
    from nvision.sim.defaults import NVISION_DRIFT_SNR, NVISION_SBED_SNR_CONTRAST

    if NVISION_DRIFT_SNR <= 0:
        raise ValueError(f"NVISION_DRIFT_SNR must be positive, got {NVISION_DRIFT_SNR}")
    sigma = NVISION_SBED_SNR_CONTRAST / NVISION_DRIFT_SNR
    noises: list[tuple[str, CompositeNoise | None]] = [
        (snr_noise_name(NVISION_DRIFT_SNR), gauss_with_drift(sigma, None))
    ]
    for spec in drift_scenarios():
        noises.append((drift_noise_name(NVISION_DRIFT_SNR, spec.label), gauss_with_drift(sigma, spec)))
    return noises


def gauss_with_drift(sigma: float, drift: DriftSpec | None) -> CompositeNoise:
    return CompositeNoise(
        over_frequency_noise=CompositeOverFrequencyNoise([OverFrequencyGaussianNoise(sigma)]),
        drift=drift,
    )


def drive_controlled_generators() -> list[tuple[str, object]]:
    """Generators whose microwave drive is a per-measurement setting (CW and pulsed ODMR).

    Both use the default Zeeman-split, unresolved-hyperfine NV spectrum; the drive physics lives in
    :mod:`nvision.spectra.nv_center_drive`. Locators that do not choose the drive measure at the
    generator's baseline drive; ``Bayesian-SBED-Drive`` chooses it per step.
    """
    return [
        (
            "NVCenter-cw-drive",
            NVCenterCoreGenerator(
                drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                variant="cw_drive",
            ),
        ),
        (
            "NVCenter-pulsed-drive",
            NVCenterCoreGenerator(
                drive_freq_min_phys=DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
                drive_freq_max_phys=DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
                variant="pulsed_drive",
            ),
        ),
    ]
