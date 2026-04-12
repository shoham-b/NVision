"""Signal generators that produce typed TrueSignal objects directly."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from nvision.spectra import (
    CompositePeakModel,
    ExponentialDecayModel,
    GaussianModel,
    LorentzianModel,
    NVCenterLorentzianModel,
    NVCenterVoigtModel,
)
from nvision.spectra.composite import CompositeSpectrum
from nvision.spectra.exponential_decay import ExponentialDecaySpectrum
from nvision.spectra.gaussian import GaussianSpectrum
from nvision.spectra.lorentzian import LorentzianSpectrum
from nvision.spectra.nv_center import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    MAX_K_NP,
    MIN_K_NP,
    NVCenterLorentzianSpectrum,
    NVCenterVoigtSpectrum,
    NVCenterVoigtSpectrumSamples,
    nv_center_lorentzian_bounds_for_domain,
    nv_center_voigt_bounds_for_domain,
)
from nvision.spectra.signal import TrueSignal


# ---------------------------------------------------------------------------
# Peak spec — a plain data struct describing one peak type's constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeakSpec:
    """Static constants that describe a single peak type.

    Attributes:
        width_key: Parameter name for the width field (e.g. ``"sigma"``
            for Gaussian, ``"linewidth"`` for Lorentzian).
        width_frac: ``(lo, hi)`` width range expressed as *fractions of the
            domain width* (so ``0.01`` means 1 % of ``x_max - x_min``).
        dip_depth: ``(lo, hi)`` allowed range for the dip depth parameter.
        background: ``(lo, hi)`` allowed range for the background parameter
            when used as a *standalone* (non-composite) peak.
        composite_background: ``(lo, hi)`` range used when the peak is one
            component of a composite model and the baseline is shared.
        background_default: The fixed background value generated peaks use
            (0.0 for bump-up models, 1.0 for dip-down models).
    """

    width_key: str
    width_frac: tuple[float, float]
    dip_depth: tuple[float, float]
    background: tuple[float, float]
    composite_background: tuple[float, float]
    background_default: float


# ---------------------------------------------------------------------------
# Named singletons for the three supported peak types
# ---------------------------------------------------------------------------

GAUSSIAN = PeakSpec(
    width_key="sigma",
    width_frac=(0.01, 0.2),
    dip_depth=(0.1, 1.4),
    background=(0.0, 0.5),
    composite_background=(0.0, 0.5),
    background_default=0.0,
)

LORENTZIAN = PeakSpec(
    width_key="linewidth",
    width_frac=(0.01, 0.2),
    dip_depth=(0.05, 1.5),
    background=(0.5, 1.2),
    composite_background=(0.0, 0.5),
    background_default=1.0,
)

EXPONENTIAL = PeakSpec(
    width_key="decay_rate",
    width_frac=(0.1, 2.0),
    dip_depth=(0.1, 1.4),
    background=(0.0, 0.5),
    composite_background=(0.0, 0.5),
    background_default=0.0,
)


# ---------------------------------------------------------------------------
# Helpers that act on a PeakSpec
# ---------------------------------------------------------------------------


def _make_bounds(
    spec: PeakSpec,
    x_min: float,
    x_max: float,
    prefix: str = "",
    *,
    composite: bool = False,
) -> dict[str, tuple[float, float]]:
    """Return parameter bounds for *spec* over the domain ``[x_min, x_max]``.

    Args:
        spec: Peak type descriptor.
        x_min: Lower domain bound.
        x_max: Upper domain bound.
        prefix: Key prefix for composite models (e.g. ``"peak1_"``).
        composite: When ``True``, use the narrower background range
            appropriate for composite signal components.
    """
    w = x_max - x_min
    bg = spec.composite_background if composite else spec.background
    return {
        f"{prefix}frequency": (x_min, x_max),
        f"{prefix}{spec.width_key}": (w * spec.width_frac[0], w * spec.width_frac[1]),
        f"{prefix}dip_depth": spec.dip_depth,
        f"{prefix}background": bg,
    }


def _make_model_and_spectrum(
    spec: PeakSpec,
    *,
    pos: float,
    width: float,
    dip_depth: float,
    background: float,
) -> tuple[object, object]:
    """Instantiate ``(model, spectrum)`` for the given *spec*.

    Returns a ``(SignalModel, typed_spectrum)`` pair.
    """
    if spec is GAUSSIAN:
        return GaussianModel(), GaussianSpectrum(
            frequency=pos, sigma=width, dip_depth=dip_depth, background=background
        )
    if spec is LORENTZIAN:
        return LorentzianModel(), LorentzianSpectrum(
            frequency=pos, linewidth=width, dip_depth=dip_depth, background=background
        )
    if spec is EXPONENTIAL:
        return ExponentialDecayModel(), ExponentialDecaySpectrum(
            decay_rate=width * 5, dip_depth=dip_depth, background=background
        )
    raise ValueError(f"Unknown PeakSpec: {spec!r}")


def _true_signal_from_typed(model, typed_params, bounds: dict[str, tuple[float, float]]) -> TrueSignal:
    """Create a backward-compatible TrueSignal from typed model params."""
    return TrueSignal.from_typed(model=model, params=typed_params, bounds=bounds)


@dataclass
class OnePeakCoreGenerator:
    """Generates single-peak signals using core architecture.

    Produces TrueSignal with either Gaussian or Lorentzian model.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    peak_config: PeakSpec = field(default_factory=lambda: GAUSSIAN)

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate a single-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with single peak at random location
        """
        width = self.x_max - self.x_min
        peak_pos = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
        peak_width = rng.uniform(0.05 * width, 0.10 * width)

        model, typed_params = _make_model_and_spectrum(
            self.peak_config,
            pos=peak_pos,
            width=peak_width,
            dip_depth=1.0,
            background=self.peak_config.background_default,
        )
        bounds = _make_bounds(self.peak_config, self.x_min, self.x_max)

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)


@dataclass
class NVCenterCoreGenerator:
    """Generates NV center ODMR signals using core architecture.

    Produces TrueSignal with physically accurate NV center signal.
    """

    x_min: float = DEFAULT_NV_CENTER_FREQ_X_MIN  # 2.6 GHz
    x_max: float = DEFAULT_NV_CENTER_FREQ_X_MAX  # 3.1 GHz
    variant: str = "lorentzian"  # "lorentzian" or "voigt"
    zero_field: bool = False  # If True, generate with delta_f_hf = 0 (single peak)

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate NV center ODMR signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            NV center signal with realistic parameters
        """
        width = self.x_max - self.x_min

        # Random central frequency (avoiding edges)
        if self.zero_field:
            # For zero-field case, can be anywhere
            center_freq = rng.uniform(self.x_min + 0.1 * width, self.x_max - 0.1 * width)
            split = 0.0
        else:
            # For hyperfine-split case, need room for side peaks
            # Generate something roughly centered around the physical values for 14N and 15N (2.16 MHz and 3.03 MHz)
            split = rng.uniform(2.0e6, 3.5e6)
            center_freq = rng.uniform(self.x_min + split + 0.05 * width, self.x_max - split - 0.05 * width)

        # Random linewidth (HWHM for Lorentzian)
        # To ensure the dip strongly returns before the next hyperfine peak,
        # we generate exceptionally sharp lines (50 kHz to 400 kHz HWHM).
        linewidth = rng.uniform(0.05e6, 0.4e6)

        # Random k_np (non-polarization factor)
        k_np = rng.uniform(MIN_K_NP, MAX_K_NP)

        # Normalize NV Center ODMR directly to [0, 1] bounds using exactly 1.0 maximum dip
        background = 1.0

        if self.variant == "lorentzian":
            model = NVCenterLorentzianModel()
            # Scale a desired contrast onto the true peak-shape maximum
            unit_dip_depth = rng.uniform(0.3, 0.95)
            lw2 = linewidth**2
            xs = np.linspace(center_freq - split, center_freq + split, 200)
            g = (
                (lw2 / k_np) / ((xs - (center_freq - split)) ** 2 + lw2)
                + lw2 / ((xs - center_freq) ** 2 + lw2)
                + (lw2 * k_np) / ((xs - (center_freq + split)) ** 2 + lw2)
            )
            dip_depth = unit_dip_depth / float(g.max())

            typed_params = NVCenterLorentzianSpectrum(
                frequency=center_freq,
                linewidth=linewidth,
                split=split,
                k_np=k_np,
                dip_depth=dip_depth,
                background=background,
            )
            bounds = nv_center_lorentzian_bounds_for_domain(self.x_min, self.x_max)
        else:  # voigt
            fwhm_lorentz = 2 * linewidth
            fwhm_gauss = fwhm_lorentz * rng.uniform(0.1, 0.3)

            model = NVCenterVoigtModel()
            # Scale a desired contrast onto the true peak-shape maximum
            unit_dip_depth = rng.uniform(0.3, 0.95)
            xs = np.linspace(center_freq - split, center_freq + split, 200)
            single = NVCenterVoigtSpectrumSamples(
                frequency=np.array([center_freq]),
                fwhm_lorentz=np.array([fwhm_lorentz]),
                fwhm_gauss=np.array([fwhm_gauss]),
                split=np.array([split]),
                k_np=np.array([k_np]),
                dip_depth=np.array([1.0]),
                background=np.array([0.0]),
            )
            g_max = float(-np.min(model.compute_vectorized_many(xs, single)))
            dip_depth = unit_dip_depth / g_max if g_max > 1e-12 else unit_dip_depth

            typed_params = NVCenterVoigtSpectrum(
                frequency=center_freq,
                fwhm_lorentz=fwhm_lorentz,
                fwhm_gauss=fwhm_gauss,
                split=split,
                k_np=k_np,
                dip_depth=dip_depth,
                background=background,
            )
            bounds = nv_center_voigt_bounds_for_domain(self.x_min, self.x_max)

        # Lock split to 0 for zero-field case (not a sought parameter)
        if self.zero_field:
            bounds["split"] = (0.0, 0.0)

        return _true_signal_from_typed(model=model, typed_params=typed_params, bounds=bounds)


@dataclass
class TwoPeakCoreGenerator:
    """Generates two-peak signals using core architecture.

    Creates a signal with two separate peaks by combining two single-peak signals.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    peak_config_left: PeakSpec = field(default_factory=lambda: GAUSSIAN)
    peak_config_right: PeakSpec = field(default_factory=lambda: GAUSSIAN)
    min_separation: float = 0.2  # Minimum separation as fraction of domain

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate two-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with two separated peaks using CompositePeakModel
        """
        width = self.x_max - self.x_min
        min_sep = self.min_separation * width

        # Generate two well-separated peak positions
        peak1_pos = rng.uniform(self.x_min + 0.1 * width, self.x_max - 0.5 * width)
        peak2_pos = rng.uniform(peak1_pos + min_sep, self.x_max - 0.1 * width)

        # Random peak parameters
        peak1_width = rng.uniform(0.02 * width, 0.08 * width)
        peak2_width = rng.uniform(0.02 * width, 0.08 * width)
        background = max(
            self.peak_config_left.background_default,
            self.peak_config_right.background_default,
        )
        model1, peak1_typed = _make_model_and_spectrum(
            self.peak_config_left,
            pos=peak1_pos, width=peak1_width, dip_depth=1.0, background=background / 2,
        )
        model2, peak2_typed = _make_model_and_spectrum(
            self.peak_config_right,
            pos=peak2_pos, width=peak2_width, dip_depth=1.0, background=background / 2,
        )
        composite_model = CompositePeakModel([("peak1", model1), ("peak2", model2)])

        bounds: dict[str, tuple[float, float]] = {}
        bounds.update(_make_bounds(self.peak_config_left, self.x_min, self.x_max, "peak1_", composite=True))
        bounds.update(_make_bounds(self.peak_config_right, self.x_min, self.x_max, "peak2_", composite=True))

        typed_params = CompositeSpectrum(peaks=(peak1_typed, peak2_typed))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)


@dataclass
class MultiPeakCoreGenerator:
    """Generates multi-peak signals using core architecture.

    Creates a signal with N separate peaks.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    count: int = 3
    peak_configs: list[PeakSpec] | None = None  # per-peak specs; defaults to all Gaussian
    min_separation: float = 0.1  # Minimum separation as fraction of domain

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate multi-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with N peaks using CompositePeakModel
        """
        width = self.x_max - self.x_min
        min_sep = self.min_separation * width

        # Resolve per-peak configs, padding with Gaussian when too few supplied.
        if self.peak_configs is None:
            configs: list[PeakSpec] = [GAUSSIAN] * self.count
        else:
            configs = list(self.peak_configs[: self.count])
            while len(configs) < self.count:
                configs.append(GAUSSIAN)

        # Generate well-separated peak positions
        positions: list[float] = []
        max_attempts = 1000
        for _ in range(self.count):
            for _ in range(max_attempts):
                pos = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
                if not positions or all(abs(pos - p) >= min_sep for p in positions):
                    positions.append(pos)
                    break
        positions.sort()

        # Create signal and typed parameters
        models = []
        typed_peak_params: list[object] = []
        bounds: dict[str, tuple[float, float]] = {}
        background = max(cfg.background_default for cfg in configs)

        for i, (pos, cfg) in enumerate(zip(positions, configs, strict=False)):
            prefix = f"peak{i + 1}"
            peak_width = rng.uniform(0.02 * width, 0.08 * width)
            model, spectrum = _make_model_and_spectrum(
                cfg, pos=pos, width=peak_width, dip_depth=1.0, background=background / self.count
            )
            typed_peak_params.append(spectrum)
            bounds.update(_make_bounds(cfg, self.x_min, self.x_max, f"{prefix}_", composite=True))
            models.append((prefix, model))

        composite_model = CompositePeakModel(models)
        typed_params = CompositeSpectrum(peaks=tuple(typed_peak_params))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)


@dataclass
class SymmetricTwoPeakCoreGenerator:
    """Generates symmetric two-peak signals using core architecture.

    Creates two peaks symmetrically placed around a center point.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    center: float = 0.5
    sep_frac: float = 0.2  # Separation as fraction of domain
    peak_config: PeakSpec = field(default_factory=lambda: GAUSSIAN)

    def generate(self, rng: random.Random):  # TrueSignal
        """Generate symmetric two-peak signal.

        Parameters
        ----------
        rng : random.Random
            Random number generator

        Returns
        -------
        TrueSignal
            Signal with two symmetric peaks
        """
        width = self.x_max - self.x_min
        delta = 0.5 * self.sep_frac * width

        left_pos = self.center - delta
        right_pos = self.center + delta

        peak_width = rng.uniform(0.02 * width, 0.06 * width)
        background = self.peak_config.background_default
        model1, peak1_typed = _make_model_and_spectrum(
            self.peak_config, pos=left_pos, width=peak_width, dip_depth=1.0, background=background / 2
        )
        model2, peak2_typed = _make_model_and_spectrum(
            self.peak_config, pos=right_pos, width=peak_width, dip_depth=1.0, background=background / 2
        )
        composite_model = CompositePeakModel([("peak1", model1), ("peak2", model2)])
        bounds: dict[str, tuple[float, float]] = {}
        bounds.update(_make_bounds(self.peak_config, self.x_min, self.x_max, "peak1_", composite=True))
        bounds.update(_make_bounds(self.peak_config, self.x_min, self.x_max, "peak2_", composite=True))

        typed_params = CompositeSpectrum(peaks=(peak1_typed, peak2_typed))
        return _true_signal_from_typed(model=composite_model, typed_params=typed_params, bounds=bounds)
