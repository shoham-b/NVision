from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from nvision.sim.gen._protocols import PeakManufacturer
from nvision.sim.gen.distributions.convolution_manufacturer import (
    ConvolutionManufacturer,
)
from nvision.sim.gen.distributions.gaussian_manufacturer import GaussianManufacturer
from nvision.sim.gen.distributions.nv_center_manufacturer import (
    MAX_K_NP,
    MAX_NV_CENTER_DELTA,
    MAX_NV_CENTER_OMEGA,
    MIN_K_NP,
    MIN_NV_CENTER_DELTA,
    MIN_NV_CENTER_OMEGA,
    NVCenterManufacturer,
)
from nvision.sim.locs import ScanBatch

NVCenterVariant = Literal["one_peak", "zeeman", "voigt_one_peak", "voigt_zeeman"]


@dataclass
class NVCenterGenerator:
    """Generator for NV center signals with different variants.

    Variants:
    - one_peak: Simple NV center with f_b = 0 (single peak)
    - zeeman: NV center with f_b != 0 (three peaks due to hyperfine splitting)
    - voigt_one_peak: Gaussian-broadened NV center with f_b = 0
    - voigt_zeeman: Gaussian-broadened NV center with f_b != 0
    """

    x_min: float = 0.0
    x_max: float = 1.0
    base: float = 0.0
    variant: NVCenterVariant = "zeeman"

    def __post_init__(self) -> None:
        if self.variant not in ("one_peak", "zeeman", "voigt_one_peak", "voigt_zeeman"):
            raise ValueError(f"Invalid variant: {self.variant}")

    def _create_manufacturer(self, rng: random.Random, delta_f_hf: float) -> PeakManufacturer:
        """Create the appropriate manufacturer based on variant."""
        omega = rng.uniform(MIN_NV_CENTER_OMEGA, MAX_NV_CENTER_OMEGA)
        k_np = rng.uniform(MIN_K_NP, MAX_K_NP)

        if self.variant == "one_peak":
            # Simple NV center with f_b = 0 (no hyperfine splitting)
            return NVCenterManufacturer(delta_f_hf=0.0, omega=omega, k_np=k_np)
        elif self.variant == "zeeman":
            # NV center with f_b != 0 (three peaks)
            return NVCenterManufacturer(delta_f_hf=delta_f_hf, omega=omega, k_np=k_np)
        sigma = delta_f_hf / 10.0 if delta_f_hf else omega / 10
        if self.variant == "voigt_one_peak":
            # Gaussian-broadened NV center with f_b = 0
            return ConvolutionManufacturer(
                NVCenterManufacturer(omega=omega, delta_f_hf=0.0, k_np=k_np),
                GaussianManufacturer(sigma=sigma),
            )
        else:  # voigt_zeeman
            # Gaussian-broadened NV center with f_b != 0
            sigma = delta_f_hf / 10.0
            return ConvolutionManufacturer(
                NVCenterManufacturer(omega=omega, delta_f_hf=delta_f_hf, k_np=k_np),
                GaussianManufacturer(sigma=sigma),
            )

    def generate(self, rng: random.Random) -> ScanBatch:
        if self.variant in ["zeeman", "voigt_zeeman"]:
            delta_f_hf = rng.uniform(MIN_NV_CENTER_DELTA, MAX_NV_CENTER_DELTA)
        else:
            delta_f_hf = 0.0

        # f_b shold be between delta and 1-delta
        x0_min = self.x_min + delta_f_hf
        x0_max = self.x_max - delta_f_hf
        x0 = rng.uniform(x0_min, x0_max)

        manufacturer = self._create_manufacturer(rng, delta_f_hf)
        f, extra_meta = manufacturer.build_peak(x0, self.base, self.x_min, self.x_max, rng)

        meta: dict[str, object] = {
            "base": self.base,
            "variant": self.variant,
            **extra_meta,
        }
        if inference := extra_meta.get("inference"):
            meta["inference"] = {"peaks": [inference]}

        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=[x0],
            signal=f,
            meta=meta,
        )
