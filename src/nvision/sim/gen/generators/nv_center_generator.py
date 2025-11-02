from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from nvision.sim.gen._protocols import PeakManufacturer
from nvision.sim.gen.distributions.convolution_manufacturer import ConvolutionManufacturer
from nvision.sim.gen.distributions.gaussian_manufacturer import GaussianManufacturer
from nvision.sim.gen.distributions.nv_center_manufacturer import NVCenterManufacturer
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

    def _create_manufacturer(self) -> PeakManufacturer:
        """Create the appropriate manufacturer based on variant."""
        if self.variant == "one_peak":
            # Simple NV center with f_b = 0 (no hyperfine splitting)
            return NVCenterManufacturer(delta_f_hf=0.0)
        elif self.variant == "zeeman":
            # NV center with f_b != 0 (three peaks)
            return NVCenterManufacturer(delta_f_hf=0.2)
        elif self.variant == "voigt_one_peak":
            # Gaussian-broadened NV center with f_b = 0
            return ConvolutionManufacturer(
                NVCenterManufacturer(omega=5, delta_f_hf=0.0), GaussianManufacturer(sigma=5)
            )
        else:  # voigt_zeeman
            # Gaussian-broadened NV center with f_b != 0
            return ConvolutionManufacturer(
                NVCenterManufacturer(omega=5, delta_f_hf=0.2), GaussianManufacturer(sigma=5)
            )

    def generate(self, rng: random.Random) -> ScanBatch:
        width = self.x_max - self.x_min
        x0 = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)

        manufacturer = self._create_manufacturer()
        f, extra_meta = manufacturer.build_peak(x0, self.base, self.x_min, self.x_max, rng)

        meta: dict[str, object] = {"base": self.base, "variant": self.variant, **extra_meta}
        if inference := extra_meta.get("inference"):
            meta["inference"] = {"peaks": [inference]}

        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=[x0],
            signal=f,
            meta=meta,
        )
