from __future__ import annotations

import random
from collections.abc import Callable

from nvision.sim.gen._protocols import PeakManufacturer


class CauchyLorentzPeakManufacturer(PeakManufacturer):
    """
    A peak manufacturer that generates a single peak with a Cauchy-Lorentz distribution shape.

    This class implements the `PeakManufacturer` protocol. Its primary role is to
    construct a callable function that represents a Lorentzian peak signal. The shape
    of the peak is defined by the equation:

        f(x) = base + amplitude / (1 + ((x - center) / gamma)^2)

    where `gamma` is the half-width at half-maximum (HWHM).

    This manufacturer is used within the simulation framework to create one of the
    fundamental signal shapes that can be measured.

    Attributes:
        amplitude (float): The peak amplitude of the Lorentzian function.
        width (float | None): The half-width at half-maximum (HWHM), also known as gamma.
            If not provided, it is estimated based on the scan range in `build_peak`.
    """

    def __init__(self, amplitude: float = 1.0, width: float | None = None, **kwargs) -> None:
        if "A" in kwargs:
            amplitude = kwargs["A"]
        self.amplitude = amplitude
        self.width = width

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        gamma = self.width if self.width is not None else max(0.05 * (x_max - x_min), 1e-6)

        def f(x: float) -> float:
            return base + self.amplitude / (1 + ((x - center) / gamma) ** 2)

        return f, {"width": float(gamma), "amplitude": self.amplitude, "mode": "cauchy"}
