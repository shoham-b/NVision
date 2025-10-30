from __future__ import annotations

import random
from collections.abc import Callable

from .._protocols import PeakManufacturer


class ConvolutionManufacturer(PeakManufacturer):
    """A manufacturer that creates a peak shape by convolving two other distributions."""

    def __init__(self, manufacturer1: PeakManufacturer, manufacturer2: PeakManufacturer):
        self.manufacturer1 = manufacturer1
        self.manufacturer2 = manufacturer2

    def build_peak(
        self,
        center: float,
        base: float,
        x_min: float,
        x_max: float,
        rng: random.Random,
    ) -> tuple[Callable[[float], float], dict[str, float]]:
        """
        Builds a peak by attempting to convolve the two manufacturers.
        It delegates the convolution logic to the manufacturers themselves.
        """
        try:
            # Let manufacturer1 handle the convolution logic.
            convolved_manufacturer = self.manufacturer1.convolve(self.manufacturer2)
            return convolved_manufacturer.build_peak(center, base, x_min, x_max, rng)
        except NotImplementedError as e:
            raise NotImplementedError(
                "Convolution for this pair of manufacturers is not supported. "
                f"({type(self.manufacturer1).__name__} + {type(self.manufacturer2).__name__})"
            ) from e
