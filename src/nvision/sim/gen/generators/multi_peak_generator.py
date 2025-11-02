from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from nvision.sim.gen._protocols import PeakManufacturer
from nvision.sim.locs import ScanBatch


@dataclass
class MultiPeakGenerator:
    x_min: float = 0.0
    x_max: float = 1.0
    count: int = 1
    base: float = 0.0
    min_sep_frac: float = 0.1
    manufacturers: list[PeakManufacturer] | None = None
    manufacturer: PeakManufacturer | None = None

    def generate(self, rng: random.Random) -> ScanBatch:
        if self.manufacturers is None and self.manufacturer is None:
            raise ValueError("MultiPeakGenerator requires manufacturer(s)")
        width = self.x_max - self.x_min
        xs: list[float] = []
        while len(xs) < max(1, self.count):
            x = rng.uniform(self.x_min + 0.05 * width, self.x_max - 0.05 * width)
            if not xs or all(abs(x - xi) >= self.min_sep_frac * width for xi in xs):
                xs.append(x)
        fns: list[Callable[[float], float]] = []
        peak_params: list[dict[str, float]] = []
        inference_peaks: list[dict[str, float]] = []
        for i, xc in enumerate(xs):
            manuf = (
                self.manufacturers[i]
                if self.manufacturers is not None and i < len(self.manufacturers)
                else self.manufacturer
            )
            if manuf is None:
                raise ValueError("Missing manufacturer for peak index {i}")
            f_i, params = manuf.build_peak(xc, self.base, self.x_min, self.x_max, rng)
            fns.append(f_i)
            peak_params.append(params)
            if inf := params.get("inference"):
                inference_peaks.append(inf)

        def f(x: float) -> float:
            return sum(fi(x) for fi in fns) - (len(fns) - 1) * self.base

        xs_sorted = sorted(xs)
        meta: dict[str, object] = {
            "base": self.base,
            "peak_params": peak_params,
        }
        if inference_peaks:
            meta["inference"] = {"peaks": inference_peaks}
        return ScanBatch(
            x_min=self.x_min,
            x_max=self.x_max,
            truth_positions=xs_sorted,
            signal=f,
            meta=meta,
        )
