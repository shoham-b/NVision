"""Composite peak model combining multiple independent signal signal."""

from __future__ import annotations

from nvision.signal.signal import Parameter, SignalModel


class CompositePeakModel(SignalModel):
    """Multiple independent peaks summed together.

    Combines multiple peak signal into a composite signal.
    Parameters are organized as: [peak1_params..., peak2_params..., ...]
    with prefixes like "peak1_", "peak2_" to distinguish them.
    """

    def __init__(self, peak_models: list[tuple[str, SignalModel]]):
        """Initialize composite model.

        Parameters
        ----------
        peak_models : list[tuple[str, SignalModel]]
            List of (prefix, model) pairs. Prefix is used to namespace parameters.
            Example: [("peak1", LorentzianModel()), ("peak2", GaussianModel())]
        """
        self.peak_models = peak_models

    def compute(self, x: float, params: list[Parameter]) -> float:
        total = 0.0
        for prefix, model in self.peak_models:
            peak_params = [p for p in params if p.name.startswith(f"{prefix}_")]
            unprefixed_params = [
                Parameter(
                    name=p.name[len(prefix) + 1 :],
                    bounds=p.bounds,
                    value=p.value,
                )
                for p in peak_params
            ]
            total += model.compute(x, unprefixed_params)
        return total

    def parameter_names(self) -> list[str]:
        """Return flattened parameter names with prefixes.

        Returns
        -------
        list[str]
            ["peak1_frequency", "peak1_amplitude", ...,
             "peak2_frequency", "peak2_amplitude", ...]
        """
        names = []
        for prefix, model in self.peak_models:
            for param_name in model.parameter_names():
                names.append(f"{prefix}_{param_name}")
        return names
