"""Build an SMC belief over an arbitrary physical model, for tests that are not NV-specific."""

from __future__ import annotations

from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.spectra.unit_cube import UnitCubeSignalModel


def make_smc(
    model,
    physical_bounds: dict[str, tuple[float, float]],
    cls: type[SMCMarginalDistribution] = SMCMarginalDistribution,
    **kwargs,
) -> SMCMarginalDistribution:
    """SMC belief whose particles live on the unit cube of ``physical_bounds``.

    ``physical_bounds["frequency"]`` (if present) is the probe axis; otherwise the probe axis is ``[0, 1]``.
    """
    x_bounds = physical_bounds.get("frequency", (0.0, 1.0))
    return cls(
        model=UnitCubeSignalModel(model, dict(physical_bounds), x_bounds),
        physical_param_bounds=dict(physical_bounds),
        physical_x_bounds=x_bounds,
        **kwargs,
    )
