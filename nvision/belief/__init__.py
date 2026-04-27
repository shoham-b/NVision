"""Posterior belief distributions: discrete grids, SMC, and unit-cube wrappers."""

from nvision.belief.abstract_marginal import AbstractMarginalDistribution, ParameterValues
from nvision.belief.grid_marginal import GridMarginalDistribution, GridParameter
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution

__all__ = [
    "AbstractMarginalDistribution",
    "GridMarginalDistribution",
    "GridParameter",
    "ParameterValues",
    "SMCMarginalDistribution",
    "StudentsTMixtureMarginalDistribution",
    "UnitCubeGridMarginalDistribution",
    "UnitCubeSMCMarginalDistribution",
]
