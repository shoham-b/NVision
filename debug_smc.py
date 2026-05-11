import numpy as np
from nvision.spectra.nv_center import NVCenterLorentzianModel, nv_center_lorentzian_bounds_for_domain
from nvision.spectra.noise_model import GaussianNoiseSignalModel
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.belief.smc_marginal import SMCMarginalDistribution
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution

sig_model = NVCenterLorentzianModel()
noise_model = GaussianNoiseSignalModel(prior_bounds={'noise_sigma': (0.01, 0.1)})
bounds = nv_center_lorentzian_bounds_for_domain(2.8e9, 2.9e9)
bounds.update(noise_model.spec.bounds)
wrapped = UnitCubeSignalModel(sig_model, bounds, x_bounds_phys=(2.8e9, 2.9e9))

param_bounds = {name: (0.0, 1.0) for name in bounds}

class CustomUnitCubeSMCMarginalDistribution(SMCMarginalDistribution):
    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeSMCMarginalDistribution requires a UnitCubeSignalModel")

        # Force parameter_bounds to unit space for internal SMC operations.
        # This ensures super().__post_init__ initializes particles in [0, 1].
        bounds = {name: (0.0, 1.0) for name in self.model.parameter_names()}
        if self.noise_model is not None:
            for name in self.noise_model.spec.names:
                bounds[name] = (0.0, 1.0)
        self.parameter_bounds = bounds
        print(f"Set self.parameter_bounds to: {self.parameter_bounds}")
        super().__post_init__()

# Now check UnitCubeSMCMarginalDistribution
print(f"UnitCubeSMCMarginalDistribution parameter_bounds before init: {param_bounds}")
b = UnitCubeSMCMarginalDistribution(
    model=wrapped,
    noise_model=noise_model,
    parameter_bounds=param_bounds,
    num_particles=100,
)
print("Success!")
