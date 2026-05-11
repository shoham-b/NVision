import numpy as np
from nvision.spectra.nv_center import NVCenterLorentzianModel, nv_center_lorentzian_bounds_for_domain
from nvision.spectra.noise_model import GaussianNoiseSignalModel
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution

sig_model = NVCenterLorentzianModel()
noise_model = GaussianNoiseSignalModel(prior_bounds={'noise_sigma': (0.01, 0.1)})
bounds = nv_center_lorentzian_bounds_for_domain(2.8e9, 2.9e9)
bounds.update(noise_model.spec.bounds)
wrapped = UnitCubeSignalModel(sig_model, bounds, x_bounds_phys=(2.8e9, 2.9e9))

param_bounds = {name: (0.0, 1.0) for name in bounds}

class DebugDist(UnitCubeSMCMarginalDistribution):
    def __post_init__(self) -> None:
        if not isinstance(self.model, UnitCubeSignalModel):
            raise TypeError("UnitCubeSMCMarginalDistribution requires a UnitCubeSignalModel")

        print(f"model.parameter_names(): {self.model.parameter_names()}")

        bounds = {name: (0.0, 1.0) for name in self.model.parameter_names()}
        if self.noise_model is not None:
            for name in self.noise_model.spec.names:
                bounds[name] = (0.0, 1.0)
        self.parameter_bounds = bounds
        print(f"self.parameter_bounds keys: {self.parameter_bounds.keys()}")
        super().__post_init__()

try:
    belief = DebugDist(
        model=wrapped,
        noise_model=noise_model,
        parameter_bounds=param_bounds,
        physical_param_bounds=bounds,
        num_particles=100,
    )
except Exception as e:
    import traceback
    traceback.print_exc()
