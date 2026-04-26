def install_rich_tracebacks() -> None:
    """Configure rich traceback to show relevant stack frames only."""
    import concurrent.futures
    import multiprocessing

    import typer
    from rich.traceback import install

    try:
        import numba

        suppress = [typer, multiprocessing, concurrent.futures, numba]
    except ImportError:
        suppress = [typer, multiprocessing, concurrent.futures]

    install(show_locals=False, suppress=suppress, width=100, word_wrap=True)


# Note: We don't call it here to avoid early import side-effects.
# The CLI entrypoints will call it as needed.

# Public API exports
from nvision.belief.unit_cube_grid_marginal import UnitCubeGridMarginalDistribution
from nvision.belief.unit_cube_smc_marginal import UnitCubeSMCMarginalDistribution
from nvision.cache.hashing import stable_config_hash
from nvision.cache.locator_keys import locator_combination_cache_config
from nvision.cache.locator_repository import LocatorResultsRepository
from nvision.cli.app_instance import app
from nvision.models.experiment import CoreExperiment
from nvision.models.locator import Locator
from nvision.models.noise import CompositeNoise, CompositeOverFrequencyNoise
from nvision.models.observer import Observer, RunResult
from nvision.runner.executor import run_loop
from nvision.runner.repeat_keys import (
    measurement_repeat_key,
    repeat_seed_int,
    signal_repeat_key,
)
from nvision.runner.signal_cache import clear_signal_experiment_cache, get_shared_core_experiment
from nvision.sim.batch import DataBatch
from nvision.sim.combinations import CombinationGrid
from nvision.sim.gen import (
    DEFAULT_NV_CENTER_FREQ_X_MAX,
    DEFAULT_NV_CENTER_FREQ_X_MIN,
    GAUSSIAN,
    LORENTZIAN,
    NVCenterCoreGenerator,
    PeakSpec,
    nv_center_lorentzian_bounds_for_domain,
)
from nvision.sim.locs.coarse import SimpleSweepLocator, StagedSobolSweepLocator
from nvision.sim.locs.bayesian import nv_center_belief
from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief
from nvision.noises import (
    OverFrequencyGaussianNoise,
    OverFrequencyOutlierSpikes,
    OverFrequencyPoissonNoise,
)
from nvision.noises.groups import OverProbeNoises as CompositeOverProbeNoise
from nvision.noises.over_probe.drift_noise import OverProbeDriftNoise
from nvision.noises.over_probe.random_walk_noise import OverProbeRandomWalkNoise
from nvision.spectra.gaussian import GaussianModel, GaussianSpectrum
from nvision.spectra.likelihood import likelihood_from_observation_model
from nvision.spectra.lorentzian import LorentzianModel, LorentzianSpectrum
from nvision.spectra.signal import TrueSignal
from nvision.spectra.unit_cube import UnitCubeSignalModel
from nvision.spectra.voigt_zeeman import VoigtZeemanModel
from nvision.viz.measurements import (
    backfill_scan_plot_data_if_missing,
    compute_scan_plot_data,
    plot_data_from_scan_figure,
)

__all__ = [
    "install_rich_tracebacks",
    # Combinations & CLI
    "CombinationGrid",
    "app",
    # Runner
    "measurement_repeat_key",
    "repeat_seed_int",
    "signal_repeat_key",
    "run_loop",
    # Generators
    "NVCenterCoreGenerator",
    "DEFAULT_NV_CENTER_FREQ_X_MAX",
    "DEFAULT_NV_CENTER_FREQ_X_MIN",
    "nv_center_lorentzian_bounds_for_domain",
    "nv_center_belief",
    "nv_center_smc_belief",
    # Peak specs
    "PeakSpec",
    "GAUSSIAN",
    "LORENTZIAN",
    "EXPONENTIAL",
    # Models
    "CoreExperiment",
    "DataBatch",
    "CompositeNoise",
    "CompositeOverFrequencyNoise",
    "CompositeOverProbeNoise",
    "OverFrequencyGaussianNoise",
    "OverFrequencyOutlierSpikes",
    "OverFrequencyPoissonNoise",
    "OverFrequencyNoises",
    "OverProbeNoises",
    "OverProbeDriftNoise",
    "OverProbeRandomWalkNoise",
    "Observer",
    "RunResult",
    "Parameter",
    "TrueSignal",
    "UnitCubeSignalModel",
    "GaussianModel",
    "GaussianSpectrum",
    "LorentzianModel",
    "LorentzianSpectrum",
    "VoigtZeemanModel",
    # Locators
    "Locator",
    "SimpleSweepLocator",
    "StagedSobolSweepLocator",
    # Spectra
    "likelihood_from_observation_model",
    # Cache
    "stable_config_hash",
    "backfill_scan_plot_data_if_missing",
    # Belief
    "UnitCubeGridMarginalDistribution",
    "UnitCubeSMCMarginalDistribution",
    # Cache
    "locator_combination_cache_config",
    "LocatorResultsRepository",
    "clear_signal_experiment_cache",
    "get_shared_core_experiment",
    # Viz
    "compute_scan_plot_data",
    "plot_data_from_scan_figure",
]
