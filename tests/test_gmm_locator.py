from nvision.models.observation import Observation
from nvision.sim.locs.ekf.gmm_locator import GaussianMixtureLocator
from nvision.spectra.nv_center import NVCenterLorentzianModel


def test_gmm_locator_create():
    model = NVCenterLorentzianModel()
    locator = GaussianMixtureLocator.create(
        signal_model=model, parameter_bounds={"frequency": (2.8e9, 2.9e9)}, max_steps=100
    )
    assert locator is not None
    assert locator.belief.n_components == 5


def test_gmm_locator_acquire_within_bounds():
    model = NVCenterLorentzianModel()
    locator = GaussianMixtureLocator.create(
        signal_model=model, parameter_bounds={"frequency": (2.8e9, 2.9e9)}, scan_param="frequency", max_steps=100
    )

    # In a full run, SequentialBayesianLocator initialization uses domain limits
    # and sets these internally through start_sweep etc.
    # We mock _acquisition_bounds directly to isolate the test to the GMM behavior.
    locator._acquisition_bounds = lambda: (2.8e9, 2.9e9)

    # Provide an observation to initialize the belief state correctly for EIG calculation
    obs = Observation(x=0.5, signal_value=1.0, noise_std=0.05)
    locator.observe(obs)

    next_f = locator._acquire()
    assert isinstance(next_f, float)
    assert 2.8e9 <= next_f <= 2.9e9


def test_gmm_locator_on_sweep_complete():
    model = NVCenterLorentzianModel()
    locator = GaussianMixtureLocator.create(
        signal_model=model, parameter_bounds={"frequency": (2.8e9, 2.9e9)}, scan_param="frequency", max_steps=100
    )
    locator._acquisition_bounds = lambda: (2.82e9, 2.88e9)
    locator._on_sweep_complete()
    assert locator.belief.physical_param_bounds["frequency"] == (2.82e9, 2.88e9)
