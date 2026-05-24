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


def test_gmm_locator_unit_cube():
    import numpy as np

    from nvision import nv_center_lorentzian_bounds_for_domain
    from nvision.belief.unit_cube_gaussian_marginal import UnitCubeGaussianMixtureMarginalDistribution
    from nvision.spectra.unit_cube import UnitCubeSignalModel

    model = NVCenterLorentzianModel()
    bounds = nv_center_lorentzian_bounds_for_domain(2.8e9, 2.9e9)
    wrapped_model = UnitCubeSignalModel(inner=model, param_bounds_phys=bounds, x_bounds_phys=(2.8e9, 2.9e9))

    locator = GaussianMixtureLocator.create(
        signal_model=wrapped_model, parameter_bounds=bounds, scan_param="frequency", max_steps=100
    )

    assert locator is not None
    assert isinstance(locator.belief, UnitCubeGaussianMixtureMarginalDistribution)
    assert locator.belief._is_unit_cube

    obs = Observation(x=0.5, signal_value=1.0, noise_std=0.05)
    locator.observe(obs)

    locator._acquisition_bounds = lambda: (2.8e9, 2.9e9)
    next_f = locator._acquire()
    assert 2.8e9 <= next_f <= 2.9e9

    pdf_val = locator.belief.marginal_pdf("frequency", np.array([2.85e9]))
    assert pdf_val[0] > 0.0


def test_gmm_and_students_t_damping_and_unweighted_posterior():
    import numpy as np

    from nvision.belief.gaussian_mixture_marginal import (
        NVISION_GAUSSIAN_MIN_EXPLORATION_FRAC,
        NVISION_GAUSSIAN_UPDATE_DAMPING,
        GaussianMixtureMarginalDistribution,
    )
    from nvision.belief.students_t_mixture_marginal import (
        NVISION_STUDENTS_T_MIN_EXPLORATION_FRAC,
        NVISION_STUDENTS_T_UPDATE_DAMPING,
        StudentsTMixtureMarginalDistribution,
    )
    from nvision.runner.plots import _extract_gaussian_mixture_posterior, _extract_mixture_posterior

    # Verify defaults
    assert NVISION_GAUSSIAN_MIN_EXPLORATION_FRAC == 0.05
    assert NVISION_GAUSSIAN_UPDATE_DAMPING == 0.2
    assert NVISION_STUDENTS_T_MIN_EXPLORATION_FRAC == 0.05
    assert NVISION_STUDENTS_T_UPDATE_DAMPING == 0.2

    # Create dummy snapshots to test plots extraction
    class DummySnapshot:
        def __init__(self, belief):
            self.belief = belief

    model = NVCenterLorentzianModel()
    bounds = {"frequency": (2.8e9, 2.9e9)}

    # 1. GMM test
    gmm = GaussianMixtureMarginalDistribution(
        model=model,
        n_components=3,
        _physical_param_bounds=bounds,
    )
    # Set explicit weights
    gmm.weights = np.array([0.2, 0.5, 0.3])
    snapshots_gmm = [DummySnapshot(gmm)]

    out_gmm = _extract_gaussian_mixture_posterior(snapshots_gmm, ["frequency"])
    hist_gmm, _grid_gmm = out_gmm["frequency"]
    post_gmm = hist_gmm[0]

    # Shape of posterior should be (2*K + 1, N_grid) = (7, 250)
    assert post_gmm.shape == (7, 250)

    # Rows 0,1,2: unweighted comp pdfs.
    # Rows 3,4,5: weighted comp pdfs.
    # Check that row 3 is exactly weights[0] * row 0
    assert np.allclose(post_gmm[3], 0.2 * post_gmm[0])
    assert np.allclose(post_gmm[4], 0.5 * post_gmm[1])
    assert np.allclose(post_gmm[5], 0.3 * post_gmm[2])
    # Row 6: combined
    assert np.allclose(post_gmm[6], post_gmm[3] + post_gmm[4] + post_gmm[5])

    # 2. Student's T test
    st = StudentsTMixtureMarginalDistribution(
        model=model,
        n_components=3,
        _physical_param_bounds=bounds,
    )
    st.weights = np.array([0.2, 0.5, 0.3])
    snapshots_st = [DummySnapshot(st)]

    out_st = _extract_mixture_posterior(snapshots_st, ["frequency"])
    hist_st, _grid_st = out_st["frequency"]
    post_st = hist_st[0]

    assert post_st.shape == (7, 250)
    assert np.allclose(post_st[3], 0.2 * post_st[0])
    assert np.allclose(post_st[4], 0.5 * post_st[1])
    assert np.allclose(post_st[5], 0.3 * post_st[2])
    assert np.allclose(post_st[6], post_st[3] + post_st[4] + post_st[5])


def test_mixture_reflective_boundaries():
    import numpy as np

    from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution
    from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
    from nvision.spectra.nv_center import NVCenterLorentzianModel

    model = NVCenterLorentzianModel()
    bounds = {
        "frequency": (2.8e9, 2.9e9),
        "linewidth": (5e6, 15e6),
        "split": (2e6, 8e6),
        "k_np": (1.0, 2.0),
        "dip_depth": (0.01, 0.5),
    }

    # 1. GMM Reflective test
    gmm = GaussianMixtureMarginalDistribution(
        model=model,
        n_components=1,
        _physical_param_bounds=bounds,
    )
    # Place mean extremely close to upper bound (2.9e9)
    gmm.means = np.array([[2.899999e9, 10e6, 5e6, 1.5, 0.25]])
    # Perform a strong update designed to push it way past the 2.9e9 bound
    # (y_obs high, x_probe close to resonance creates a strong EKF pull/step)
    gmm._update_mixtures(x_probe=2.899e9, y_obs=100.0, sigma_eta=0.001)

    # Verify that the mean is still strictly within bounds
    # and has successfully reflected/clipped instead of being lost
    for i, name in enumerate(gmm._param_names):
        lo, hi = bounds[name]
        assert lo <= gmm.means[0, i] <= hi

    # 2. Student's T Reflective test
    st = StudentsTMixtureMarginalDistribution(
        model=model,
        n_components=1,
        _physical_param_bounds=bounds,
    )
    st.means = np.array([[2.899999e9, 10e6, 5e6, 1.5, 0.25]])
    st._update_mixtures(x_probe=2.899e9, y_obs=100.0, sigma_eta=0.001)

    for i, name in enumerate(st._param_names):
        lo, hi = bounds[name]
        assert lo <= st.means[0, i] <= hi
