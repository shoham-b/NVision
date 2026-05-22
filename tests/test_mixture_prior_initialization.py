"""Tests for GMM and Student's T mixture prior/default spaced initialization and copy operations."""

from __future__ import annotations

import numpy as np

from nvision.belief.gaussian_mixture_marginal import GaussianMixtureMarginalDistribution
from nvision.belief.students_t_mixture_marginal import StudentsTMixtureMarginalDistribution
from nvision.belief.unit_cube_gaussian_marginal import UnitCubeGaussianMixtureMarginalDistribution
from nvision.belief.unit_cube_students_t_marginal import UnitCubeStudentsTMixtureMarginalDistribution
from nvision.spectra.nv_center import NVCenterLorentzianModel
from nvision.spectra.unit_cube import UnitCubeSignalModel


def test_mixture_default_initialization():
    """Verify default spaced initialization spacing and standard deviation for frequency parameter."""
    model = NVCenterLorentzianModel()
    # Range width is 2.9e9 - 2.8e9 = 1e8
    phys_bounds = {"frequency": (2.8e9, 2.9e9)}

    # 1. Test GMM Default Spaced
    gmm = GaussianMixtureMarginalDistribution(
        model=model,
        n_components=3,
        _physical_param_bounds=phys_bounds,
    )
    # Expected means for frequency (idx 0 usually):
    # mid = 2.85e9. width = 1e8.
    # offset for k=0: (0 - 1) * width * 0.25 = -2.5e7 -> 2.825e9
    # offset for k=1: 0 * width * 0.25 = 0 -> 2.85e9
    # offset for k=2: 1 * width * 0.25 = 2.5e7 -> 2.875e9
    idx = gmm._param_names.index("frequency")
    expected_means = [2.8e9 + (k + 0.5) * (1e8 / 3.0) for k in range(3)]
    assert np.allclose(gmm.means[:, idx], expected_means)

    # Standard deviation check: std = part_width / 2.0 = (1e8 / 3) / 2 = 1e8 / 6
    # var = std^2. precision = 1 / var
    expected_prec = 1.0 / ((1e8 / 6.0) ** 2)
    assert np.allclose(gmm.precisions[:, idx, idx], expected_prec)

    # 2. Test Student's T Default Spaced
    st = StudentsTMixtureMarginalDistribution(
        model=model,
        n_components=3,
        _physical_param_bounds=phys_bounds,
    )
    assert np.allclose(st.means[:, idx], expected_means)
    assert np.allclose(st.precisions[:, idx, idx], expected_prec)


def test_mixture_prior_initialization():
    """Verify prior-based initialization for GMM and Student's T mixture distributions."""
    model = NVCenterLorentzianModel()
    phys_bounds = {"frequency": (2.8e9, 2.9e9)}
    priors = {"frequency": (2.86e9, 5e6)}  # mean = 2.86e9, std = 5e6

    # 1. Test GMM Prior sampling
    gmm = GaussianMixtureMarginalDistribution(
        model=model,
        n_components=10,  # larger component count to ensure sampling behaves correctly
        priors=priors,
        _physical_param_bounds=phys_bounds,
    )
    idx = gmm._param_names.index("frequency")
    # Means should be sampled from the prior, clipped to bounds
    for m in gmm.means[:, idx]:
        assert 2.8e9 <= m <= 2.9e9

    # Precision should match prior variance (1 / std^2)
    expected_prec = 1.0 / (5e6**2)
    assert np.allclose(gmm.precisions[:, idx, idx], expected_prec)

    # 2. Test Student's T Prior sampling
    st = StudentsTMixtureMarginalDistribution(
        model=model,
        n_components=10,
        priors=priors,
        _physical_param_bounds=phys_bounds,
    )
    for m in st.means[:, idx]:
        assert 2.8e9 <= m <= 2.9e9
    assert np.allclose(st.precisions[:, idx, idx], expected_prec)


def test_unit_cube_mixture_prior_initialization():
    """Verify prior-based initialization for Unit Cube variants."""
    model = NVCenterLorentzianModel()
    phys_bounds = {
        "frequency": (2.8e9, 2.9e9),
        "linewidth": (5e6, 15e6),
        "split": (2e6, 8e6),
        "k_np": (1.0, 2.0),
        "dip_depth": (0.01, 0.5),
    }
    wrapped_model = UnitCubeSignalModel(
        inner=model,
        param_bounds_phys=phys_bounds,
        x_bounds_phys=(2.8e9, 2.9e9),
    )
    priors = {"frequency": (2.86e9, 5e6)}  # physical space prior

    # 1. Test UnitCube GMM Prior
    gmm = UnitCubeGaussianMixtureMarginalDistribution(
        model=wrapped_model,
        n_components=10,
        priors=priors,
        _physical_param_bounds=phys_bounds,
    )
    idx = gmm._param_names.index("frequency")

    # Internal means should be in [0, 1] unit space
    # Prior mean 2.86e9 -> unit space (2.86e9 - 2.8e9) / 1e8 = 0.6
    # Prior std 5e6 -> unit space 5e6 / 1e8 = 0.05
    for m in gmm.means[:, idx]:
        assert 0.0 <= m <= 1.0

    expected_prec = 1.0 / (0.05**2)
    assert np.allclose(gmm.precisions[:, idx, idx], expected_prec)

    # 2. Test UnitCube Student's T Prior
    st = UnitCubeStudentsTMixtureMarginalDistribution(
        model=wrapped_model,
        n_components=10,
        priors=priors,
        _physical_param_bounds=phys_bounds,
    )
    for m in st.means[:, idx]:
        assert 0.0 <= m <= 1.0
    assert np.allclose(st.precisions[:, idx, idx], expected_prec)


def test_mixture_copy_propagation():
    """Verify copy operations replicate priors and protective params."""
    model = NVCenterLorentzianModel()
    phys_bounds = {"frequency": (2.8e9, 2.9e9)}
    priors = {"frequency": (2.86e9, 5e6)}

    # GMM Copy
    gmm = GaussianMixtureMarginalDistribution(
        model=model,
        n_components=3,
        weight_floor=0.08,
        weight_floor_steps=20,
        priors=priors,
        _physical_param_bounds=phys_bounds,
    )
    gmm_copy = gmm.copy()
    assert gmm_copy.priors == priors
    assert gmm_copy.weight_floor == 0.08
    assert gmm_copy.weight_floor_steps == 20
    assert np.allclose(gmm_copy.means, gmm.means)
    assert np.allclose(gmm_copy.precisions, gmm.precisions)

    # Student's T Copy
    st = StudentsTMixtureMarginalDistribution(
        model=model,
        n_components=3,
        weight_floor=0.08,
        weight_floor_steps=20,
        priors=priors,
        _physical_param_bounds=phys_bounds,
    )
    st_copy = st.copy()
    assert st_copy.priors == priors
    assert st_copy.weight_floor == 0.08
    assert st_copy.weight_floor_steps == 20
    assert np.allclose(st_copy.means, st.means)
    assert np.allclose(st_copy.precisions, st.precisions)

    # Unit Cube GMM Copy
    wrapped_model = UnitCubeSignalModel(
        inner=model,
        param_bounds_phys=phys_bounds,
        x_bounds_phys=(2.8e9, 2.9e9),
    )
    uc_gmm = UnitCubeGaussianMixtureMarginalDistribution(
        model=wrapped_model,
        n_components=3,
        weight_floor=0.08,
        weight_floor_steps=20,
        priors=priors,
        _physical_param_bounds=phys_bounds,
    )
    uc_gmm_copy = uc_gmm.copy()
    assert uc_gmm_copy.priors == priors
    assert uc_gmm_copy.weight_floor == 0.08
    assert uc_gmm_copy.weight_floor_steps == 20
    assert np.allclose(uc_gmm_copy.means, uc_gmm.means)

    # Unit Cube Student's T Copy
    uc_st = UnitCubeStudentsTMixtureMarginalDistribution(
        model=wrapped_model,
        n_components=3,
        weight_floor=0.08,
        weight_floor_steps=20,
        priors=priors,
        _physical_param_bounds=phys_bounds,
    )
    uc_st_copy = uc_st.copy()
    assert uc_st_copy.priors == priors
    assert uc_st_copy.weight_floor == 0.08
    assert uc_st_copy.weight_floor_steps == 20
    assert np.allclose(uc_st_copy.means, uc_st.means)


def test_frequency_linewidth_uncertainty_constraint():
    """Verify that frequency uncertainty is always at least the linewidth uncertainty."""
    model = NVCenterLorentzianModel()
    phys_bounds = {
        "frequency": (2.8e9, 2.9e9),  # width = 1e8
        "linewidth": (5e6, 15e6),  # width = 1e7
        "split": (0.0, 1e7),
        "k_np": (0.0, 1.0),
        "dip_depth": (0.0, 1.0),
    }

    # 1. Test GMM Physical
    gmm = GaussianMixtureMarginalDistribution(
        model=model,
        n_components=3,
        _physical_param_bounds=phys_bounds,
    )
    freq_idx = gmm._param_names.index("frequency")
    lw_idx = gmm._param_names.index("linewidth")

    # Set exact non-eps-dominated precision diagonal matrices to violate the constraint
    for k in range(gmm.n_components):
        gmm.precisions[k] = np.eye(gmm._dim)
        gmm.precisions[k, lw_idx, lw_idx] = 0.01  # var_lw = 100
        gmm.precisions[k, freq_idx, freq_idx] = 1.0  # var_freq = 1.0 < var_lw

    gmm._recompute_covariances()

    # Verify that component covariances are corrected
    for k in range(gmm.n_components):
        assert gmm._covariances[k, freq_idx, freq_idx] >= gmm._covariances[k, lw_idx, lw_idx]
        assert np.allclose(gmm.precisions[k, freq_idx, freq_idx], 1.0 / gmm._covariances[k, freq_idx, freq_idx])

    # Verify that combined empirical uncertainty also satisfies the constraint
    emp_unc = gmm._empirical_uncertainty()
    assert emp_unc["frequency"] >= emp_unc["linewidth"]

    # 2. Test GMM Unit Cube (wrapped)
    wrapped_model = UnitCubeSignalModel(
        inner=model,
        param_bounds_phys=phys_bounds,
        x_bounds_phys=(2.8e9, 2.9e9),
    )
    uc_gmm = UnitCubeGaussianMixtureMarginalDistribution(
        model=wrapped_model,
        n_components=3,
        _physical_param_bounds=phys_bounds,
    )
    # Unit-cube GMM is actually an instance of GaussianMixtureMarginalDistribution with self._is_unit_cube = True
    # We want to violate the constraint in unit cube.
    # lw width = 1e7, freq width = 1e8.
    # scale_factor = (lw_w / freq_w)^2 = 0.01.
    # var_freq_min = var_lw * 0.01.
    # If we set var_lw = 100 (prec = 0.01), then var_freq_min = 1.0.
    # We choose var_freq = 0.25 (prec = 4.0).
    # Since var_freq = 0.25 < var_freq_min = 1.0, it violates the constraint and should trigger clamping.
    for k in range(uc_gmm.n_components):
        uc_gmm.precisions[k] = np.eye(uc_gmm._dim)
        uc_gmm.precisions[k, lw_idx, lw_idx] = 0.01
        uc_gmm.precisions[k, freq_idx, freq_idx] = 4.0

    uc_gmm._recompute_covariances()

    # In unit cube, the physical standard deviation is returned by public APIs,
    # but the internal _covariances are in unit cube.
    # Let's verify internal _covariances are clamped:
    for k in range(uc_gmm.n_components):
        # internal freq_cov should be clamped to var_lw * 0.01 = 1.0
        assert uc_gmm._covariances[k, freq_idx, freq_idx] >= 0.9
        assert uc_gmm.precisions[k, freq_idx, freq_idx] <= 1.1

    # Verify that public empirical uncertainty (which is mapped to physical) also satisfies the constraint
    emp_unc_uc = uc_gmm.uncertainty()
    assert emp_unc_uc["frequency"] >= emp_unc_uc["linewidth"]

    # 3. Test Student's T Physical
    st = StudentsTMixtureMarginalDistribution(
        model=model,
        n_components=3,
        _physical_param_bounds=phys_bounds,
    )
    for k in range(st.n_components):
        st.precisions[k] = np.eye(st._dim)
        st.precisions[k, lw_idx, lw_idx] = 0.01  # var_lw = 100
        st.precisions[k, freq_idx, freq_idx] = 1.0  # var_freq = 1.0

    st._recompute_covariances()

    for k in range(st.n_components):
        assert st._covariances[k, freq_idx, freq_idx] >= st._covariances[k, lw_idx, lw_idx]
        assert np.allclose(st.precisions[k, freq_idx, freq_idx], 1.0 / st._covariances[k, freq_idx, freq_idx])

    emp_unc_st = st._empirical_uncertainty()
    assert emp_unc_st["frequency"] >= emp_unc_st["linewidth"]

    # 4. Test Student's T Unit Cube (wrapped)
    uc_st = UnitCubeStudentsTMixtureMarginalDistribution(
        model=wrapped_model,
        n_components=3,
        _physical_param_bounds=phys_bounds,
    )
    for k in range(uc_st.n_components):
        uc_st.precisions[k] = np.eye(uc_st._dim)
        uc_st.precisions[k, lw_idx, lw_idx] = 0.01
        uc_st.precisions[k, freq_idx, freq_idx] = 4.0

    uc_st._recompute_covariances()

    for k in range(uc_st.n_components):
        assert uc_st._covariances[k, freq_idx, freq_idx] >= 0.9
        assert uc_st.precisions[k, freq_idx, freq_idx] <= 1.1

    emp_unc_uc_st = uc_st.uncertainty()
    assert emp_unc_uc_st["frequency"] >= emp_unc_uc_st["linewidth"]
