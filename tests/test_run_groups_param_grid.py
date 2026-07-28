"""SBED run-groups default to a saturation-Voigt (contrast x inhomogeneous) grid."""

from __future__ import annotations

import numpy as np
import pytest

import nvision.sim.defaults as sim_defaults
from nvision.sim import presets as sim_presets
from nvision.sim import run_groups as sim_run_groups
from nvision.sim.combinations import CombinationGrid
from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar


def test_lorentzian_sbed_group_has_full_saturation_voigt_grid():
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("lorentzian-sbed")
    expected = sim_defaults.NVISION_SBED_CONTRAST_STEPS * sim_defaults.NVISION_SBED_SIGMA_INHOM_STEPS
    assert len(group.generator_names) == expected
    assert len(set(group.generator_names)) == expected
    assert group.extra_generators is not None
    assert set(group.extra_generators.keys()) == set(group.generator_names)
    for name in group.generator_names:
        assert name.startswith("NVCenter-saturation_voigt-c")
    for gen in group.extra_generators.values():
        assert gen.variant == "saturation_voigt"


def test_saturation_voigt_grid_hits_its_target_contrast_exactly():
    """The point of sweeping contrast directly: every grid point's *realized*
    contrast (recomputed from the solved saturation) must match the intended
    target contrast, not just approximate it. Compares against the same
    np.linspace the generator builds internally -- not the 2-decimal-rounded
    name string, which loses precision (e.g. 0.175 formats as "c0.18")."""
    target_contrasts = np.linspace(
        sim_defaults.NVISION_SBED_CONTRAST_MIN,
        sim_defaults.NVISION_SBED_CONTRAST_MAX,
        sim_defaults.NVISION_SBED_CONTRAST_STEPS,
    )
    generators = dict(sim_presets.saturation_voigt_param_grid_generators())
    for gen in generators.values():
        _fwhm_total, _lorentz_frac, realized_contrast = _saturation_voigt_reparam_scalar(
            gen.saturation, gen.sigma_inhom, NV_SATURATION_C_MAX
        )
        assert any(realized_contrast == pytest.approx(float(c), abs=1e-9) for c in target_contrasts)


def test_saturation_voigt_grid_rejects_c_max_at_or_below_contrast_max(monkeypatch):
    """Contrast can only approach c_max asymptotically -- c_max must stay strictly
    above the highest contrast the grid targets, or the inversion is unphysical."""
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_CONTRAST_MAX", 0.4)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_C_MAX", 0.4)
    with pytest.raises(ValueError, match="NVISION_SBED_CONTRAST_MAX"):
        sim_presets.saturation_voigt_param_grid_generators()


def test_lorentzian_sbed_group_noise_grid_uses_dedicated_sbed_config(monkeypatch):
    """Noise is swept the same way as saturation/sigma_inhom, via its own dedicated
    NVISION_SBED_NOISE_* config -- independent of the generic noise globals
    used by plain `nvision run` (which may be tuned very differently, e.g.
    via .env, for unrelated purposes)."""
    monkeypatch.setattr(sim_defaults, "NVISION_NOISE_MAX_GAUSS", 999.0)
    monkeypatch.setattr(sim_defaults, "NVISION_NOISE_GAUSS_STEPS", 999)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_NOISE_MIN", 0.0)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_NOISE_MAX", 0.1)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_NOISE_STEPS", 3)

    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("lorentzian-sbed")

    assert group.noise_names == ["Gauss(0.0)", "Gauss(0.05)", "Gauss(0.1)"]
    sim_run_groups.clear_run_group_cache()


def test_all_three_sbed_variant_groups_share_the_same_param_grid():
    sim_run_groups.clear_run_group_cache()
    names = {g.name: set(g.generator_names) for g in sim_run_groups.run_groups()}
    assert names["lorentzian-sbed"] == names["lorentzian-sbed-only"]
    assert names["lorentzian-sbed"] == names["lorentzian-sweep-only"]


def test_combination_grid_resolves_swept_generator_names_without_explicit_extra():
    """cache/metrics/render tooling call CombinationGrid() bare and must still
    resolve historical results produced by a run-group's parameter grid."""
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("lorentzian-sbed")
    a_name = group.generator_names[0]
    a_noise = group.noise_names[0]

    grid = CombinationGrid()  # no extra_generators passed
    combo = grid.resolve(a_name, a_noise, "Bayesian-SBED")
    assert combo is not None
    assert combo.generator_name == a_name


def test_bayesian_strategy_config_uses_saturation_voigt_lineshape_for_study_grid():
    """The belief must be built with the matching lineshape, or inference models
    the wrong signal shape entirely."""
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("lorentzian-sbed")
    a_name = group.generator_names[0]

    grid = CombinationGrid()
    combo = grid.resolve(a_name, group.noise_names[0], "Bayesian-SBED")
    assert combo.strategy["config"]["lineshape"] == "saturation_voigt"


def test_plain_lorentzian_generator_keeps_default_lineshape():
    """Plain `nvision run` generators (unrelated to the SBED study grid) must
    not be affected -- lineshape stays unset, defaulting to lorentzian."""
    grid = CombinationGrid()
    combo = grid.resolve("NVCenter-lorentzian", "Gauss(0.01)", "Bayesian-SBED")
    assert combo is not None
    assert "lineshape" not in combo.strategy["config"]


def test_combination_grid_iteration_unaffected_by_sbed_param_grid():
    """Plain `nvision run` (no --run-group) must keep enumerating only the
    default generators; the SBED grid must not leak into iter()/all_combinations()."""
    default_names = {
        "NVCenter-lorentzian",
        "NVCenter-voigt",
        "NVCenter-inhom-0",
        "NVCenter-inhom-low",
        "NVCenter-inhom-high",
    }
    grid = CombinationGrid()
    assert set(grid.generators.keys()) == default_names
    combos = list(grid.iter())
    names = {c.generator_name for c in combos}
    assert names == default_names


def test_voigt_sbed_group_has_full_width_contrast_grid():
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("voigt-sbed")
    expected = sim_defaults.NVISION_SBED_WIDTH_STEPS * sim_defaults.NVISION_SBED_CONTRAST_STEPS
    assert len(group.generator_names) == expected
    assert len(set(group.generator_names)) == expected
    assert group.extra_generators is not None
    assert set(group.extra_generators.keys()) == set(group.generator_names)
    for name in group.generator_names:
        assert name.startswith("NVCenter-voigt-w")
    for gen in group.extra_generators.values():
        assert gen.variant == "voigt"


def test_all_three_voigt_variant_groups_share_the_same_param_grid():
    sim_run_groups.clear_run_group_cache()
    names = {g.name: set(g.generator_names) for g in sim_run_groups.run_groups()}
    assert names["voigt-sbed"] == names["voigt-sbed-only"]
    assert names["voigt-sbed"] == names["voigt-sweep-only"]


def test_bayesian_strategy_config_uses_voigt_lineshape_for_width_contrast_grid():
    """The belief must be built with the matching lineshape, or inference models
    the wrong signal shape entirely."""
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("voigt-sbed")
    a_name = group.generator_names[0]

    grid = CombinationGrid()
    combo = grid.resolve(a_name, group.noise_names[0], "Bayesian-SBED")
    assert combo.strategy["config"]["lineshape"] == "voigt"


def test_bare_voigt_generator_keeps_default_lineshape():
    """The bare NVCenter-voigt generator (plain `nvision run`, no run-group) must not
    be affected -- lineshape stays unset (pre-existing gap, deliberately out of scope)."""
    grid = CombinationGrid()
    combo = grid.resolve("NVCenter-voigt", "Gauss(0.01)", "Bayesian-SBED")
    assert combo is not None
    assert "lineshape" not in combo.strategy["config"]
