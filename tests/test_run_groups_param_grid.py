"""Run-group registry and the study parameter grids they (and historical results) use."""

from __future__ import annotations

import numpy as np
import pytest

import nvision.sim.defaults as sim_defaults
from nvision.sim import presets as sim_presets
from nvision.sim import run_groups as sim_run_groups
from nvision.sim.combinations import CombinationGrid
from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar


def test_both_sbed_group_uses_voigt_inhom_grid():
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("both-sbed")
    assert group.generator_names
    assert len(set(group.generator_names)) == len(group.generator_names)
    assert group.extra_generators is not None
    assert set(group.extra_generators.keys()) == set(group.generator_names)
    assert group.strategy_names == ["Bayesian-SBED", "SimpleSobol", "SimpleSweep"]


def test_both_variant_groups_share_the_same_param_grid():
    sim_run_groups.clear_run_group_cache()
    groups = {g.name: g for g in sim_run_groups.run_groups()}
    assert set(groups["both-sbed"].generator_names) == set(groups["both-sbed-only"].generator_names)
    assert set(groups["both-sbed"].generator_names) == set(groups["both-sweep-only"].generator_names)
    assert groups["both-sbed-only"].strategy_names == ["Bayesian-SBED"]
    assert groups["both-sweep-only"].strategy_names == ["SimpleSweep"]


def test_default_run_group_is_both_sbed():
    assert sim_run_groups.default_run_group().name == "both-sbed"


def test_both_sbed_group_noise_grid_uses_dedicated_sbed_config(monkeypatch):
    """Noise is swept via its own dedicated NVISION_SBED_NOISE_* config -- independent of
    the generic noise globals used by plain `nvision run`."""
    monkeypatch.setattr(sim_defaults, "NVISION_NOISE_MAX_GAUSS", 999.0)
    monkeypatch.setattr(sim_defaults, "NVISION_NOISE_GAUSS_STEPS", 999)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_NOISE_MIN", 0.0)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_NOISE_MAX", 0.1)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_NOISE_STEPS", 3)

    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("both-sbed")

    assert group.noise_names == ["Gauss(0.0)", "Gauss(0.05)", "Gauss(0.1)"]
    sim_run_groups.clear_run_group_cache()


def test_saturation_voigt_grid_hits_its_target_contrast_exactly():
    """Every grid point's *realized* contrast (recomputed from the solved saturation) must
    match the intended target contrast, not just approximate it."""
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


def test_combination_grid_resolves_swept_generator_names_without_explicit_extra():
    """cache/metrics/render tooling call CombinationGrid() bare and must still
    resolve historical results produced by a study parameter grid."""
    a_name = next(iter(dict(sim_presets.saturation_voigt_param_grid_generators())))
    a_noise = sim_presets.sbed_study_noises()[0][0]

    grid = CombinationGrid()  # no extra_generators passed
    combo = grid.resolve(a_name, a_noise, "Bayesian-SBED")
    assert combo is not None
    assert combo.generator_name == a_name


def test_bayesian_strategy_config_uses_saturation_voigt_lineshape_for_study_grid():
    """The belief must be built with the matching lineshape, or inference models
    the wrong signal shape entirely."""
    a_name = next(iter(dict(sim_presets.saturation_voigt_param_grid_generators())))

    combo = CombinationGrid().resolve(a_name, sim_presets.sbed_study_noises()[0][0], "Bayesian-SBED")
    assert combo.strategy["config"]["lineshape"] == "saturation_voigt"


def test_bayesian_strategy_config_uses_voigt_lineshape_for_width_contrast_grid():
    a_name = next(iter(dict(sim_presets.param_grid_generators(variant="voigt"))))

    combo = CombinationGrid().resolve(a_name, sim_presets.sbed_study_noises()[0][0], "Bayesian-SBED")
    assert combo.strategy["config"]["lineshape"] == "voigt"


def test_plain_lorentzian_generator_keeps_default_lineshape():
    """Plain `nvision run` generators (unrelated to the SBED study grid) must
    not be affected -- lineshape stays unset, defaulting to lorentzian."""
    combo = CombinationGrid().resolve("NVCenter-lorentzian", "Gauss(0.01)", "Bayesian-SBED")
    assert combo is not None
    assert "lineshape" not in combo.strategy["config"]


def test_bare_voigt_generator_keeps_default_lineshape():
    """The bare NVCenter-voigt generator (plain `nvision run`, no run-group) must not
    be affected -- lineshape stays unset (pre-existing gap, deliberately out of scope)."""
    combo = CombinationGrid().resolve("NVCenter-voigt", "Gauss(0.01)", "Bayesian-SBED")
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
