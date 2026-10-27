"""Run-group registry and the study parameter grids they (and historical results) use."""

from __future__ import annotations

import numpy as np
import pytest

import nvision.sim.defaults as sim_defaults
from nvision.sim import presets as sim_presets
from nvision.sim import run_groups as sim_run_groups
from nvision.sim.combinations import CombinationGrid
from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator
from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar


def test_voigt_grid_group_uses_voigt_inhom_grid_and_all_locators():
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("voigt-grid")
    assert group.generator_names
    assert len(set(group.generator_names)) == len(group.generator_names)
    assert group.extra_generators is not None
    assert set(group.extra_generators.keys()) == set(group.generator_names)
    assert set(group.strategy_names) == {
        "SimpleSweep-2perFWHM",
        "SimpleSweep-3perFWHM",
        "SimpleSweep-4perFWHM",
        "Smart-Sweep",
        "Bayesian-SBED",
        "Smart-SBED",
    }
    assert group.task_count == len(group.generator_names) * len(group.noise_names) * len(group.strategy_names)


def test_former_group_names_are_strategy_subsets_of_their_scenario_group():
    sim_run_groups.clear_run_group_cache()
    listed = {g.name for g in sim_run_groups.run_groups()}
    assert listed == {"voigt-grid", "drive-control", "all"}
    expected = {
        "both-sbed": ("voigt-grid", ["Bayesian-SBED", "SimpleSweep-3perFWHM"]),
        "both-sbed-only": ("voigt-grid", ["Bayesian-SBED"]),
        "both-sweep-only": ("voigt-grid", ["SimpleSweep-3perFWHM"]),
    }
    assert set(sim_run_groups.alias_names()) == set(expected)
    for alias, (canonical, strategies) in expected.items():
        group, base = sim_run_groups.get_run_group(alias), sim_run_groups.get_run_group(canonical)
        assert group.name == alias
        assert group.generator_names == base.generator_names
        assert group.noise_names == base.noise_names
        assert group.strategy_names == strategies


def test_all_group_is_the_union_of_every_scenario_with_its_own_locators():
    sim_run_groups.clear_run_group_cache()
    scenarios = [sim_run_groups.get_run_group(n) for n in ("voigt-grid", "drive-control")]
    everything = sim_run_groups.get_run_group("all")
    assert everything.task_count == sum(g.task_count for g in scenarios)
    combos = set(everything.combinations())
    assert len(combos) == everything.task_count
    # Drive-choosing SBED only runs on the drive-controlled generators, not on every generator.
    drive_gens = set(scenarios[1].generator_names)
    assert {g for g, _, s in combos if s == "Bayesian-SBED-Drive"} == drive_gens
    # Running only the sweeps across every scenario is a plain --filter-strategy on the composite group.
    sweeps = [c for c in combos if "Sweep" in c[2]]
    assert sweeps
    assert {c[0] for c in sweeps} == {g for sc in scenarios for g in sc.generator_names} - drive_gens
    assert set(everything.extra_generators) == {g for sc in scenarios for g in sc.generator_names}


def test_default_run_group_is_voigt_grid():
    assert sim_run_groups.default_run_group().name == "voigt-grid"


def test_voigt_grid_group_noise_axis_is_snr(monkeypatch):
    """The voigt-grid noise axis is posed in SNR = contrast / sigma, independent of the generic noise globals."""
    monkeypatch.setattr(sim_defaults, "NVISION_NOISE_MAX_GAUSS", 999.0)
    monkeypatch.setattr(sim_defaults, "NVISION_NOISE_GAUSS_STEPS", 999)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_SNR_CONTRAST", 0.25)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_SNR_VALUES", (5.0, 10.0, 50.0))

    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("voigt-grid")

    assert group.noise_names == ["SNR(5)", "SNR(10)", "SNR(50)"]
    sim_run_groups.clear_run_group_cache()


def test_snr_noise_names_are_the_grid_snr_and_sigma_is_derived_per_signal():
    """A run is named by its SNR; the sigma is the signal's contrast / SNR, not part of the name."""
    from nvision.sim.combinations import noise_sigma_for, parse_snr

    snrs = sim_defaults.NVISION_SBED_SNR_VALUES
    names = [name for name, _ in sim_presets.snr_study_noises()]
    assert [parse_snr(n) for n in names] == list(snrs)
    gen = next(g for _, g in sim_presets.voigt_snr_window_param_grid_generators())
    assert [noise_sigma_for(gen, n) for n in names] == pytest.approx([gen.contrast_total / s for s in snrs])


def test_snr_noise_resolves_to_the_sigma_of_each_signal(monkeypatch):
    """The same name is the same SNR for a signal with another contrast, and keeps it if the constant moves."""
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

    grid = CombinationGrid(
        extra_generators={
            "NVCenter-voigt-occ0.1-test-c0.2": NVCenterCoreGenerator(variant="voigt", contrast_total=0.2),
            "NVCenter-voigt-occ0.1-test-free": NVCenterCoreGenerator(variant="voigt"),
        }
    )

    def sigma(gen: str) -> float:
        return grid.resolve(gen, "SNR(20)", "Bayesian-SBED").noise.over_frequency_noise._parts[0].std

    assert sigma("NVCenter-voigt-occ0.1-test-c0.2") == pytest.approx(0.2 / 20)
    # A signal with no fixed contrast: the SNR is nominal, against the study contrast.
    assert sigma("NVCenter-voigt-occ0.1-test-free") == pytest.approx(sim_defaults.NVISION_SBED_SNR_CONTRAST / 20)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_SNR_CONTRAST", 0.5)
    assert sigma("NVCenter-voigt-occ0.1-test-free") == pytest.approx(0.5 / 20)
    assert grid.resolve("NVCenter-voigt-occ0.1-test-c0.2", "SNR(20)+Drift(no-such)", "Bayesian-SBED") is None


def test_snr_noises_reject_non_positive_snr(monkeypatch):
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_SNR_VALUES", (10.0, 0.0))
    with pytest.raises(ValueError, match="SNR"):
        sim_presets.snr_study_noises()


def test_snr_window_grid_has_one_total_fwhm_and_varies_one_thing_per_axis():
    """Window = N x ONE total FWHM; Gaussian share and isotope vary at that same FWHM."""
    from nvision.spectra.nv_center import _voigt_reparam_scalar, max_zeeman_shift_for_window

    generators = dict(sim_presets.voigt_snr_window_param_grid_generators())
    assert len(generators) == 36  # 4 occupancies x 3 Gaussian shares x 3 isotopes
    assert {g.hyperfine for g in generators.values()} == set(sim_defaults.NVISION_SBED_HYPERFINE_VALUES)
    for gen in generators.values():
        assert gen.contrast_total == sim_defaults.NVISION_SBED_SNR_CONTRAST
        assert gen.hyperfine in sim_defaults.NVISION_SBED_HYPERFINE_VALUES
        fwhm_total, lorentz_frac = _voigt_reparam_scalar(gen.hwhm, gen.sigma_inhom)
        assert fwhm_total == pytest.approx(sim_defaults.NVISION_SBED_FWHM)
        assert gen.gaussian_share == pytest.approx(1.0 - lorentz_frac)
        window = gen.drive_freq_max_phys - gen.drive_freq_min_phys
        assert window == pytest.approx(gen.window_fwhm * sim_defaults.NVISION_SBED_FWHM)
        assert gen.zeeman_max_phys == pytest.approx(max_zeeman_shift_for_window(window))
    # The same window is the same absolute width for every line shape, and a share is the same shape in every window.
    windows = {}
    for gen in generators.values():
        windows.setdefault(gen.window_fwhm, set()).add(round(gen.drive_freq_max_phys - gen.drive_freq_min_phys))
    assert all(len(v) == 1 for v in windows.values())
    shares = {round(g.gaussian_share, 9) for g in generators.values()}
    assert shares == {round(x, 9) for x in sim_defaults.NVISION_SBED_GAUSSIAN_SHARES}


def test_snr_window_grid_rejects_a_share_that_leaves_no_lorentzian(monkeypatch):
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_GAUSSIAN_SHARES", (0.95,))
    with pytest.raises(ValueError, match="Lorentzian HWHM"):
        sim_presets.voigt_snr_window_param_grid_generators()
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_GAUSSIAN_SHARES", (1.0,))
    with pytest.raises(ValueError, match="Gaussian share"):
        sim_presets.voigt_snr_window_param_grid_generators()


def test_gaussian_share_is_only_defined_for_the_snr_window_study():
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

    assert NVCenterCoreGenerator(variant="voigt", hwhm=1e6, sigma_inhom=5e5).gaussian_share is None
    assert next(iter(sim_presets.voigt_snr_window_param_grid_generators()))[1].gaussian_share is not None


def test_snr_window_generator_draws_dips_inside_its_window():
    import random

    gen = next(g for n, g in sim_presets.voigt_snr_window_param_grid_generators() if "-occ0.05-" in n)
    for seed in range(20):
        signal = gen.generate(random.Random(seed))
        values = signal.parameter_values()
        window = gen.drive_freq_max_phys - gen.drive_freq_min_phys
        assert 0.0 <= values["zeeman_shift"] <= gen.zeeman_max_phys < window
        assert signal.bounds["zeeman_shift"] == (0.0, gen.zeeman_max_phys)
        assert signal.bounds["zero_field_freq"] == (gen.drive_freq_min_phys, gen.drive_freq_max_phys)


def test_default_window_zeeman_range_is_unchanged():
    """Generators without window_fwhm keep MAX_ZEEMAN_SHIFT, which is exactly the default window's fraction."""
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator
    from nvision.spectra.nv_center import (
        DEFAULT_NV_DRIVE_FREQ_MAX_PHYS,
        DEFAULT_NV_DRIVE_FREQ_MIN_PHYS,
        MAX_ZEEMAN_SHIFT,
        max_zeeman_shift_for_window,
    )

    assert NVCenterCoreGenerator(variant="voigt").zeeman_max_phys == MAX_ZEEMAN_SHIFT
    default_window = DEFAULT_NV_DRIVE_FREQ_MAX_PHYS - DEFAULT_NV_DRIVE_FREQ_MIN_PHYS
    assert max_zeeman_shift_for_window(default_window) == pytest.approx(MAX_ZEEMAN_SHIFT)


def test_window_fwhm_requires_voigt_variant():
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator

    with pytest.raises(ValueError, match="window_fwhm"):
        _ = NVCenterCoreGenerator(variant="lorentzian", window_fwhm=40.0).zeeman_max_phys


def test_snr_window_strategy_config_matches_the_signal_isotope():
    for name, gen in sim_presets.voigt_snr_window_param_grid_generators():
        combo = CombinationGrid().resolve(name, sim_presets.snr_study_noises()[0][0], "Bayesian-SBED")
        assert combo is not None
        assert combo.strategy["config"]["lineshape"] == "voigt"
        assert combo.strategy["config"]["hyperfine"] == gen.hyperfine
        assert not combo.strategy["config"].get("infer_hyperfine", False)


def test_snr_window_belief_inherits_the_generator_window():
    import random

    from nvision.sim.locs.bayesian.belief_builders import nv_center_smc_belief

    name = next(n for n, _ in sim_presets.voigt_snr_window_param_grid_generators() if "-occ0.05-" in n)
    combo = CombinationGrid().resolve(name, sim_presets.snr_study_noises()[0][0], "Bayesian-SBED")
    signal = combo.generator.generate(random.Random(0))
    cfg = {k: v for k, v in combo.strategy["config"].items() if k not in ("builder", "max_steps")}
    belief = nv_center_smc_belief(signal.bounds, noise_model=combo.noise.to_noise_signal_model(), **cfg)
    lo, hi = belief.drive_freq_bounds_phys
    assert (lo, hi) == (combo.generator.drive_freq_min_phys, combo.generator.drive_freq_max_phys)
    assert belief.physical_param_bounds["zeeman_shift"][1] == pytest.approx(combo.generator.zeeman_max_phys)


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
    a_noise = sim_presets.snr_study_noises()[0][0]

    grid = CombinationGrid()  # no extra_generators passed
    combo = grid.resolve(a_name, a_noise, "Bayesian-SBED")
    assert combo is not None
    assert combo.generator_name == a_name


def test_bayesian_strategy_config_uses_saturation_voigt_lineshape_for_study_grid():
    """The belief must be built with the matching lineshape, or inference models
    the wrong signal shape entirely."""
    a_name = next(iter(dict(sim_presets.saturation_voigt_param_grid_generators())))

    combo = CombinationGrid().resolve(a_name, sim_presets.snr_study_noises()[0][0], "Bayesian-SBED")
    assert combo.strategy["config"]["lineshape"] == "saturation_voigt"


def test_bayesian_strategy_config_uses_voigt_lineshape_for_width_contrast_grid():
    a_name = next(iter(dict(sim_presets.param_grid_generators(variant="voigt"))))

    combo = CombinationGrid().resolve(a_name, sim_presets.snr_study_noises()[0][0], "Bayesian-SBED")
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


def test_snr_window_width_bounds_and_priors_do_not_move_with_the_window():
    """Only positions follow the window: the linewidth bounds and its prior std are identical for every N."""
    import random

    by_share_hf: dict[tuple[float, str], list[tuple[dict, dict]]] = {}
    for _, gen in sim_presets.voigt_snr_window_param_grid_generators():
        signal = gen.generate(random.Random(0))
        priors = signal.bounds["_priors"]
        by_share_hf.setdefault((gen.gaussian_share, gen.hyperfine), []).append(
            (
                {n: signal.bounds[n] for n in ("hwhm_lorentz", "sigma_inhom", "contrast_total")},
                {n: priors[n][1] for n in ("hwhm_lorentz", "sigma_inhom", "contrast_total")},
            )
        )
    assert by_share_hf
    for runs in by_share_hf.values():
        assert all(r == runs[0] for r in runs)


def test_select_snr_window_narrows_a_group_by_snr_and_window():
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("voigt-grid")
    combos = group.combinations()
    gens = group.extra_generators

    at_window = sim_run_groups.select_snr_window(combos, gens, signal_occupancy=[0.1])
    assert at_window
    assert {round(gens[g].signal_occupancy, 9) for g, _, _ in at_window} == {0.1}
    assert {n for _, n, _ in at_window} == set(group.noise_names)

    at_snr = sim_run_groups.select_snr_window(combos, gens, snr=[10.0])
    assert [sim_run_groups.snr_of_noise(n) for _, n, _ in at_snr] == pytest.approx([10.0] * len(at_snr))
    assert {g for g, _, _ in at_snr} == set(group.generator_names)

    both = sim_run_groups.select_snr_window(combos, gens, snr=[2.0, 10.0], signal_occupancy=[0.2, 0.05])
    # 18 generators have signal_occupancy in [0.2, 0.05] (3 shares x 3 isotopes at each),
    # across 2 SNRs and all locators
    assert len(both) == 18 * 2 * len(group.strategy_names)
    assert sim_run_groups.select_snr_window(combos, gens, snr=[3.3]) == []
    assert sim_run_groups.select_snr_window(combos, gens) == combos


def test_select_snr_window_matches_nothing_for_signals_without_a_stated_window():
    sim_run_groups.clear_run_group_cache()
    group = sim_run_groups.get_run_group("drive-control")
    combos = group.combinations()
    assert sim_run_groups.select_snr_window(combos, group.extra_generators, signal_occupancy=[0.1]) == []
    assert sim_run_groups.select_snr_window(combos, group.extra_generators, snr=[20.0])


def test_drive_control_noise_axis_is_the_snr_grid():
    sim_run_groups.clear_run_group_cache()
    names = [n for n, _ in sim_presets.snr_study_noises()]
    assert sim_run_groups.get_run_group("drive-control").noise_names == names


def test_measurement_budget_is_the_same_for_every_window():
    from types import SimpleNamespace

    from nvision.runner.executor import _TaskRunner
    from nvision.sim.gen.nv_center_generator import NVCenterCoreGenerator
    from nvision.sim.locs.bayesian.sbed_locator import SequentialBayesianExperimentDesignLocator

    def budget(generator) -> int:
        runner = SimpleNamespace(
            task=SimpleNamespace(
                generator=generator,
                strategy_spec=SimpleNamespace(locator_class=SequentialBayesianExperimentDesignLocator),
            )
        )
        experiment = SimpleNamespace(drive_freq_min_phys=2.8e9, drive_freq_max_phys=2.9e9)
        return _TaskRunner._resolve_sweep_max_steps(runner, experiment)

    by_window = {g.window_fwhm: budget(g) for _, g in sim_presets.voigt_snr_window_param_grid_generators()}
    assert len(by_window) > 1
    assert set(by_window.values()) == {sim_defaults.NVISION_MAX_MEASUREMENTS}
    assert budget(NVCenterCoreGenerator()) == sim_defaults.NVISION_MAX_MEASUREMENTS


def test_drift_study_noise_is_posed_as_an_snr(monkeypatch):
    monkeypatch.setattr(sim_defaults, "NVISION_DRIFT_SNR", 50.0)
    monkeypatch.setattr(sim_defaults, "NVISION_SBED_SNR_CONTRAST", 0.25)
    names = [n for n, _ in sim_presets.drift_study_noises()]
    assert names[0] == "SNR(50)"
    assert all(n.startswith("SNR(50)") for n in names)
    assert names[1:] == [f"SNR(50)+Drift({spec.label})" for spec in sim_presets.drift_scenarios()]


def test_signal_occupancy_is_the_union_of_the_hyperfine_lines_fwhm_intervals():
    """Occupancy = covered frequency / window: n lines of one FWHM each, minus where neighbouring lines overlap."""
    from nvision.spectra.nv_center import hyperfine_geometry

    fwhm = sim_defaults.NVISION_SBED_FWHM
    by_name = dict(sim_presets.voigt_snr_window_param_grid_generators())
    for gen in by_name.values():
        window = gen.drive_freq_max_phys - gen.drive_freq_min_phys
        offset, _, n_lines = hyperfine_geometry(gen.hyperfine)
        # adjacent lines are `spacing` apart: N-14 offset, N-15 twice its (half-coupling) offset
        spacing = {1: fwhm, 2: 2.0 * offset, 3: offset}[n_lines]
        covered = fwhm + (n_lines - 1) * min(fwhm, spacing)
        assert gen.signal_occupancy == pytest.approx(covered / window)
        assert 0.0 < gen.signal_occupancy <= 1.0
    # The grid is posed in occupancy, so every isotope hits the grid value exactly; the window is what differs.
    grid_values = {round(g.signal_occupancy, 9) for g in by_name.values()}
    assert grid_values == {round(v, 9) for v in sim_defaults.NVISION_SBED_SIGNAL_OCCUPANCY_VALUES}
    at_occ = {
        g.hyperfine: g.drive_freq_max_phys - g.drive_freq_min_phys for n, g in by_name.items() if "-occ0.1-gs0-" in n
    }
    assert at_occ["unresolved"] == pytest.approx(10 * fwhm)
    assert at_occ["n15"] == pytest.approx(20 * fwhm)  # lines 3.03 MHz apart > FWHM: no overlap, 2 FWHM covered
    assert at_occ["n14"] == pytest.approx(30 * fwhm)  # lines 2.16 MHz apart > FWHM: no overlap, 3 FWHM covered
    assert NVCenterCoreGenerator(variant="voigt").signal_occupancy is None


def test_signal_occupancy_counts_overlapping_lines_once(monkeypatch):
    """Lines closer than a FWHM share their overlap: three lines A apart cover 3 FWHM - 2 (FWHM - A)."""
    import nvision.spectra.nv_center as nv

    fwhm = sim_defaults.NVISION_SBED_FWHM
    a = 0.75 * fwhm
    monkeypatch.setattr(nv, "hyperfine_geometry", lambda hyperfine: (a, 1.0, 3))
    assert nv.hyperfine_covered_width(fwhm, "n14") == pytest.approx(3 * fwhm - 2 * (fwhm - a))
    monkeypatch.setattr(nv, "hyperfine_geometry", lambda hyperfine: (a / 2.0, 0.0, 2))
    assert nv.hyperfine_covered_width(fwhm, "n15") == pytest.approx(2 * fwhm - (fwhm - a))
