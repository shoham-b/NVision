from __future__ import annotations

import random

from nvision import (
    NVCenterCoreGenerator,
    TrueSignal,
)


def test_nv_center_lorentzian_default_has_zeeman_parameters():
    """Default generator uses Zeeman splitting (3 free params; frequency is fixed)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"linewidth", "zeeman_split", "c_total"}


def test_nv_center_lorentzian_no_zeeman_has_three_parameters():
    """Explicit with_zeeman_splitting=False gives single-dip 2-param model (frequency fixed)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian", with_zeeman_splitting=False)
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"linewidth", "c_total"}


def test_nv_center_lorentzian_with_hyperfine_only_has_five_parameters():
    """Hyperfine-only (no Zeeman) gives 4-param model (frequency fixed)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        x_min=2.6e9, x_max=3.1e9, variant="lorentzian",
        with_hyperfine_splitting=True, with_zeeman_splitting=False,
    )
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"linewidth", "split", "k_np", "c_total"}


def test_nv_center_lorentzian_with_zeeman_and_hyperfine_has_six_parameters():
    """Zeeman + hyperfine gives 5-param model (frequency fixed)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        x_min=2.6e9, x_max=3.1e9, variant="lorentzian",
        with_hyperfine_splitting=True, with_zeeman_splitting=True,
    )
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"linewidth", "zeeman_split", "split", "k_np", "c_total"}


def test_nv_center_lorentzian_fixed_linewidth_and_contrast():
    """Fixed linewidth/c_total are used verbatim instead of randomized."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian", linewidth=1e6, c_total=0.1)
    sig = gen.generate(rng)
    assert sig.get_param_value("linewidth") == 1e6
    assert sig.get_param_value("c_total") == 0.1


def test_nv_center_lorentzian_unset_linewidth_and_contrast_still_randomized():
    """None (default) keeps the historical randomized behavior."""
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    sig_a = gen.generate(random.Random(1))
    sig_b = gen.generate(random.Random(2))
    assert sig_a.get_param_value("linewidth") != sig_b.get_param_value("linewidth")
    assert sig_a.get_param_value("c_total") != sig_b.get_param_value("c_total")


def test_nv_center_saturation_voigt_default_has_zeeman_parameters():
    """Default generator uses Zeeman splitting: saturation, sigma_inhom, zeeman_split.

    c_max is not a parameter -- it's a fixed constant (NV_SATURATION_C_MAX), not
    inferred/drawn per repeat. frequency is also fixed (with_fixed_frequency=True
    default), so it's not among the free/inferred parameters either.
    """
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="saturation_voigt")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"saturation", "sigma_inhom", "zeeman_split"}


def test_nv_center_saturation_voigt_fixed_values_used_verbatim():
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        x_min=2.6e9, x_max=3.1e9, variant="saturation_voigt", saturation=10.0, sigma_inhom=5e5
    )
    sig = gen.generate(rng)
    assert sig.get_param_value("saturation") == 10.0
    assert sig.get_param_value("sigma_inhom") == 5e5


def test_nv_center_saturation_voigt_unset_values_still_randomized():
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="saturation_voigt")
    sig_a = gen.generate(random.Random(1))
    sig_b = gen.generate(random.Random(2))
    assert sig_a.get_param_value("saturation") != sig_b.get_param_value("saturation")
    assert sig_a.get_param_value("sigma_inhom") != sig_b.get_param_value("sigma_inhom")


def _dip_cluster_extent(variant: str, params: dict, frequency: float) -> tuple[float, float]:
    """Worst-case (outer_lo, outer_hi) of the full dip cluster for one drawn signal."""
    f = frequency
    zeeman = params.get("zeeman_split", 0.0)
    split = params.get("split", 0.0)
    if variant == "saturation_voigt":
        from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar

        fwhm, _, _ = _saturation_voigt_reparam_scalar(
            params["saturation"], params["sigma_inhom"], NV_SATURATION_C_MAX
        )
    elif variant == "voigt":
        from nvision.spectra.nv_center import _voigt_reparam_scalar

        fwhm, _ = _voigt_reparam_scalar(params.get("homogeneous_linewidth", 0.0), params.get("sigma_inhom", 0.0))
    else:
        fwhm = params.get("linewidth", 0.0) * 2.0
    return f - zeeman - split - fwhm / 2.0, f + zeeman + split + fwhm / 2.0


def test_dip_cluster_stays_within_domain_for_all_variants():
    """Regression: the FULL dip cluster (not just center_freq) must stay inside
    [x_min, x_max] across many draws, for every variant/splitting combination."""
    x_min, x_max = 2.6e9, 3.1e9
    configs = [
        ("lorentzian", {}),
        ("lorentzian", {"with_zeeman_splitting": False}),
        ("lorentzian", {"with_hyperfine_splitting": True, "with_zeeman_splitting": True}),
        ("voigt", {}),
        ("saturation_voigt", {}),
    ]
    for variant, kwargs in configs:
        gen = NVCenterCoreGenerator(x_min=x_min, x_max=x_max, variant=variant, **kwargs)
        for seed in range(100):
            sig = gen.generate(random.Random(seed))
            frequency = sig.get_param_value("frequency")
            outer_lo, outer_hi = _dip_cluster_extent(variant, sig.parameter_values(), frequency)
            assert outer_lo >= x_min, (variant, kwargs, seed, outer_lo)
            assert outer_hi <= x_max, (variant, kwargs, seed, outer_hi)


def test_nv_center_voigt_has_different_params_than_lorentzian():
    rng_l = random.Random(22)
    rng_v = random.Random(22)
    gen_l = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    gen_v = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="voigt")
    sig_l = gen_l.generate(rng_l)
    sig_v = gen_v.generate(rng_v)
    assert isinstance(sig_v, TrueSignal)
    names_l = set(sig_l.parameter_names)
    names_v = set(sig_v.parameter_names)
    # Voigt model should have additional broadening parameters not in Lorentzian
    assert names_v != names_l, "Voigt and Lorentzian should have different parameter sets"


def test_nv_center_voigt_fixed_contrast_flows_through():
    """c_total is now voigt's direct, population-normalized amplitude parameter
    (no more dip_depth/g_max renormalization) -- an override must land exactly."""
    gen_a = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="voigt", c_total=0.2)
    gen_b = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="voigt", c_total=0.8)
    sig_a = gen_a.generate(random.Random(7))
    sig_b = gen_b.generate(random.Random(7))
    assert sig_a.get_param_value("c_total") == 0.2
    assert sig_b.get_param_value("c_total") == 0.8


def test_nv_center_voigt_unset_contrast_still_randomized():
    """None (default) keeps the historical randomized behavior."""
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="voigt")
    sig_a = gen.generate(random.Random(1))
    sig_b = gen.generate(random.Random(2))
    assert sig_a.get_param_value("c_total") != sig_b.get_param_value("c_total")
