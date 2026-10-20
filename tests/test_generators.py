from __future__ import annotations

import random

from nvision import (
    NVCenterCoreGenerator,
    TrueSignal,
)


def test_nv_center_lorentzian_default_has_zeeman_parameters():
    """Default generator uses Zeeman splitting (3 free params; zero_field_freq is fixed
    -- see NVCenterCoreGenerator's docstring -- so it's not in parameter_names)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"hwhm", "zeeman_shift", "contrast_total"}


def test_nv_center_lorentzian_no_zeeman_has_three_parameters():
    """Explicit with_zeeman_splitting=False gives single-dip 2-free-param model
    (zero_field_freq is fixed, not in parameter_names)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian", with_zeeman_splitting=False
    )
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"hwhm", "contrast_total"}


def test_nv_center_lorentzian_with_hyperfine_only_has_five_parameters():
    """Hyperfine-only (no Zeeman) gives 4-free-param model (zero_field_freq is fixed,
    not in parameter_names)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9,
        drive_freq_max_phys=3.1e9,
        variant="lorentzian",
        hyperfine="n14",
        infer_hyperfine=True,
        with_zeeman_splitting=False,
    )
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"hwhm", "hyperfine_shift", "k_np", "contrast_total"}


def test_nv_center_lorentzian_with_zeeman_and_hyperfine_has_six_parameters():
    """Zeeman + hyperfine gives 5-free-param model (zero_field_freq is fixed, not in
    parameter_names)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9,
        drive_freq_max_phys=3.1e9,
        variant="lorentzian",
        hyperfine="n14",
        infer_hyperfine=True,
        with_zeeman_splitting=True,
    )
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"hwhm", "zeeman_shift", "hyperfine_shift", "k_np", "contrast_total"}


def test_nv_center_lorentzian_fixed_hwhm_and_contrast():
    """Fixed hwhm/contrast_total are used verbatim instead of randomized."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian", hwhm=1e6, contrast_total=0.1
    )
    sig = gen.generate(rng)
    assert sig.get_param_value("hwhm") == 1e6
    assert sig.get_param_value("contrast_total") == 0.1


def test_nv_center_lorentzian_unset_hwhm_and_contrast_still_randomized():
    """None (default) keeps the historical randomized behavior."""
    gen = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian")
    sig_a = gen.generate(random.Random(1))
    sig_b = gen.generate(random.Random(2))
    assert sig_a.get_param_value("hwhm") != sig_b.get_param_value("hwhm")
    assert sig_a.get_param_value("contrast_total") != sig_b.get_param_value("contrast_total")


def test_nv_center_saturation_voigt_default_has_zeeman_parameters():
    """Default generator uses Zeeman splitting: saturation, sigma_inhom, zeeman_shift.

    c_max is not a parameter -- it's a fixed constant (NV_SATURATION_C_MAX), not
    inferred/drawn per repeat. zero_field_freq is also fixed (not inferred, per
    NVCenterCoreGenerator's docstring) and so isn't in parameter_names either.
    """
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="saturation_voigt")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"saturation", "sigma_inhom", "zeeman_shift"}


def test_nv_center_saturation_voigt_fixed_values_used_verbatim():
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9,
        drive_freq_max_phys=3.1e9,
        variant="saturation_voigt",
        saturation=10.0,
        sigma_inhom=5e5,
    )
    sig = gen.generate(rng)
    assert sig.get_param_value("saturation") == 10.0
    assert sig.get_param_value("sigma_inhom") == 5e5


def test_nv_center_saturation_voigt_unset_values_still_randomized():
    gen = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="saturation_voigt")
    sig_a = gen.generate(random.Random(1))
    sig_b = gen.generate(random.Random(2))
    assert sig_a.get_param_value("saturation") != sig_b.get_param_value("saturation")
    assert sig_a.get_param_value("sigma_inhom") != sig_b.get_param_value("sigma_inhom")


def _dip_cluster_extent(variant: str, params: dict) -> tuple[float, float]:
    """Worst-case (outer_lo, outer_hi) of the full dip cluster for one drawn signal."""
    f = params["zero_field_freq"]
    zeeman = params.get("zeeman_shift", 0.0)
    hyperfine_shift = params.get("hyperfine_shift", 0.0)
    if variant == "saturation_voigt":
        from nvision.spectra.nv_center import NV_SATURATION_C_MAX, _saturation_voigt_reparam_scalar

        fwhm, _, _ = _saturation_voigt_reparam_scalar(params["saturation"], params["sigma_inhom"], NV_SATURATION_C_MAX)
    elif variant == "voigt":
        from nvision.spectra.nv_center import _voigt_reparam_scalar

        fwhm, _ = _voigt_reparam_scalar(params.get("hwhm_lorentz", 0.0), params.get("sigma_inhom", 0.0))
    else:
        fwhm = params.get("hwhm", 0.0) * 2.0
    return f - zeeman - hyperfine_shift - fwhm / 2.0, f + zeeman + hyperfine_shift + fwhm / 2.0


def test_dip_cluster_stays_within_domain_for_all_variants():
    """Regression: the FULL dip cluster (not just zero_field_freq) must stay inside
    [drive_freq_min_phys, drive_freq_max_phys] across many draws, for every variant/splitting combination."""
    drive_freq_min_phys, drive_freq_max_phys = 2.6e9, 3.1e9
    configs = [
        ("lorentzian", {}),
        ("lorentzian", {"with_zeeman_splitting": False}),
        ("lorentzian", {"hyperfine": "n14", "infer_hyperfine": True, "with_zeeman_splitting": True}),
        ("voigt", {}),
        ("saturation_voigt", {}),
    ]
    for variant, kwargs in configs:
        gen = NVCenterCoreGenerator(
            drive_freq_min_phys=drive_freq_min_phys, drive_freq_max_phys=drive_freq_max_phys, variant=variant, **kwargs
        )
        for seed in range(100):
            sig = gen.generate(random.Random(seed))
            # zero_field_freq is fixed (not inferred) by generator design, so it's absent
            # from parameter_values() -- fetch it via get_param_value's fixed-value
            # fallback instead, and merge it into the params dict _dip_cluster_extent expects.
            params = {**sig.parameter_values(), "zero_field_freq": sig.get_param_value("zero_field_freq")}
            outer_lo, outer_hi = _dip_cluster_extent(variant, params)
            assert outer_lo >= drive_freq_min_phys, (variant, kwargs, seed, outer_lo)
            assert outer_hi <= drive_freq_max_phys, (variant, kwargs, seed, outer_hi)


def test_nv_center_voigt_has_different_params_than_lorentzian():
    rng_l = random.Random(22)
    rng_v = random.Random(22)
    gen_l = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="lorentzian")
    gen_v = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="voigt")
    sig_l = gen_l.generate(rng_l)
    sig_v = gen_v.generate(rng_v)
    assert isinstance(sig_v, TrueSignal)
    names_l = set(sig_l.parameter_names)
    names_v = set(sig_v.parameter_names)
    # Voigt model should have additional broadening parameters not in Lorentzian
    assert names_v != names_l, "Voigt and Lorentzian should have different parameter sets"


def test_nv_center_voigt_fixed_contrast_flows_through():
    """contrast_total is now voigt's direct, population-normalized amplitude parameter
    (no more contrast/g_max renormalization) -- an override must land exactly."""
    gen_a = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="voigt", contrast_total=0.2
    )
    gen_b = NVCenterCoreGenerator(
        drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="voigt", contrast_total=0.8
    )
    sig_a = gen_a.generate(random.Random(7))
    sig_b = gen_b.generate(random.Random(7))
    assert sig_a.get_param_value("contrast_total") == 0.2
    assert sig_b.get_param_value("contrast_total") == 0.8


def test_nv_center_voigt_unset_contrast_still_randomized():
    """None (default) keeps the historical randomized behavior."""
    gen = NVCenterCoreGenerator(drive_freq_min_phys=2.6e9, drive_freq_max_phys=3.1e9, variant="voigt")
    sig_a = gen.generate(random.Random(1))
    sig_b = gen.generate(random.Random(2))
    assert sig_a.get_param_value("contrast_total") != sig_b.get_param_value("contrast_total")
