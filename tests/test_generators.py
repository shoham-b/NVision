from __future__ import annotations

import random

from nvision import (
    NVCenterCoreGenerator,
    TrueSignal,
)


def test_nv_center_lorentzian_default_has_zeeman_parameters():
    """Default generator uses Zeeman splitting (4 params)."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"frequency", "linewidth", "zeeman_split", "c_total"}


def test_nv_center_lorentzian_no_zeeman_has_three_parameters():
    """Explicit with_zeeman_splitting=False gives single-dip 3-param model."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian", with_zeeman_splitting=False)
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"frequency", "linewidth", "c_total"}


def test_nv_center_lorentzian_with_hyperfine_only_has_five_parameters():
    """Hyperfine-only (no Zeeman) gives 5-param model."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        x_min=2.6e9, x_max=3.1e9, variant="lorentzian",
        with_hyperfine_splitting=True, with_zeeman_splitting=False,
    )
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"frequency", "linewidth", "split", "k_np", "c_total"}


def test_nv_center_lorentzian_with_zeeman_and_hyperfine_has_six_parameters():
    """Zeeman + hyperfine gives 6-param model."""
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(
        x_min=2.6e9, x_max=3.1e9, variant="lorentzian",
        with_hyperfine_splitting=True, with_zeeman_splitting=True,
    )
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"frequency", "linewidth", "zeeman_split", "split", "k_np", "c_total"}


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
