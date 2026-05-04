from __future__ import annotations

import random

from nvision import (
    NVCenterCoreGenerator,
    TrueSignal,
)


def test_nv_center_lorentzian_has_six_parameters():
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = set(sig.parameter_names)
    assert names == {"frequency", "linewidth", "split", "k_np", "dip_depth", "background"}


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
