from __future__ import annotations

import math
import random

from nvision.signal.signal import TrueSignal
from nvision.sim.gen.core_generators import (
    MultiPeakCoreGenerator,
    NVCenterCoreGenerator,
    OnePeakCoreGenerator,
    SymmetricTwoPeakCoreGenerator,
    TwoPeakCoreGenerator,
)


def _peak_value(signal: TrueSignal, n: int = 2001) -> float:
    """Find the x-value of the maximum of the signal over its domain."""
    freq_param = next((p for p in signal.parameters if "frequency" in p.name and "peak1" not in p.name), None)
    if freq_param is None:
        freq_param = next((p for p in signal.parameters if "frequency" in p.name), None)
    if freq_param is None:
        return float("nan")
    x_min, x_max = freq_param.bounds
    step = (x_max - x_min) / (n - 1)
    best_x, best_y = x_min, -float("inf")
    for i in range(n):
        x = x_min + i * step
        y = signal(x)
        if y > best_y:
            best_y, best_x = y, x
    return best_x


def test_one_peak_gaussian_produces_true_signal():
    rng = random.Random(321)
    gen = OnePeakCoreGenerator(x_min=0.0, x_max=1.0, peak_type="gaussian")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    assert len(sig.parameters) == 4
    freq = next(p for p in sig.parameters if p.name == "frequency")
    assert math.isfinite(freq.value)
    assert 0.0 <= freq.value <= 1.0


def test_one_peak_lorentzian_produces_true_signal():
    rng = random.Random(42)
    gen = OnePeakCoreGenerator(x_min=2.6e9, x_max=3.1e9, peak_type="lorentzian")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    freq = next(p for p in sig.parameters if p.name == "frequency")
    assert 2.6e9 <= freq.value <= 3.1e9


def test_lorentzian_and_nv_have_nonflat_contrast_on_ghz_domain():
    """Lorentzian dip depth is amplitude/linewidth²; O(1) amplitudes look flat at GHz scale."""
    rng = random.Random(0)
    lorentz = OnePeakCoreGenerator(x_min=2.6e9, x_max=3.1e9, peak_type="lorentzian").generate(rng)
    nv = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian").generate(rng)
    x0, x1 = 2.6e9, 3.1e9
    grid = [x0 + (x1 - x0) * i / 500 for i in range(501)]
    for sig in (lorentz, nv):
        ys = [sig(x) for x in grid]
        assert max(ys) - min(ys) > 0.02


def test_two_peak_composite_model():
    rng = random.Random(7)
    gen = TwoPeakCoreGenerator(x_min=0.0, x_max=1.0)
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    freq_params = [p for p in sig.parameters if "frequency" in p.name]
    assert len(freq_params) == 2
    # Two peaks should be separated
    assert abs(freq_params[0].value - freq_params[1].value) > 0.01


def test_nv_center_lorentzian_has_six_parameters():
    rng = random.Random(11)
    gen = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    names = {p.name for p in sig.parameters}
    assert names == {"frequency", "linewidth", "split", "k_np", "amplitude", "background"}


def test_nv_center_voigt_has_different_params_than_lorentzian():
    rng_l = random.Random(22)
    rng_v = random.Random(22)
    gen_l = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="lorentzian")
    gen_v = NVCenterCoreGenerator(x_min=2.6e9, x_max=3.1e9, variant="voigt")
    sig_l = gen_l.generate(rng_l)
    sig_v = gen_v.generate(rng_v)
    assert isinstance(sig_v, TrueSignal)
    names_l = {p.name for p in sig_l.parameters}
    names_v = {p.name for p in sig_v.parameters}
    # Voigt model should have additional broadening parameters not in Lorentzian
    assert names_v != names_l, "Voigt and Lorentzian should have different parameter sets"


def test_multi_peak_generator():
    rng = random.Random(99)
    gen = MultiPeakCoreGenerator(x_min=0.0, x_max=1.0, count=3)
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    freq_params = [p for p in sig.parameters if "frequency" in p.name]
    assert len(freq_params) == 3


def test_symmetric_two_peak_generator():
    rng = random.Random(55)
    gen = SymmetricTwoPeakCoreGenerator(x_min=0.0, x_max=1.0, center=0.5, sep_frac=0.3)
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    freq_params = [p for p in sig.parameters if "frequency" in p.name]
    assert len(freq_params) == 2
    # Should be roughly symmetric around center
    positions = sorted(p.value for p in freq_params)
    mid = (positions[0] + positions[1]) / 2
    assert abs(mid - 0.5) < 0.05


def test_signal_is_callable():
    rng = random.Random(1)
    gen = OnePeakCoreGenerator(x_min=0.0, x_max=1.0)
    sig = gen.generate(rng)
    val = sig(0.5)
    assert math.isfinite(val)
