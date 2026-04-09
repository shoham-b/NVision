from __future__ import annotations

import math
import random

from nvision import (
    MultiPeakCoreGenerator,
    NVCenterCoreGenerator,
    OnePeakCoreGenerator,
    SymmetricTwoPeakCoreGenerator,
    TwoPeakCoreGenerator,
)
from nvision import TrueSignal


def _peak_value(signal: TrueSignal, n: int = 2001) -> float:
    """Find the x-value of the maximum of the signal over its domain."""
    # Find frequency parameter name (prefer non-peak1 frequency for multi-peak)
    freq_names = [name for name in signal.parameter_names if "frequency" in name]
    freq_name = next((name for name in freq_names if "peak1" not in name), None)
    if freq_name is None and freq_names:
        freq_name = freq_names[0]
    if freq_name is None:
        return float("nan")
    x_min, x_max = signal.get_param_bounds(freq_name)
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
    assert len(sig.parameter_names) == 4
    freq_value = sig.get_param_value("frequency")
    assert math.isfinite(freq_value)
    assert 0.0 <= freq_value <= 1.0


def test_one_peak_lorentzian_produces_true_signal():
    rng = random.Random(42)
    gen = OnePeakCoreGenerator(x_min=2.6e9, x_max=3.1e9, peak_type="lorentzian")
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    freq_value = sig.get_param_value("frequency")
    assert 2.6e9 <= freq_value <= 3.1e9


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
    freq_names = [name for name in sig.parameter_names if "frequency" in name]
    assert len(freq_names) == 2
    # Two peaks should be separated
    freq_values = [sig.get_param_value(name) for name in freq_names]
    assert abs(freq_values[0] - freq_values[1]) > 0.01


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


def test_multi_peak_generator():
    rng = random.Random(99)
    gen = MultiPeakCoreGenerator(x_min=0.0, x_max=1.0, count=3)
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    freq_names = [name for name in sig.parameter_names if "frequency" in name]
    assert len(freq_names) == 3


def test_symmetric_two_peak_generator():
    rng = random.Random(55)
    gen = SymmetricTwoPeakCoreGenerator(x_min=0.0, x_max=1.0, center=0.5, sep_frac=0.3)
    sig = gen.generate(rng)
    assert isinstance(sig, TrueSignal)
    freq_names = [name for name in sig.parameter_names if "frequency" in name]
    assert len(freq_names) == 2
    # Should be roughly symmetric around center
    positions = sorted(sig.get_param_value(name) for name in freq_names)
    mid = (positions[0] + positions[1]) / 2
    assert abs(mid - 0.5) < 0.05


def test_signal_is_callable():
    rng = random.Random(1)
    gen = OnePeakCoreGenerator(x_min=0.0, x_max=1.0)
    sig = gen.generate(rng)
    val = sig(0.5)
    assert math.isfinite(val)
